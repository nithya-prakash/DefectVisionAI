"""Simplified PatchCore-style anomaly detector.

What this implements (from Roth et al., "Towards Total Recall in Industrial Anomaly
Detection", CVPR 2022):
  - mid-level CNN patch embeddings as the feature space
  - a memory bank of normal-image patch embeddings
  - nearest-neighbor distance in that space as the anomaly score

What is simplified relative to the original paper (state this honestly everywhere this
class is discussed):
  - Memory bank subsampling uses uniform random sampling, not the paper's greedy
    minimax coreset selection.
  - No reweighting of the top-k nearest-neighbor score; this uses the single nearest
    neighbor per patch (k=1), not the paper's k-NN re-weighting scheme.
  - No sensor/domain adaptation tricks beyond ImageNet-pretrained features.

This is a *reference implementation for a portfolio project*, not a reproduction of the
paper's benchmark numbers.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from automotive_vision.models.embeddings import PatchFeatureExtractor


@dataclass
class AnomalyScores:
    image_scores: np.ndarray  # (N,) max-patch-distance anomaly score per image
    pixel_maps: np.ndarray  # (N, H, W) upsampled anomaly heatmap per image, in [0, inf)
    labels: np.ndarray  # (N,) 1 = anomalous, 0 = normal (ground truth)
    masks: np.ndarray  # (N, H, W) ground-truth binary masks (all-zero for normal images)
    paths: list[str]
    defect_types: list[str]


class PatchCoreLite:
    def __init__(
        self,
        image_size: int,
        memory_bank_size: int = 10_000,
        device: str = "cpu",
        seed: int = 42,
        nn_chunk_size: int = 4096,
    ):
        self.image_size = image_size
        self.memory_bank_size = memory_bank_size
        self.device = torch.device(device)
        self.seed = seed
        self.nn_chunk_size = nn_chunk_size
        self.extractor = PatchFeatureExtractor().to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.patch_grid_size: int | None = None

    @torch.no_grad()
    def fit(self, normal_loader) -> None:
        """Builds the memory bank from normal-only training images."""
        all_patches = []
        for batch in normal_loader:
            images = batch["image"].to(self.device)
            patch_map = self.extractor(images)  # (B, C, H, W)
            b, c, h, w = patch_map.shape
            self.patch_grid_size = h
            patches = patch_map.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, C)
            all_patches.append(patches.cpu())

        bank = torch.cat(all_patches, dim=0)
        if bank.shape[0] > self.memory_bank_size:
            generator = torch.Generator().manual_seed(self.seed)
            idx = torch.randperm(bank.shape[0], generator=generator)[: self.memory_bank_size]
            bank = bank[idx]
        self.memory_bank = bank.to(self.device)

    @torch.no_grad()
    def _nearest_neighbor_distance(self, query_patches: torch.Tensor) -> torch.Tensor:
        """Chunked min-distance search so we never materialize the full (N, M) matrix."""
        min_dist = torch.full((query_patches.shape[0],), float("inf"), device=self.device)
        for start in range(0, self.memory_bank.shape[0], self.nn_chunk_size):
            chunk = self.memory_bank[start : start + self.nn_chunk_size]
            dist = torch.cdist(query_patches, chunk)  # (N, chunk_size)
            min_dist = torch.minimum(min_dist, dist.min(dim=1).values)
        return min_dist

    def save(self, path: str | Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before save().")
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "image_size": self.image_size,
                "memory_bank_size": self.memory_bank_size,
                "seed": self.seed,
                "nn_chunk_size": self.nn_chunk_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCoreLite":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        detector = cls(
            image_size=checkpoint["image_size"],
            memory_bank_size=checkpoint["memory_bank_size"],
            device=device,
            seed=checkpoint["seed"],
            nn_chunk_size=checkpoint["nn_chunk_size"],
        )
        detector.memory_bank = checkpoint["memory_bank"].to(detector.device)
        return detector

    @torch.no_grad()
    def score_image(self, image_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        """Scores a single preprocessed image (C, H, W); returns (image_score, pixel_map)."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() or load() before score_image().")
        batch = image_tensor.unsqueeze(0).to(self.device)
        patch_map = self.extractor(batch)
        _, c, h, w = patch_map.shape
        patches = patch_map.permute(0, 2, 3, 1).reshape(h * w, c)
        dist = self._nearest_neighbor_distance(patches)
        score_map = dist.reshape(1, 1, h, w)
        upsampled = F.interpolate(score_map, size=(self.image_size, self.image_size), mode="bilinear")[0, 0]
        return float(dist.max().item()), upsampled.cpu().numpy()

    @torch.no_grad()
    def score_dataset(self, loader) -> AnomalyScores:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before score_dataset().")

        image_scores, pixel_maps, labels, masks, paths, defect_types = [], [], [], [], [], []

        for batch in loader:
            images = batch["image"].to(self.device)
            patch_map = self.extractor(images)
            b, c, h, w = patch_map.shape
            patches = patch_map.permute(0, 2, 3, 1).reshape(b, h * w, c)

            for i in range(b):
                dist = self._nearest_neighbor_distance(patches[i])  # (H*W,)
                score_map = dist.reshape(1, 1, h, w)
                upsampled = F.interpolate(
                    score_map, size=(self.image_size, self.image_size), mode="bilinear"
                )[0, 0]
                pixel_maps.append(upsampled.cpu().numpy())
                image_scores.append(dist.max().item())

            labels.extend(batch["label"].numpy().tolist())
            masks.extend(batch["mask"].numpy())
            paths.extend(batch["path"])
            defect_types.extend(batch["defect_type"])

        return AnomalyScores(
            image_scores=np.array(image_scores),
            pixel_maps=np.stack(pixel_maps),
            labels=np.array(labels),
            masks=np.stack(masks),
            paths=paths,
            defect_types=defect_types,
        )
