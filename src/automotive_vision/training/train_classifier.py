"""Training loop for the supervised DefectClassifier.

Early stopping and LR scheduling are both driven by validation macro-F1 (not
accuracy), since the labeled pool is small and per-class support is uneven enough that
accuracy alone can hide a collapsed minority class.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from automotive_vision.models.classifier import DefectClassifier
from automotive_vision.training.evaluate import classification_metrics


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = 0.0


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0  # avoid div-by-zero for classes absent from this split
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw logits + labels for an entire loader, for calibration/temperature-scaling."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        all_logits.append(model(images).cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def evaluate_model(model, loader, device, class_names) -> tuple[float, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = classification_metrics(np.array(all_labels), np.array(all_preds), class_names)
    return avg_loss, metrics


def train_classifier(
    train_dataset,
    val_dataset,
    class_names: list[str],
    config: dict,
    device: str,
) -> tuple[DefectClassifier, TrainingHistory]:
    model = DefectClassifier(
        num_classes=len(class_names),
        freeze_until_layer4=config["freeze_until_layer4"],
        dropout_p=config["dropout_p"],
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    if config["use_class_weights"]:
        train_labels = [train_dataset.label_to_idx[r.defect_type] for r in train_dataset.records]
        class_weights = compute_class_weights(train_labels, len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["lr_scheduler_factor"], patience=config["lr_scheduler_patience"]
    )

    history = TrainingHistory()
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_dataset)
        val_loss, val_metrics = evaluate_model(model, val_loader, device, class_names)
        val_macro_f1 = val_metrics["macro_f1"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_macro_f1.append(val_macro_f1)
        scheduler.step(val_macro_f1)

        if val_macro_f1 > history.best_val_macro_f1:
            history.best_val_macro_f1 = val_macro_f1
            history.best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["early_stopping_patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
