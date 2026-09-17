"""A fake Predictor for API tests, so tests don't load the real ~45MB backbone or
require a trained checkpoint to exist -- per the "mock model-heavy components" testing
guidance."""


class FakePredictor:
    model = object()  # truthy sentinel so /health reports classifier_loaded=True
    anomaly_detector = object()

    def predict(self, image_bytes: bytes, save_heatmap: bool = True) -> dict:
        return {
            "prediction": "bent",
            "confidence": 0.42,
            "uncertainty": 0.81,
            "margin": 0.05,
            "anomaly_score": 3.2,
            "needs_human_review": True,
            "heatmap_path": None,
            "model_version": "fake-v1",
        }
