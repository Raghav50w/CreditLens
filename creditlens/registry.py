"""Loads the trained models from disk and runs predictions."""

import json
import pickle
from pathlib import Path

from creditlens.config import (
    DATASET_META,
    DEFAULT_MODEL,
    METRICS_FILENAME,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
)


class ModelRegistry:
    """Holds every trained model in memory so predictions don't reload files."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.datasets = {}
        self.metrics = {}

    def load(self):
        """Read the artifacts for every dataset listed in DATASET_META."""
        for dataset in DATASET_META:
            models = {}
            for model_type in MODEL_DISPLAY_NAMES:
                model_path = self.models_dir / f"{dataset}_{model_type}.pkl"
                with open(model_path, "rb") as f:
                    models[model_type] = pickle.load(f)

            self.datasets[dataset] = {
                "features": list(DATASET_META[dataset]["features"]),
                "models": models,
            }

        metrics_path = self.models_dir / METRICS_FILENAME
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def catalog(self):
        """The /api/datasets payload: everything the frontend needs to draw a form."""
        catalog = {}

        for dataset, artifacts in self.datasets.items():
            meta = DATASET_META[dataset]

            features = []
            for name in artifacts["features"]:
                feature = {"name": name}
                feature.update(meta["features"][name])
                features.append(feature)

            models = []
            for model_type in sorted(artifacts["models"]):
                models.append(
                    {
                        "name": model_type,
                        "display_name": MODEL_DISPLAY_NAMES[model_type],
                    }
                )

            catalog[dataset] = {
                "display_name": meta["display_name"],
                "description": meta["description"],
                "features": features,
                "default_model": DEFAULT_MODEL,
                "models": models,
            }

        return {"datasets": catalog}

    def predict(self, dataset, model_type, values):
        """Run one prediction. `values` must already be validated, in feature order."""
        # Each model is a full pipeline: it scales and encodes the raw values
        # itself, exactly as it did in training.
        model = self.datasets[dataset]["models"][model_type]

        prediction = int(model.predict([values])[0])

        positive = model.predict_proba([values])[0][1]
        probability = round(float(positive) * 100, 1)

        # Whether a positive prediction is good news depends on the dataset:
        # an accepted offer is, a default is not. The UI colours on `outcome`.
        meta = DATASET_META[dataset]
        if prediction == 1:
            summary = meta["positive_label"]
            good = meta["positive_is_good"]
        else:
            summary = meta["negative_label"]
            good = not meta["positive_is_good"]

        return {
            "dataset": dataset,
            "model_used": model_type,
            "prediction": prediction,
            "outcome": "good" if good else "bad",
            "summary": summary,
            "probability": probability,
        }
