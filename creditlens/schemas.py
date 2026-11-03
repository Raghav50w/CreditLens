"""Request shape and input checks for /api/predict."""

import math

from pydantic import BaseModel, ConfigDict

from creditlens.config import DATASET_META


class PredictRequest(BaseModel):
    """The JSON body of a prediction request.

    Feature values (balance, age, ...) differ per dataset, so they arrive as
    extra fields and are checked by `validate_features` against the bounds in
    DATASET_META.
    """

    model_config = ConfigDict(extra="allow")

    dataset: str
    model_type: str | None = None


def validate_features(dataset, features, values):
    """Check every feature against its DATASET_META bounds.

    Returns (numbers, errors). `numbers` is in the same order as `features`.
    Every problem is collected so a user fixing the form sees all of them at
    once, rather than one per submit.
    """
    numbers = []
    errors = {}

    for name in features:
        meta = DATASET_META[dataset]["features"][name]
        label = meta["label"]
        raw = values.get(name)

        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors[name] = f"{label} is required."
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue

        if not math.isfinite(number):
            errors[name] = f"{label} must be a finite number."
            continue

        unit = f" {meta['unit']}" if meta["unit"] else ""

        if number < meta["min"]:
            errors[name] = f"{label} must be at least {meta['min']:g}{unit}."
        elif number > meta["max"]:
            errors[name] = f"{label} must be at most {meta['max']:g}{unit}."

        numbers.append(number)

    return numbers, errors
