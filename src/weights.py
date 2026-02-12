import os
from typing import Dict, Optional

import torch

from tnp import TransformerNeuralProcess


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _model_hparams(model: TransformerNeuralProcess) -> Dict[str, object]:
    return {
        "x_dim": model.x_dim,
        "y_dim": model.y_dim,
        "dim_model": model.dim_model,
        "embedder_depth": model.embedder_depth,
        "predictor_depth": model.predictor_depth,
        "num_heads": model.num_heads,
        "encoder_depth": model.encoder_depth,
        "dim_feedforward": model.dim_feedforward,
        "dropout": model.dropout,
    }


def save_model(model: TransformerNeuralProcess, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "hparams": _model_hparams(model)
    }
    torch.save(payload, path)


def load_model(path: str, device: Optional[str] = None) -> TransformerNeuralProcess:
    if device is None:
        device = _default_device()

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "hparams" not in payload:
        raise ValueError(
            "Weights file does not contain hyperparameters. "
            "Re-save the model with save_model to include them."
        )

    hparams = payload["hparams"]
    state_dict = payload["state_dict"]

    model = TransformerNeuralProcess(**hparams)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_state_dict(
    model: TransformerNeuralProcess,
    path: str,
    strict: bool = True
) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "hparams" in payload:
        hparams = payload["hparams"]
        if strict:
            current = _model_hparams(model)
            mismatched = {
                key: (current.get(key), hparams.get(key))
                for key in hparams.keys()
                if current.get(key) != hparams.get(key)
            }
            if mismatched:
                details = ", ".join(
                    f"{key} (model={values[0]}, weights={values[1]})"
                    for key, values in mismatched.items()
                )
                raise ValueError(
                    f"Model hyperparameters do not match weights: {details}"
                )
        model.load_state_dict(payload["state_dict"])
        return

    model.load_state_dict(payload)
