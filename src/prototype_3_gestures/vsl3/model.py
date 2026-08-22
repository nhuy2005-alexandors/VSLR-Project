from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURES_VERSION


POOLING_FWD_LAST_BWD_FIRST = "fwd_last_bwd_first"
POOLING_LEGACY_LAST_STEP = "legacy_last_step"
POOLING_MODES = (POOLING_FWD_LAST_BWD_FIRST, POOLING_LEGACY_LAST_STEP)


class GestureLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        bidirectional: bool = True,
        pooling: str = POOLING_FWD_LAST_BWD_FIRST,
    ):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unknown pooling {pooling!r}; expected one of {POOLING_MODES}")
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else 0.15,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def pool_sequence(self, out: torch.Tensor) -> torch.Tensor:
        """Collapse [batch, time, 2*hidden] to [batch, 2*hidden].

        On a bidirectional LSTM the reverse direction runs from the last frame backwards, so
        `out[:, -1, hidden:]` has only consumed a single frame — taking both halves at t=-1
        wastes half the representation. Forward's summary is at t=-1, backward's at t=0.
        """
        if not self.bidirectional or self.pooling == POOLING_LEGACY_LAST_STEP:
            return out[:, -1, :]
        hidden = self.lstm.hidden_size
        return torch.cat([out[:, -1, :hidden], out[:, 0, hidden:]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        return self.head(self.pool_sequence(out))


def save_checkpoint(path: str | Path, model: GestureLSTM, labels: list[str], config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    allow_incompatible_features: bool = False,
) -> tuple[GestureLSTM, list[str], dict]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    # A checkpoint with no "features_version" key was written before the key existed, which means
    # version 1. Feeding it v2 landmarks changes the model input distribution without changing a
    # tensor shape, so load_state_dict cannot catch the incompatibility.
    stored_features_version = int(config.get("features_version", 1))
    if stored_features_version != FEATURES_VERSION:
        message = (
            f"{path} was trained on landmarks at FEATURES_VERSION={stored_features_version}"
            f"{' (no key: assumed 1)' if 'features_version' not in config else ''}, but this install "
            f"extracts version {FEATURES_VERSION}. Predictions are unreliable until you retrain."
        )
        if not allow_incompatible_features:
            raise ValueError(
                message
                + " Refusing inference by default; retrain the checkpoint or explicitly opt in to "
                "incompatible features for a temporary legacy demo."
            )
        warnings.warn(message, stacklevel=2)

    pooling = config.get("pooling")
    if pooling is None:
        # ponytail: checkpoints written before the pooling fix carry no key. Legacy is the correct
        # default for them, but pooling changes predictions without changing any tensor shape, so
        # load_state_dict cannot detect a wrong guess — say so out loud rather than assume silently.
        pooling = POOLING_LEGACY_LAST_STEP
        warnings.warn(
            f"{path} has no 'pooling' key, loading it as {POOLING_LEGACY_LAST_STEP!r}. Correct for a "
            "checkpoint trained before the bidirectional pooling fix; retrain to get an explicit key.",
            stacklevel=2,
        )

    model = GestureLSTM(
        input_dim=int(config["input_dim"]),
        num_classes=len(checkpoint["labels"]),
        hidden_size=int(config.get("hidden_size", 96)),
        num_layers=int(config.get("num_layers", 1)),
        bidirectional=bool(config.get("bidirectional", True)),
        pooling=str(pooling),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, list(checkpoint["labels"]), config


def predict_sequence(
    model: GestureLSTM,
    sequence: np.ndarray,
    labels: list[str],
    device: torch.device | str,
) -> tuple[str, float, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = torch.max(probabilities, dim=0)
    return labels[int(index)], float(confidence), probabilities.detach().cpu().numpy()
