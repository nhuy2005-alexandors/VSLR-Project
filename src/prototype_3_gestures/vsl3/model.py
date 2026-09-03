from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_CONTRACT, FEATURE_DIM, FEATURES_VERSION, SEQUENCE_LENGTH
from .reject import load_reject_policy


POOLING_FWD_LAST_BWD_FIRST = "fwd_last_bwd_first"
POOLING_LEGACY_LAST_STEP = "legacy_last_step"
POOLING_MODES = (POOLING_FWD_LAST_BWD_FIRST, POOLING_LEGACY_LAST_STEP)

# Bump when the network topology or its fixed dropout/head defaults change. Persisting this beside
# the concrete dimensions keeps a LOSO report from being paired with weights made by another model.
MODEL_ARCHITECTURE_VERSION = 1
REJECTION_CONTRACT = "closed-set-reject-v1-calibration-only"
SEGMENTATION_CONTRACT = "timestamp-segment-v2-min-active-no-tail"
DEFAULT_HIDDEN_SIZE = 96
DEFAULT_NUM_LAYERS = 1
DEFAULT_BIDIRECTIONAL = True


class GestureLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        bidirectional: bool = DEFAULT_BIDIRECTIONAL,
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
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "model_state": model.state_dict(),
                "labels": labels,
                "config": config,
            },
            staging,
        )
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[GestureLSTM, list[str], dict]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "labels" not in checkpoint:
        raise ValueError(f"{path} is not a VSLR checkpoint with config and labels")
    config = checkpoint["config"]
    if (
        not isinstance(config, dict)
        or not isinstance(checkpoint["labels"], list)
        or not checkpoint["labels"]
        or not all(isinstance(label, str) and label for label in checkpoint["labels"])
        or len(set(checkpoint["labels"])) != len(checkpoint["labels"])
    ):
        raise ValueError(f"{path} has an invalid checkpoint config or empty labels")

    # Missing means the checkpoint predates this key but uses the original topology (= v1).
    # A future/same-shape architecture can change semantics without tripping load_state_dict.
    stored_architecture_version = int(config.get("model_architecture_version", 1))
    if stored_architecture_version != MODEL_ARCHITECTURE_VERSION:
        raise ValueError(
            f"{path} uses model architecture version {stored_architecture_version}, but this install "
            f"implements version {MODEL_ARCHITECTURE_VERSION}. Refusing to interpret those weights "
            "with different model architecture semantics."
        )

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
        raise ValueError(message + " Refusing inference; retrain the checkpoint with this feature contract.")

    stored_feature_contract = config.get("feature_contract")
    if stored_features_version == FEATURES_VERSION and stored_feature_contract != FEATURE_CONTRACT:
        raise ValueError(
            f"{path} has feature_contract={stored_feature_contract!r}, expected {FEATURE_CONTRACT!r}. "
            "Refusing to interpret a same-version checkpoint with unknown preprocessing semantics."
        )
    stored_input_dim = int(config.get("input_dim", -1))
    if stored_input_dim != FEATURE_DIM:
        raise ValueError(
            f"{path} has input_dim={stored_input_dim}, but this install extracts FEATURE_DIM={FEATURE_DIM}. "
            "Retrain the checkpoint."
        )
    if "sequence_length" not in config and stored_features_version == FEATURES_VERSION:
        raise ValueError(f"{path} is missing checkpoint sequence_length; refusing incomplete metadata")
    stored_sequence_length = int(config.get("sequence_length", SEQUENCE_LENGTH))
    if stored_sequence_length != SEQUENCE_LENGTH:
        raise ValueError(
            f"{path} has sequence_length={stored_sequence_length}, but this install uses "
            f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}. Retrain the checkpoint."
        )
    if stored_features_version == FEATURES_VERSION:
        required_metadata = {
            "rejection_contract": REJECTION_CONTRACT,
            "segmentation_contract": SEGMENTATION_CONTRACT,
            "training_signature": None,
            "recording_plan": None,
            "reject_policy": None,
        }
        for key, expected in required_metadata.items():
            if key not in config:
                raise ValueError(f"{path} is missing checkpoint metadata field {key!r}")
            value = config[key]
            if expected is not None and value != expected:
                raise ValueError(f"{path} has incompatible checkpoint {key}={value!r}")
            if key == "training_signature" and (not isinstance(value, str) or not value):
                raise ValueError(f"{path} has invalid checkpoint training_signature")
            if key == "recording_plan" and not isinstance(value, dict):
                raise ValueError(f"{path} has invalid checkpoint recording_plan metadata")
    if "reject_policy" in config:
        try:
            load_reject_policy(config["reject_policy"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} has invalid reject_policy metadata: {exc}") from exc

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
        input_dim=stored_input_dim,
        num_classes=len(checkpoint["labels"]),
        hidden_size=int(config.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
        num_layers=int(config.get("num_layers", DEFAULT_NUM_LAYERS)),
        bidirectional=bool(config.get("bidirectional", DEFAULT_BIDIRECTIONAL)),
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
