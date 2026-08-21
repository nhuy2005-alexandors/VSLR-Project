from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


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


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[GestureLSTM, list[str], dict]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = GestureLSTM(
        input_dim=int(config["input_dim"]),
        num_classes=len(checkpoint["labels"]),
        hidden_size=int(config.get("hidden_size", 96)),
        num_layers=int(config.get("num_layers", 1)),
        bidirectional=bool(config.get("bidirectional", True)),
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
