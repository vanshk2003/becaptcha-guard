"""Deep behavioural model: 1D-CNN feature extractor + BiLSTM + attention pooling.

The trajectory is a variable-shape time series; a 1D-CNN captures local
kinematic motifs (acceleration bursts, tremor), the BiLSTM captures longer-range
temporal structure (sub-movements, pauses), and additive attention pools the
sequence into a session embedding for binary human/bot classification.

Trains on CPU in well under a minute on the synthetic set; swap in real
telemetry and the same architecture applies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BehaviorNet(nn.Module):
    def __init__(self, in_ch: int = 4, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, return_attn: bool = False):   # x: (B, L, C)
        h = self.conv(x.transpose(1, 2))     # (B, 64, L)
        h = h.transpose(1, 2)                # (B, L, 64)
        h, _ = self.lstm(h)                  # (B, L, 2H)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # (B, L)
        pooled = (h * w.unsqueeze(-1)).sum(1)               # (B, 2H)
        logits = self.head(pooled)
        if return_attn:
            return logits, w
        return logits


def _to_binary(y):
    return (y != 0).astype(np.int64)  # 0 human, 1 bot(any family)


def train_deep(Xtr, ytr, Xva, yva, cfg, device="cpu"):
    torch.manual_seed(cfg.seed)
    ytr_b, yva_b = _to_binary(ytr), _to_binary(yva)
    model = BehaviorNet(in_ch=Xtr.shape[2]).to(device)

    # class weights (bots are the majority of the non-human set; humans dominate overall)
    counts = np.bincount(ytr_b, minlength=2).astype(float)
    w = torch.tensor(counts.sum() / (2 * counts + 1e-9), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr_b)),
                    batch_size=cfg.batch_size, shuffle=True)
    best_state, best_va = None, -1.0
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        # validation accuracy for early-stop selection
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(Xva).to(device)).argmax(1).cpu().numpy()
        acc = float((pv == yva_b).mean())
        if acc > best_va:
            best_va, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best_va


@torch.no_grad()
def predict_proba_deep(model, X, device="cpu"):
    model.eval()
    logits = model(torch.tensor(np.asarray(X, np.float32)).to(device))
    return torch.softmax(logits, 1).cpu().numpy()[:, 1]  # p(bot)
