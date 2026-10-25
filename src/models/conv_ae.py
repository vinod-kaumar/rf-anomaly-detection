# FILE: src/models/conv_ae.py
# What this file does:
#   - ConvAEDetector: a convolutional autoencoder for the 32x32 log-spectrogram
#     pictures made by src/dsp/spectro.py (input: [n, 1024], reshaped to
#     32 time x 32 frequency).
#   - Encoder: 3 strided conv layers, then a narrow "bottleneck" layer (default
#     16 numbers). Decoder mirrors it. Trained ONLY on normal windows.
#   - 10% of the training windows are held out for early stopping on
#     reconstruction loss; the best weights are restored.
#   - error_map(): squared error per pixel [n, 32, 32] (for heatmaps).
#   - score(mode="mean"): mean squared error over the picture.
#     score(mode="max"): largest error after averaging over 4x4 regions, so a
#     small local anomaly is not diluted. HIGHER = MORE ANOMALOUS.

import copy

import numpy as np
import torch
from torch import nn

SIDE = 32


def _net(bottleneck: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(1024, bottleneck),
        nn.Linear(bottleneck, 1024), nn.ReLU(), nn.Unflatten(1, (64, 4, 4)),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 4, 2, 1),
    )


class ConvAEDetector:
    def __init__(self, bottleneck: int = 16, seed: int = 0, epochs: int = 60,
                 batch: int = 128, lr: float = 1e-3, patience: int = 8):
        self.bottleneck, self.seed, self.epochs = bottleneck, seed, epochs
        self.batch, self.lr, self.patience = batch, lr, patience
        self.net = None
        self.epochs_run = 0

    def _prep(self, x) -> torch.Tensor:
        z = (np.asarray(x, dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(z.reshape(-1, 1, SIDE, SIDE))

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        self.mean, self.std = float(x.mean()), float(x.std() + 1e-8)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(x))
        n_hold = max(1, len(x) // 10)
        hold, trn = self._prep(x[order[:n_hold]]), self._prep(x[order[n_hold:]])
        self.net = _net(self.bottleneck)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(trn))
            for i in range(0, len(trn), self.batch):
                b = trn[perm[i:i + self.batch]]
                opt.zero_grad()
                loss_fn(self.net(b), b).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(hold), hold).item()
            self.epochs_run = ep + 1
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def error_map(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("call fit() before score()")
        z = self._prep(x)
        out = []
        with torch.no_grad():
            for i in range(0, len(z), 512):
                b = z[i:i + 512]
                out.append(((self.net(b) - b) ** 2).squeeze(1).numpy())
        return np.concatenate(out)

    def score(self, x: np.ndarray, mode: str = "mean") -> np.ndarray:
        em = self.error_map(x)
        if mode == "mean":
            return em.mean(axis=(1, 2))
        if mode == "max":
            n = len(em)
            return em.reshape(n, 8, 4, 8, 4).mean(axis=(2, 4)).reshape(n, -1).max(axis=1)
        raise ValueError("mode must be 'mean' or 'max'")