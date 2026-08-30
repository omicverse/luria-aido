"""The 15 locked metric cells (RULER.md) with anchors and controls.

Conventions follow the anchor papers:
  F1 (Norman 2019 / GEARS): held-out perturbation MSE + Pearson-delta, plus
      GEARS combo splits combo_seen0 / combo_seen2.
  F2 (sci-Plex3 / chemCPA): held-out compound R^2, all genes and top-50 DE,
      per cell line (A549 / K562 / MCF7).
  F3 (GenerTeam): DeepSTARR Dev/Hk Pearson, gener-tasks accuracy, variant
      AUROC - official test splits, never our own.

Controls are FUNCTIONS returning the same metric on the same rows, so they
appear as table rows, not footnotes:
  - training-mean predictor (F1 mandatory control)
  - ECFP4 nearest-neighbour (F2)
  - HVG+PCA baseline (F1/F2)
  - constant predictor
  - random-embedding (same shapes)
  - equal-budget shuffled-label null (best over K configs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# F1 - GEARS-style metrics
# ---------------------------------------------------------------------------

def mse_all_genes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1-1: MSE over all genes, held-out perturbations."""
    return float(np.mean((y_true - y_pred) ** 2))


def mse_top20_de(y_true: np.ndarray, y_pred: np.ndarray, de_mask: np.ndarray) -> float:
    """F1-2: MSE restricted to the top-20 differentially expressed genes."""
    return float(np.mean((y_true[:, de_mask] - y_pred[:, de_mask]) ** 2))


def pearson_delta(y_true: np.ndarray, y_pred: np.ndarray, ctrl: np.ndarray) -> float:
    """F1-3/4: Pearson correlation of delta expression (GEARS/CPA convention).

    delta = perturbation - control, per gene; correlation across genes of the
    mean per-perturbation delta. Returns mean over held-out perturbations.
    """
    d_true = y_true - ctrl
    d_pred = y_pred - ctrl
    cors = []
    for t, p in zip(d_true, d_pred):
        if np.std(t) == 0 or np.std(p) == 0:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.mean(cors))


def split_combo_seen0(perturbations: pd.DataFrame) -> np.ndarray:
    """F1-5: GEARS combo_seen0 split - combos whose genes were never single-perturbed in train."""
    raise NotImplementedError("split logic is implemented in scoreboard/metrics_gears.py after data inspection")


def split_combo_seen2(perturbations: pd.DataFrame) -> np.ndarray:
    """F1-6: GEARS combo_seen2 split - combos whose genes were both single-perturbed in train."""
    raise NotImplementedError("split logic is implemented in scoreboard/metrics_gears.py after data inspection")


# ---------------------------------------------------------------------------
# F2 - chemCPA-style metrics
# ---------------------------------------------------------------------------

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def r2_top50_de(y_true: np.ndarray, y_pred: np.ndarray, de_mask: np.ndarray) -> float:
    """F2-8: R^2 restricted to top-50 DE genes."""
    return r2(y_true[:, de_mask], y_pred[:, de_mask])


# ---------------------------------------------------------------------------
# Controls (used as rows in the scoreboard)
# ---------------------------------------------------------------------------

class ConstantPredictor:
    """Predicts the training-mean response for every perturbation."""

    def fit(self, y_train: np.ndarray) -> "ConstantPredictor":
        self.value = np.mean(y_train, axis=0)
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.tile(self.value, (n, 1))


class TrainingMeanPredictor:
    """F1 mandatory control: mean of the TRAINING perturbations' responses."""

    def fit(self, y_train: np.ndarray) -> "TrainingMeanPredictor":
        self.value = np.mean(y_train, axis=0)
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.tile(self.value, (n, 1))


class RandomEmbeddingPredictor:
    """Same adapter architecture, random (frozen) embeddings of same shapes."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomEmbeddingPredictor":
        # linear regression on random projections: same shapes, no signal
        P = self.rng.standard_normal((X.shape[1], 64))
        Z = X @ P
        self.coef_, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._P @ self.coef_ if hasattr(self, "_P") else X[:, :64] @ self.coef_


def shuffled_label_null(score_fn, X, y, K: int, rng: np.random.Generator) -> list[float]:
    """Equal-budget shuffled-label null: best of K configs on permuted labels."""
    scores = []
    for _ in range(K):
        y_perm = rng.permutation(y)
        scores.append(score_fn(X, y_perm))
    return sorted(scores, reverse=True)
