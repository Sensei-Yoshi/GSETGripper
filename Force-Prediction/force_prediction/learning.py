"""Physics-residual learner (E6): predict r = F* - N_physics, then N̂ = N_physics + r̂.

Learning the residual (not the raw force) means the model does not have to
rediscover gravity, mass scaling, or the roughness/contact trends the physics
already captures — much easier on ~100 objects. Base features are physically
meaningful; a small PCA of the semantic embedding is appended. Fit inside each
training fold only.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge

from .config import Config


def base_features(mass_g: float, roughness_class: int, contact: float, physics_force: float) -> list[float]:
    return [math.log(mass_g), float(roughness_class), float(contact), float(physics_force)]


class ResidualForceModel:
    """Wraps a scikit-learn regressor + an embedding PCA."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pca: PCA | None = None
        self.model = self._make_model()

    def _make_model(self):  # noqa: ANN202
        lc = self.cfg.learning
        if lc.residual_model == "ridge":
            return Ridge(alpha=lc.ridge_alpha)
        if lc.residual_model == "gp":
            return GaussianProcessRegressor(normalize_y=True)
        return GradientBoostingRegressor(
            n_estimators=lc.gbt_n_estimators,
            max_depth=lc.gbt_max_depth,
            learning_rate=lc.gbt_learning_rate,
            random_state=self.cfg.seed,
        )

    def _assemble(self, base: np.ndarray, embeddings: np.ndarray, fit: bool) -> np.ndarray:
        dims = self.cfg.learning.embedding_pca_dims
        if dims <= 0 or embeddings.size == 0:
            return base
        n_comp = min(dims, embeddings.shape[1], max(1, embeddings.shape[0] - 1))
        if fit:
            self.pca = PCA(n_components=n_comp, random_state=self.cfg.seed).fit(embeddings)
        assert self.pca is not None
        return np.hstack([base, self.pca.transform(embeddings)])

    def fit(self, base: np.ndarray, embeddings: np.ndarray, residuals: np.ndarray) -> ResidualForceModel:
        x = self._assemble(base, embeddings, fit=True)
        self.model.fit(x, residuals)
        return self

    def predict_residual(self, base: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        x = self._assemble(base, embeddings, fit=False)
        return self.model.predict(x)
