"""Reduced-order physics prior for the two grippers (James et al., IEEE #11522915).

The physics supplies a deterministic numerical force prior; experiences and the
VLM correct it. Coefficients are SMOOTH and MONOTONE in roughness class so we fit
~5-6 parameters instead of 16 free per-class values (which would overfit ~100
objects with sparse class coverage).

Model (holding / tangential capacity as a function of applied normal force N):

    silicone:  T_sil(N) = alpha_sil(c) * a * N
    gecko:     T_geo(N) = alpha_geo(c) * a * N + beta(c) * a * N/(N + N50)

with
    alpha_*(c) = alpha0 * exp(-decay * (c - 1))     (friction, decays with roughness)
    beta(c)    = beta0  * clip(1 - beta_decay*(c-1), 0, 1)   (adhesion, decays with roughness)

An object of weight W = (mass_g/1000)*g lifts when T(N) >= W. The minimum feasible
N is found in closed form (silicone) or by a robust root solve (gecko). Feasibility
is physics-driven: infeasible if no N <= force.limit_n satisfies the constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq, least_squares

from .config import Config
from .contracts import ExperienceRecord, Gripper


@dataclass
class PhysicsParams:
    """Calibrated coefficients. Defaults come from config; refit inside each fold."""

    alpha_sil0: float
    alpha_sil_decay: float
    alpha_geo0: float
    alpha_geo_decay: float
    beta0: float
    beta_decay: float
    n50: float

    @classmethod
    def from_config(cls, cfg: Config) -> PhysicsParams:
        p = cfg.physics
        return cls(
            alpha_sil0=p.alpha_sil0,
            alpha_sil_decay=p.alpha_sil_decay,
            alpha_geo0=p.alpha_geo0,
            alpha_geo_decay=p.alpha_geo_decay,
            beta0=p.beta0,
            beta_decay=p.beta_decay,
            n50=p.n50,
        )


@dataclass
class PhysicsEstimate:
    gripper: Gripper
    feasible: bool
    min_force_n: float | None  # continuous command, clamped to hardware range
    raw_force_n: float | None  # unconstrained continuous solve; None if infeasible


def weight_n(mass_g: float, gravity: float) -> float:
    return (mass_g / 1000.0) * gravity


class PhysicsModel:
    """Evaluates holding capacity and solves for the minimum feasible normal force."""

    def __init__(self, params: PhysicsParams, cfg: Config) -> None:
        self.p = params
        self.cfg = cfg

    # --- coefficient shapes (smooth, monotone in class) --------------------- #
    def alpha_sil(self, c: int) -> float:
        return self.p.alpha_sil0 * math.exp(-self.p.alpha_sil_decay * (c - 1))

    def alpha_geo(self, c: int) -> float:
        return self.p.alpha_geo0 * math.exp(-self.p.alpha_geo_decay * (c - 1))

    def beta(self, c: int) -> float:
        return self.p.beta0 * max(0.0, min(1.0, 1.0 - self.p.beta_decay * (c - 1)))

    # --- holding capacity T(N) --------------------------------------------- #
    def holding_force(self, gripper: Gripper, n: float, c: int, a: float) -> float:
        if gripper is Gripper.SILICONE:
            return self.alpha_sil(c) * a * n
        return self.alpha_geo(c) * a * n + self.beta(c) * a * n / (n + self.p.n50)

    # --- minimum feasible normal force ------------------------------------- #
    def min_force(self, gripper: Gripper, mass_g: float, c: int, a: float) -> PhysicsEstimate:
        limit = self.cfg.force.limit_n
        w = weight_n(mass_g, self.cfg.force.gravity)

        if a <= 0.0:  # no projected contact -> cannot support any load
            return PhysicsEstimate(gripper, feasible=False, min_force_n=None, raw_force_n=None)

        if gripper is Gripper.SILICONE:
            denom = self.alpha_sil(c) * a
            raw = w / denom if denom > 0 else math.inf
        else:
            if self.holding_force(gripper, limit, c, a) < w:
                raw = math.inf
            else:
                raw = brentq(lambda n: self.holding_force(gripper, n, c, a) - w, 0.0, limit)

        if not math.isfinite(raw) or raw > limit:
            return PhysicsEstimate(gripper, feasible=False, min_force_n=None, raw_force_n=None)

        command = max(self.cfg.force.min_n, raw)
        return PhysicsEstimate(gripper, feasible=True, min_force_n=command, raw_force_n=raw)


# --------------------------------------------------------------------------- #
# Calibration — fit coefficients to feasible training records (inside a fold).
# --------------------------------------------------------------------------- #

def calibrate(records: list[ExperienceRecord], cfg: Config) -> PhysicsParams:
    """Fit smooth-in-class coefficients so predicted T(N*) matches weight at the
    observed minimum force N*. Silicone (2 params) and gecko (5 params) are fit
    independently. Falls back to config defaults for a gripper with too few points.
    """
    defaults = PhysicsParams.from_config(cfg)
    g = cfg.force.gravity
    b = cfg.physics.bounds

    sil = [r for r in records if r.gripper is Gripper.SILICONE and r.feasible and r.min_force_n]
    geo = [r for r in records if r.gripper is Gripper.GECKO and r.feasible and r.min_force_n]

    # --- silicone: residual_i = alpha_sil(c)*a*N* - W ---------------------- #
    if len(sil) >= 2:
        def sil_resid(x: list[float]) -> list[float]:
            a0, dec = x
            out = []
            for r in sil:
                assert r.min_force_n is not None
                alpha = a0 * math.exp(-dec * (r.roughness_class - 1))
                pred = alpha * r.projected_contact_fraction * r.min_force_n
                out.append(pred - weight_n(r.mass_g, g))
            return out

        res = least_squares(
            sil_resid,
            x0=[defaults.alpha_sil0, defaults.alpha_sil_decay],
            bounds=([b.alpha0[0], b.decay[0]], [b.alpha0[1], b.decay[1]]),
        )
        defaults.alpha_sil0, defaults.alpha_sil_decay = float(res.x[0]), float(res.x[1])

    # --- gecko: residual_i = alpha_geo(c)*a*N* + beta(c)*a*N*/(N*+n50) - W -- #
    if len(geo) >= 3:
        def geo_resid(x: list[float]) -> list[float]:
            a0, dec, be0, be_dec, n50 = x
            out = []
            for r in geo:
                assert r.min_force_n is not None
                c, a, n = r.roughness_class, r.projected_contact_fraction, r.min_force_n
                alpha = a0 * math.exp(-dec * (c - 1))
                beta = be0 * max(0.0, min(1.0, 1.0 - be_dec * (c - 1)))
                pred = alpha * a * n + beta * a * n / (n + n50)
                out.append(pred - weight_n(r.mass_g, g))
            return out

        res = least_squares(
            geo_resid,
            x0=[defaults.alpha_geo0, defaults.alpha_geo_decay, defaults.beta0,
                defaults.beta_decay, defaults.n50],
            bounds=(
                [b.alpha0[0], b.decay[0], b.beta0[0], b.beta_decay[0], b.n50[0]],
                [b.alpha0[1], b.decay[1], b.beta0[1], b.beta_decay[1], b.n50[1]],
            ),
        )
        (defaults.alpha_geo0, defaults.alpha_geo_decay, defaults.beta0,
         defaults.beta_decay, defaults.n50) = (float(v) for v in res.x)

    return defaults
