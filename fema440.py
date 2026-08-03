"""FEMA 440 preliminary pushover post-processing helpers.

The routines here intentionally provide capacity-curve based screening values.
They do not replace a complete FEMA 440 performance-point calculation, which
requires a project/design response spectrum and an SDOF transformation.
"""

from __future__ import annotations

import math
from typing import Any


SOIL_TO_COEFFICIENT_SITE: dict[str, tuple[str, float]] = {
    "ZA": ("B", 130.0),
    "ZB": ("B", 130.0),
    "ZC": ("C", 90.0),
    "ZD": ("D", 60.0),
    "ZE": ("E", 60.0),
}


def calculate_fema440_by_direction(model: dict[str, Any]) -> dict[str, Any]:
    """Return FEMA 440 preliminary values for every available pushover direction."""
    curves = model.get("curves", {}) if isinstance(model.get("curves"), dict) else {}
    return {
        str(direction).upper(): calculate_fema440_for_curve(
            curve,
            story_count=_to_float(model.get("story_count")),
            soil_class=str(model.get("soil_class") or ""),
        )
        for direction, curve in curves.items()
        if isinstance(curve, dict)
    }


def calculate_fema440_for_curve(curve: dict[str, Any], story_count: float | None, soil_class: str) -> dict[str, Any]:
    """Compute approximate FEMA 440 EL and displacement-modification values."""
    points = _capacity_points(curve.get("points", []))
    if len(points) < 2:
        return {
            "available": False,
            "message": "Insufficient capacity-curve points; FEMA 440 preliminary assessment could not be performed.",
        }

    ideal = _idealize_capacity(points)
    if not ideal["available"]:
        return ideal

    mu = max(1.0, ideal["ultimate_displacement_m"] / ideal["yield_displacement_m"])
    t0 = _initial_period_proxy(story_count)
    teff = _effective_period(t0, mu)
    beta_eff = _effective_damping_percent(mu)
    beta_reduction = _damping_reduction_factor(beta_eff)
    site_class, site_a = SOIL_TO_COEFFICIENT_SITE.get(soil_class.upper(), ("C", 90.0))
    r_proxy = max(1.0, ideal["peak_base_shear_kn"] / max(ideal["yield_base_shear_kn"], 1e-9))
    c0 = 1.0
    c1 = _coefficient_c1(r_proxy, t0, site_a)
    c2 = 1.0
    c3 = 1.0
    elastic_displacement_proxy = ideal["yield_displacement_m"] * r_proxy
    target_displacement_proxy = c0 * c1 * c2 * c3 * elastic_displacement_proxy

    return {
        "available": True,
        "method_note": "Preliminary assessment: because no demand spectrum or modal participation factor is available, capacity-curve-based FEMA 440 indicators are computed rather than a formal performance point.",
        "idealization": ideal,
        "equivalent_linearization": {
            "ductility_mu": round(mu, 4),
            "initial_period_proxy_s": round(t0, 4),
            "effective_period_teff_s": round(teff, 4),
            "effective_period_ratio_teff_t0": round(teff / t0, 4) if t0 else None,
            "effective_damping_beta_percent": round(beta_eff, 4),
            "damping_reduction_factor_b_beta": round(beta_reduction, 4),
            "performance_displacement_m": None,
            "message": "A 5% damped demand spectrum is required for an actual FEMA 440 EL performance point.",
        },
        "displacement_modification": {
            "coefficient_site_class": site_class,
            "site_coefficient_a": site_a,
            "r_capacity_proxy": round(r_proxy, 4),
            "c0": c0,
            "c1": round(c1, 4),
            "c2": c2,
            "c3": c3,
            "elastic_displacement_proxy_m": round(elastic_displacement_proxy, 5),
            "target_displacement_proxy_m": round(target_displacement_proxy, 5),
            "message": "This value is derived from capacity-curve proxy values rather than spectral demand; it is not a final FEMA 440 target displacement.",
        },
    }


def _capacity_points(raw_points: object) -> list[dict[str, float]]:
    """Return sorted nonnegative displacement/base-shear pairs."""
    if not isinstance(raw_points, list):
        return []
    points: list[dict[str, float]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        d = _to_float(item.get("control_displacement_m"))
        v = _to_float(item.get("base_shear_kn"))
        step = _to_float(item.get("step_number") or item.get("step"))
        if d is None or v is None:
            continue
        if d < 0:
            continue
        points.append({"step_number": step if step is not None else 0.0, "displacement_m": d, "base_shear_kn": abs(v)})
    points.sort(key=lambda row: (row["displacement_m"], row["step_number"]))
    unique: list[dict[str, float]] = []
    for point in points:
        if unique and abs(unique[-1]["displacement_m"] - point["displacement_m"]) < 1e-9:
            if point["base_shear_kn"] > unique[-1]["base_shear_kn"]:
                unique[-1] = point
        else:
            unique.append(point)
    return unique


def _idealize_capacity(points: list[dict[str, float]]) -> dict[str, Any]:
    """Idealize a capacity curve using a stiffness-degradation yield proxy."""
    positive = [point for point in points if point["displacement_m"] > 1e-9 and point["base_shear_kn"] > 1e-9]
    if len(positive) < 2:
        return {"available": False, "message": "Insufficient positive capacity-curve points."}

    initial = positive[0]
    initial_stiffness = initial["base_shear_kn"] / initial["displacement_m"]
    peak = max(positive, key=lambda row: row["base_shear_kn"])
    ultimate = positive[-1]
    yield_point = None
    for point in positive[1:]:
        secant = point["base_shear_kn"] / max(point["displacement_m"], 1e-9)
        if point["base_shear_kn"] >= 0.6 * peak["base_shear_kn"] and secant <= 0.6 * initial_stiffness:
            yield_point = point
            break
    if yield_point is None:
        yield_shear = 0.6 * peak["base_shear_kn"]
        yield_disp = yield_shear / max(initial_stiffness, 1e-9)
    else:
        yield_shear = yield_point["base_shear_kn"]
        yield_disp = yield_point["displacement_m"]

    yield_disp = max(yield_disp, 1e-6)
    ultimate_disp = max(ultimate["displacement_m"], yield_disp)
    post_yield_slope = (peak["base_shear_kn"] - yield_shear) / max(peak["displacement_m"] - yield_disp, 1e-9)
    alpha = post_yield_slope / max(initial_stiffness, 1e-9)
    return {
        "available": True,
        "yield_displacement_m": round(yield_disp, 5),
        "yield_base_shear_kn": round(yield_shear, 3),
        "ultimate_displacement_m": round(ultimate_disp, 5),
        "ultimate_base_shear_kn": round(ultimate["base_shear_kn"], 3),
        "peak_displacement_m": round(peak["displacement_m"], 5),
        "peak_base_shear_kn": round(peak["base_shear_kn"], 3),
        "initial_stiffness_kn_m": round(initial_stiffness, 3),
        "post_yield_stiffness_ratio_alpha": round(alpha, 5),
        "idealization_note": "The yield point is approximated from stiffness degradation on the capacity curve and the 0.6*Vpeak threshold.",
    }


def _initial_period_proxy(story_count: float | None) -> float:
    """Return a bounded first-mode period proxy in the FEMA 440 equation range."""
    if story_count is None or story_count <= 0:
        return 0.5
    return max(0.2, min(2.0, 0.10 * story_count))


def _effective_damping_percent(mu: float, beta0_percent: float = 5.0) -> float:
    """Return FEMA 440 general effective damping approximation in percent."""
    mu = max(1.0, mu)
    if mu <= 1.0:
        return beta0_percent
    if mu < 4.0:
        x = mu - 1.0
        return beta0_percent + 4.9 * x**2 - 1.1 * x**3
    capped = min(mu, 6.5)
    return beta0_percent + 14.0 + 0.32 * (capped - 1.0)


def _effective_period(t0: float, mu: float) -> float:
    """Return FEMA 440 general effective period approximation."""
    mu = max(1.0, mu)
    if mu <= 1.0:
        return t0
    if mu < 4.0:
        x = mu - 1.0
        return (1.0 + 0.20 * x**2 - 0.038 * x**3) * t0
    capped = min(mu, 6.5)
    return (1.0 + 0.28 + 0.13 * (capped - 1.0)) * t0


def _damping_reduction_factor(beta_percent: float) -> float | None:
    """Return FEMA 440/ATC damping reduction factor B(beta) approximation."""
    if beta_percent <= 0:
        return None
    return 4.0 / max(1e-9, 5.6 - math.log(beta_percent))


def _coefficient_c1(r: float, t0: float, site_a: float) -> float:
    """Return a FEMA 440-style short-period C1 proxy."""
    if t0 >= 1.0:
        return 1.0
    return max(1.0, 1.0 + (max(r, 1.0) - 1.0) / max(site_a * t0**2, 1e-9))


def _to_float(value: object) -> float | None:
    """Parse a finite float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
