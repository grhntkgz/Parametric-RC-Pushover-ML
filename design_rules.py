"""Preliminary TS 500 / TBDY 2018 screening rules.

The functions in this module are intentionally conservative pre-design checks
for synthetic model generation. They are not a substitute for final reinforced
concrete member design, detailing, nonlinear assessment, or licensed
engineering judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import GeneratorConfig, Section


CONCRETE_FCK_MPA: dict[str, float] = {
    "C25": 25.0,
    "C30": 30.0,
    "C35": 35.0,
    "C40": 40.0,
}


@dataclass
class CheckResult:
    """Result of a single preliminary design check."""

    passed: bool
    message: str = ""


@dataclass
class ValidationResult:
    """Aggregate validation result for a generated candidate model."""

    is_valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def rejection_text(self) -> str:
        """Return all rejection reasons in a metadata-friendly string."""
        return "; ".join(self.rejection_reasons)


def concrete_elastic_modulus_mpa(concrete_class: str) -> float:
    """Estimate concrete modulus of elasticity in MPa for metadata and SAP input."""
    fck = CONCRETE_FCK_MPA[concrete_class]
    return 3250.0 * (fck**0.5) + 14000.0


def check_column_min_dimension(section: Section, cfg: GeneratorConfig) -> CheckResult:
    """Check minimum column dimension against configurable predesign limit."""
    ok = min(section.width, section.depth) >= cfg.min_column_dimension_m
    return CheckResult(ok, "" if ok else "Column minimum dimension is below limit.")


def check_column_reinforcement_ratio(rho_col: float, cfg: GeneratorConfig) -> CheckResult:
    """Check column reinforcement ratio against preliminary TS 500/TBDY limits."""
    ok = cfg.rho_col_min <= rho_col <= cfg.rho_col_max
    return CheckResult(ok, "" if ok else "Column reinforcement ratio is outside limits.")


def check_beam_reinforcement_ratio(rho_beam: float, cfg: GeneratorConfig) -> CheckResult:
    """Check beam reinforcement ratio against preliminary screening limits."""
    ok = cfg.rho_beam_min <= rho_beam <= cfg.rho_beam_max
    return CheckResult(ok, "" if ok else "Beam reinforcement ratio is outside limits.")


def check_beam_reinforcement_ratios(beam_top_ratio_support: float, beam_bottom_ratio_span: float, cfg: GeneratorConfig) -> CheckResult:
    """Check separate support-top and span-bottom beam reinforcement ratios."""
    ratios = (beam_top_ratio_support, beam_bottom_ratio_span)
    ok = all(cfg.rho_beam_min <= ratio <= cfg.rho_beam_max for ratio in ratios)
    return CheckResult(ok, "" if ok else "Beam top/support or bottom/span reinforcement ratio is outside limits.")


def check_beam_min_dimensions(section: Section, cfg: GeneratorConfig) -> CheckResult:
    """Check minimum beam width and depth against configurable limits."""
    ok = section.width >= cfg.min_beam_width_m and section.depth >= cfg.min_beam_depth_m
    return CheckResult(ok, "" if ok else "Beam dimensions are below minimum limits.")


def check_column_beam_geometry(column: Section, beam: Section) -> CheckResult:
    """Check simple geometric compatibility between selected column and beam sizes."""
    ok = column.width >= beam.width and column.depth >= beam.width
    return CheckResult(ok, "" if ok else "Beam width is not compatible with column size.")


def check_span_to_beam_depth(spans_x: list[float], spans_y: list[float], beam: Section, cfg: GeneratorConfig) -> CheckResult:
    """Check L/h ratio for all beam spans using the selected beam depth."""
    max_ratio = max([span / beam.depth for span in spans_x + spans_y])
    ok = max_ratio <= cfg.max_span_to_beam_depth_ratio
    return CheckResult(ok, "" if ok else f"Maximum span/depth ratio {max_ratio:.2f} exceeds limit.")


def approximate_moment_capacity(rho: float, section: Section, fy_mpa: float) -> float:
    """Return approximate capacity proxy rho*b*h^2*fy.

    This simplified expression is only a relative pre-screening proxy for
    parametric data generation. It is not a code-level flexural capacity.
    """
    return rho * section.width * (section.depth**2) * fy_mpa


def steel_yield_strength_mpa(steel_class: str, cfg: GeneratorConfig) -> float:
    """Return steel yield strength from the configured class map."""
    return cfg.steel_fy_mpa_by_class.get(steel_class, cfg.steel_fy_mpa)


def check_strong_column_weak_beam(column: Section, beam: Section, rho_col: float, rho_beam: float, steel_class: str, cfg: GeneratorConfig) -> CheckResult:
    """Apply an approximate strong-column weak-beam pre-screening check."""
    fy_mpa = steel_yield_strength_mpa(steel_class, cfg)
    sum_m_col = 2.0 * approximate_moment_capacity(rho_col, column, fy_mpa)
    sum_m_beam = 2.0 * approximate_moment_capacity(rho_beam, beam, fy_mpa)
    ok = sum_m_col >= cfg.strong_column_factor * sum_m_beam
    return CheckResult(ok, "" if ok else "Approximate strong-column weak-beam check failed.")


def check_rebar_ratio_candidates(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check that selected ratios came from the configured discrete automation sets."""
    ok = (
        candidate["rho_col"] in cfg.column_rebar_ratios
        and candidate["beam_top_ratio_support"] in cfg.beam_top_ratio_supports
        and candidate["beam_bottom_ratio_span"] in cfg.beam_bottom_ratio_spans
        and candidate["slab_rebar_ratio"] in cfg.slab_rebar_ratios
        and candidate["raft_rebar_ratio"] in cfg.raft_rebar_ratios
    )
    return CheckResult(ok, "" if ok else "Selected reinforcement ratios are not in configured automation sets.")


def check_story_height_total_height_placeholder(story_count: int, story_height: float) -> CheckResult:
    """Placeholder pre-stability check for story and total height reasonableness."""
    total_height = story_count * story_height
    ok = 12.0 <= total_height <= 30.0
    return CheckResult(ok, "" if ok else "Total height is outside placeholder stability range.")


def check_soil_class(soil_class: str, cfg: GeneratorConfig) -> CheckResult:
    """Check that the selected soil class is one of the configured TBDY classes."""
    ok = soil_class in cfg.soil_classes and soil_class in cfg.soil_subgrade_modulus_kn_m3
    return CheckResult(ok, "" if ok else "Soil class is not available in configured TBDY class set.")


def check_raft_thickness(raft_thickness: float, cfg: GeneratorConfig) -> CheckResult:
    """Check raft thickness against a configurable preliminary lower bound."""
    ok = raft_thickness >= cfg.min_raft_thickness_m
    return CheckResult(ok, "" if ok else "Raft foundation thickness is below preliminary minimum.")


def check_raft_reinforcement_ratio(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check preliminary raft reinforcement ratio limits.

    This is a synthetic pre-screen based on slab/mat foundation reinforcement
    reasonableness. Final raft reinforcement must still be designed from
    bending, shear, punching shear, soil pressure, and settlement checks.
    """
    if not cfg.enable_raft_foundation:
        return CheckResult(True)
    rho = candidate["raft_rebar_ratio"]
    ok = cfg.raft_rho_min <= rho <= cfg.raft_rho_max
    return CheckResult(ok, "" if ok else "Raft reinforcement ratio is outside preliminary foundation limits.")


def check_subgrade_modulus(subgrade_modulus_kn_m3: float) -> CheckResult:
    """Check that the Winkler spring modulus proxy is positive."""
    ok = subgrade_modulus_kn_m3 > 0.0
    return CheckResult(ok, "" if ok else "Soil subgrade modulus must be positive.")


def check_wall_thickness(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check preliminary TBDY wall thickness limits when shear walls are enabled."""
    if not candidate.get("has_shear_walls"):
        return CheckResult(True)
    required = max(cfg.min_wall_thickness_m, candidate["story_height"] / 16.0)
    ok = candidate["wall_thickness_m"] >= required
    return CheckResult(ok, "" if ok else f"Wall thickness is below max(250 mm, story_height/16) = {required:.3f} m.")


def check_wall_aspect_ratio(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check lw/tw ratio for preliminary TBDY shear wall classification."""
    if not candidate.get("has_shear_walls"):
        return CheckResult(True)
    ratio = candidate["wall_length_m"] / candidate["wall_thickness_m"]
    ok = ratio >= cfg.min_wall_aspect_ratio
    return CheckResult(ok, "" if ok else f"Wall length/thickness ratio {ratio:.2f} is below {cfg.min_wall_aspect_ratio:.1f}.")


def check_wall_reinforcement_ratio(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check wall reinforcement ratio against configurable preliminary limits."""
    if not candidate.get("has_shear_walls"):
        return CheckResult(True)
    ok = cfg.wall_rho_min <= candidate["wall_rebar_ratio"] <= cfg.wall_rho_max
    return CheckResult(ok, "" if ok else "Wall reinforcement ratio is outside configured limits.")


def check_slab_thickness(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check preliminary TS 500 thickness for beam-supported continuous slabs."""
    if not cfg.enable_floor_slabs:
        return CheckResult(True)
    governing_short_span = max(min(span_x, span_y) for span_x in candidate["spans_x"] for span_y in candidate["spans_y"])
    required = max(cfg.min_slab_thickness_m, governing_short_span / cfg.max_slab_continuous_span_to_thickness_ratio)
    ok = candidate["slab_thickness_m"] >= required
    return CheckResult(ok, "" if ok else f"Slab thickness is below preliminary max({cfg.min_slab_thickness_m:.3f} m, short_span/{cfg.max_slab_continuous_span_to_thickness_ratio:.0f}) = {required:.3f} m.")


def check_slab_reinforcement_ratio(candidate: dict[str, Any], cfg: GeneratorConfig) -> CheckResult:
    """Check conservative slab reinforcement ratio applied in both directions."""
    if not cfg.enable_floor_slabs:
        return CheckResult(True)
    rho = candidate["slab_rebar_ratio"]
    ok = cfg.slab_rho_min <= rho <= cfg.slab_rho_max and 2.0 * rho >= 0.0035
    return CheckResult(ok, "" if ok else "Slab reinforcement ratio is outside conservative TS 500 preliminary limits.")


def check_slab_in_plane_stress_placeholder() -> CheckResult:
    """Placeholder for future TBDY 7.11.3 elastic-diaphragm stress checks."""
    return CheckResult(True)


def check_column_slenderness_placeholder() -> CheckResult:
    """Placeholder for future column slenderness / second-order screening."""
    return CheckResult(True)


def check_soft_story_placeholder() -> CheckResult:
    """Placeholder for future TBDY soft-story irregularity screening."""
    return CheckResult(True)


def check_torsional_irregularity_placeholder() -> CheckResult:
    """Placeholder for future TBDY torsional irregularity screening."""
    return CheckResult(True)


def check_mass_irregularity_placeholder() -> CheckResult:
    """Placeholder for future TBDY mass irregularity screening."""
    return CheckResult(True)


def validate_candidate_model(candidate: dict[str, Any], cfg: GeneratorConfig) -> ValidationResult:
    """Run all preliminary code-sanity checks for a generated candidate."""
    checks_to_run: dict[str, CheckResult] = {
        "kolon_min_boyut": check_column_min_dimension(candidate["column_section"], cfg),
        "kolon_min_donati": CheckResult(candidate["rho_col"] >= cfg.rho_col_min, "Column rho is below minimum."),
        "kolon_max_donati": CheckResult(candidate["rho_col"] <= cfg.rho_col_max, "Column rho is above maximum."),
        "kiris_min_donati": CheckResult(
            min(candidate["beam_top_ratio_support"], candidate["beam_bottom_ratio_span"]) >= cfg.rho_beam_min,
            "Beam rho is below minimum.",
        ),
        "kiris_max_donati": CheckResult(
            max(candidate["beam_top_ratio_support"], candidate["beam_bottom_ratio_span"]) <= cfg.rho_beam_max,
            "Beam rho is above maximum.",
        ),
        "kiris_ust_alt_donati": check_beam_reinforcement_ratios(candidate["beam_top_ratio_support"], candidate["beam_bottom_ratio_span"], cfg),
        "donati_orani_aday_kumesi": check_rebar_ratio_candidates(candidate, cfg),
        "kiris_min_boyut": check_beam_min_dimensions(candidate["beam_section"], cfg),
        "kolon_kiris_geometri": check_column_beam_geometry(candidate["column_section"], candidate["beam_section"]),
        "aciklik_kiris_derinlik": check_span_to_beam_depth(candidate["spans_x"], candidate["spans_y"], candidate["beam_section"], cfg),
        "guclu_kolon_zayif_kiris": check_strong_column_weak_beam(
            candidate["column_section"],
            candidate["beam_section"],
            candidate["rho_col"],
            candidate["beam_top_ratio_support"],
            candidate["steel_class"],
            cfg,
        ),
        "kat_yuksekligi_toplam_yukseklik": check_story_height_total_height_placeholder(candidate["story_count"], candidate["story_height"]),
        "zemin_sinifi": check_soil_class(candidate["soil_class"], cfg),
        "radye_kalinligi": check_raft_thickness(candidate["raft_thickness_m"], cfg),
        "radye_donati_orani": check_raft_reinforcement_ratio(candidate, cfg),
        "zemin_yatak_katsayisi": check_subgrade_modulus(candidate["subgrade_modulus_kn_m3"]),
        "perde_kalinligi": check_wall_thickness(candidate, cfg),
        "perde_lw_tw": check_wall_aspect_ratio(candidate, cfg),
        "perde_donati_orani": check_wall_reinforcement_ratio(candidate, cfg),
        "doseme_kalinligi": check_slab_thickness(candidate, cfg),
        "doseme_donati_orani": check_slab_reinforcement_ratio(candidate, cfg),
        "doseme_duzlem_ici_gerilme_placeholder": check_slab_in_plane_stress_placeholder(),
        "kolon_narinlik_placeholder": check_column_slenderness_placeholder(),
        "yumusak_kat_placeholder": check_soft_story_placeholder(),
        "burulma_duzensizligi_placeholder": check_torsional_irregularity_placeholder(),
        "kutle_duzensizligi_placeholder": check_mass_irregularity_placeholder(),
    }

    checks = {name: result.passed for name, result in checks_to_run.items()}
    reasons = [result.message for result in checks_to_run.values() if not result.passed and result.message]
    return ValidationResult(is_valid=all(checks.values()), checks=checks, rejection_reasons=reasons)
