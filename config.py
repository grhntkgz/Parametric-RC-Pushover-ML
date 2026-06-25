"""Configuration for automatic SAP2000 reinforced-concrete model generation.

All dimensions used by the generator and SAP2000 API are in kN, m, C units
unless noted otherwise. Code-check limits are preliminary screening values,
not final TS 500 / TBDY 2018 design calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Section:
    """Rectangular frame section dimensions in meters."""

    width: float
    depth: float

    @property
    def label_cm(self) -> str:
        """Return a compact label such as 50x50."""
        return f"{round(self.width * 100):02d}x{round(self.depth * 100):02d}"


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level settings for model generation."""

    n_iter: int = 5
    random_seed: Optional[int] = 42
    max_attempts_per_model: int = 100
    output_dir: Path = Path(r"D:\sap2000_generated_models")
    sap2000_exe_path: Path = Path(r"C:\Program Files\Computers and Structures\SAP2000 22\SAP2000.exe")
    keep_sap_open: bool = False

    story_count_min: int = 4
    story_count_max: int = 8
    x_bay_count_min: int = 2
    x_bay_count_max: int = 5
    y_bay_count_min: int = 2
    y_bay_count_max: int = 4

    bay_length_min_m: float = 4.0
    bay_length_max_m: float = 7.0
    bay_length_step_m: float = 0.5
    story_height_min_m: float = 3.0
    story_height_max_m: float = 3.5
    story_height_step_m: float = 0.1

    concrete_classes: tuple[str, ...] = ("C25", "C30", "C35", "C40")
    steel_classes: tuple[str, ...] = ("B420C", "B500C")
    steel_fy_mpa: float = 420.0
    steel_fy_mpa_by_class: dict[str, float] = field(
        default_factory=lambda: {
            "B420C": 420.0,
            "B500C": 500.0,
        }
    )

    column_sections: tuple[Section, ...] = (
        Section(0.40, 0.40),
        Section(0.45, 0.45),
        Section(0.50, 0.50),
        Section(0.55, 0.55),
        Section(0.60, 0.60),
        Section(0.70, 0.70),
    )
    beam_sections: tuple[Section, ...] = (
        Section(0.25, 0.50),
        Section(0.30, 0.50),
        Section(0.30, 0.60),
        Section(0.35, 0.60),
        Section(0.35, 0.70),
    )

    rho_col_min: float = 0.01
    rho_col_max: float = 0.04
    rho_beam_min: float = 0.003
    rho_beam_max: float = 0.025
    rho_col_random_min: float = 0.01
    rho_col_random_max: float = 0.04
    rho_beam_random_min: float = 0.003
    rho_beam_random_max: float = 0.025
    column_rebar_ratios: tuple[float, ...] = (0.010, 0.015, 0.020, 0.025, 0.030)
    beam_top_ratio_supports: tuple[float, ...] = (0.005, 0.008, 0.012, 0.016, 0.020)
    beam_bottom_ratio_spans: tuple[float, ...] = (0.005, 0.008, 0.012, 0.016)
    wall_rebar_ratios: tuple[float, ...] = (0.0025, 0.0030, 0.0040, 0.0050, 0.0060)
    slab_rebar_ratios: tuple[float, ...] = (0.0020, 0.0025, 0.0030, 0.0035, 0.0040)
    raft_rebar_ratios: tuple[float, ...] = (0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050)

    min_column_dimension_m: float = 0.30
    min_beam_width_m: float = 0.25
    min_beam_depth_m: float = 0.50
    max_span_to_beam_depth_ratio: float = 15.0
    strong_column_factor: float = 1.20

    # Beam-supported floor slabs are modeled as horizontal shell panels. One
    # selected ratio is applied in both orthogonal directions. Requiring at
    # least 0.0020 in each direction is a conservative preliminary screen that
    # covers the TS 500 S420/S500 one-way main-rebar minimum and the two-way
    # slab per-direction / total-ratio minimums.
    enable_floor_slabs: bool = True
    slab_thickness_min_m: float = 0.15
    slab_thickness_max_m: float = 0.25
    slab_thickness_step_m: float = 0.01
    min_slab_thickness_m: float = 0.12
    max_slab_continuous_span_to_thickness_ratio: float = 30.0
    slab_rho_min: float = 0.0020
    slab_rho_max: float = 0.0100

    enable_shear_walls: bool = False
    wall_thickness_min_m: float = 0.25
    wall_thickness_max_m: float = 0.30
    wall_thickness_step_m: float = 0.05
    wall_length_min_m: float = 1.50
    wall_length_max_m: float = 4.00
    wall_length_step_m: float = 0.50
    wall_rho_min: float = 0.0025
    wall_rho_max: float = 0.0100
    min_wall_thickness_m: float = 0.25
    min_wall_aspect_ratio: float = 6.0
    wall_placement: str = "perimeter_symmetric"

    # Raft foundation and preliminary TBDY 2018 soil-class parameters.
    # The subgrade modulus values are configurable Winkler spring proxies
    # (kN/m3); they are not a replacement for project-specific geotechnical
    # reports or final soil-structure interaction design.
    enable_raft_foundation: bool = True
    soil_classes: tuple[str, ...] = ("ZA", "ZB", "ZC", "ZD", "ZE")
    soil_subgrade_modulus_kn_m3: dict[str, float] = field(
        default_factory=lambda: {
            "ZA": 120_000.0,
            "ZB": 80_000.0,
            "ZC": 50_000.0,
            "ZD": 25_000.0,
            "ZE": 12_000.0,
        }
    )
    raft_thickness_min_m: float = 0.50
    raft_thickness_max_m: float = 1.00
    raft_thickness_step_m: float = 0.10
    min_raft_thickness_m: float = 0.40
    raft_rho_min: float = 0.0020
    raft_rho_max: float = 0.0100
    foundation_edge_offset_m: float = 0.50
    assign_soil_vertical_springs: bool = True

    dead_load_kn_m: float = 10.0
    live_load_kn_m: float = 4.0

    # Pushover settings. These are deliberately configurable because the target
    # displacement is a study/optimization parameter, not a fixed code result.
    enable_pushover_cases: bool = True
    run_analysis_after_save: bool = True
    pushover_directions: tuple[str, ...] = ("X", "Y")
    fix_pushover_target_drift_ratio: bool = False
    pushover_target_drift_ratio_fixed: float = 0.040
    pushover_target_drift_ratio_min: float = 0.015
    pushover_target_drift_ratio_max: float = 0.040
    pushover_target_drift_ratio_step: float = 0.005
    pushover_base_shear_proxy_kn: float = 1000.0
    pushover_load_distribution: str = "triangular"
    enable_plastic_hinges: bool = True
    require_plastic_hinge_assignment: bool = True
    column_hinge_property: str = "Default-PMM"
    beam_hinge_property: str = "Default-M3"
    shear_hinge_v2_property: str = "Default-V2"
    shear_hinge_v3_property: str = "Default-V3"
    assign_column_pmm_hinges: bool = True
    assign_beam_m3_hinges: bool = True
    assign_shear_hinges: bool = False
    hinge_relative_distances: tuple[float, float] = (0.0, 1.0)
    hinge_result_case: str = "PUSHOVER_X"
    enable_exact_hinge_ui_export: bool = False
    enable_inline_exact_hinge_export: bool = False
    exact_hinge_export_max_attempts_per_model: int = 1
    exact_hinge_export_min_rows_per_direction: int = 100
    enable_sap_screenshots: bool = False
    sap_screenshot_delay_s: float = 0.8

    # Machine-learning persistence/training settings. The dashboard keeps these
    # model files separate from generated SAP2000 artifacts so a long study can
    # be stopped, restarted, or cleaned without losing learned predictors.
    ml_auto_train_after_generation: bool = False
    ml_auto_train_algorithm: str = "random_forest"
    ml_preserve_existing_weights: bool = True
    ml_random_seed: int = 42
    ml_use_som_features: bool = False


CONFIG = GeneratorConfig()
