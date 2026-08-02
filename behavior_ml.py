"""Behavior classification analyses for article-oriented interpretation.

This module trains one-vs-rest tree classifiers for selected structural
behavior headings. It intentionally stays separate from ``ml_model.py``:
``ml_model.py`` is used for prediction, while this file is used to produce
paper-ready parameter interpretation tables.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fema440 import calculate_fema440_for_curve

STATE_RANK = {"A-B": 0, "B-IO": 1, "IO-LS": 2, "LS-CP": 3, "CP-C": 4, "C-D": 5, "D-E": 6, "beyond E": 7, "Beyond E": 7}
HIGH_STATES = {"LS-CP", "CP-C", "C-D", "D-E", "beyond E", "Beyond E"}
LIMITED_STATES = {"B-IO", "IO-LS"}
CP_OR_MORE_STATES = {"CP-C", "C-D", "D-E", "beyond E", "Beyond E"}
OUTPUT_NUMERIC_FEATURES = {
    "peak_base_shear",
    "max_story_drift_ratio",
    "max_rotation",
    "critical_story",
    "first_story",
    "lscp_ratio",
    "cp_ratio",
    "fema_target_capacity_ratio",
}


DIRECTIONAL_TARGET_GROUPS: dict[str, dict[str, Any]] = {
    "damage_classes": {
        "label": "Damage Class Evaluations",
        "dataset": "directional",
        "targets": {
            "damage_bio": {"label": "B-IO", "kind": "damage_state", "state": "B-IO"},
            "damage_iols": {"label": "IO-LS", "kind": "damage_state", "state": "IO-LS"},
            "damage_lscp": {"label": "LS-CP", "kind": "damage_state", "state": "LS-CP"},
        },
    },
    "rare_final": {
        "label": "Rare Final-State Behaviors",
        "dataset": "directional",
        "targets": {
            "critical_beam": {"label": "Critical element remains beam-controlled", "kind": "critical_type", "value": "beam"},
            "critical_middle": {"label": "Critical element occurs on an interior axis", "kind": "critical_axis", "value": "middle"},
            "critical_upper": {"label": "Critical element occurs above the first story", "kind": "critical_upper_story"},
        },
    },
    "rare_transition": {
        "label": "Rare Initial and Transition Behaviors",
        "dataset": "directional",
        "targets": {
            "first_hinge_story_ge_3": {"label": "First plastic hinge starts at story 3 or above", "kind": "first_story_ge", "value": 3},
            "beam_to_beam": {"label": "Starts in beam and remains beam-critical", "kind": "first_type_critical_type", "first": "beam", "critical": "beam"},
            "upper_to_upper": {"label": "Starts in upper story and remains upper-story critical", "kind": "first_upper_critical_upper"},
            "edge_to_middle": {"label": "Starts on edge axis and shifts to interior axis", "kind": "first_axis_critical_axis", "first": "edge", "critical": "middle"},
            "beam_middle_to_column_edge": {
                "label": "Beam/interior initiation shifting to column/edge criticality",
                "kind": "compound_transition",
                "first_type": "beam",
                "first_axis": "middle",
                "critical_type": "column",
                "critical_axis": "edge",
            },
        },
    },
    "performance_outliers": {
        "label": "Performance-Level Outlier Behaviors",
        "dataset": "directional",
        "targets": {
            "low_drift_high_damage": {"label": "LS-CP+ high damage at low target drift", "kind": "low_drift_high_damage"},
            "high_drift_limited_damage": {"label": "B-IO/IO-LS limited damage at high target drift", "kind": "high_drift_limited_damage"},
            "early_ls_step_le_5": {"label": "Very early LS occurrence, step <= 5", "kind": "early_ls_step", "value": 5},
            "high_drift_only_bio": {"label": "B-IO only despite high target drift", "kind": "high_drift_only_bio"},
        },
    },
    "primary_analysis_targets": {
        "label": "Primary Analysis Targets",
        "dataset": "directional",
        "targets": {
            "critical_element_type_beam": {
                "label": "Critical element type: beam rather than column",
                "task": "classification",
                "kind": "critical_type",
                "value": "beam",
            },
            "first_hinge_type_beam": {
                "label": "First plastic hinge element type: beam rather than column",
                "task": "classification",
                "kind": "first_type",
                "value": "beam",
            },
            "critical_element_story": {
                "label": "Critical element story number",
                "task": "regression",
                "field": "critical_story",
            },
            "critical_element_plan_zone": {
                "label": "Critical element plan zone: interior axis rather than edge axis",
                "task": "classification",
                "kind": "critical_axis",
                "value": "middle",
            },
        },
    },
    "structural_response_parameters": {
        "label": "Structural Performance Response Parameters",
        "dataset": "directional",
        "targets": {
            "first_hinge_story": {"label": "First plastic hinge story", "task": "regression", "field": "first_story"},
            "first_hinge_plan_zone": {"label": "First plastic hinge on interior axis", "task": "classification", "kind": "first_axis", "value": "middle"},
            "max_story_drift_ratio": {"label": "Maximum interstory drift ratio", "task": "regression", "field": "max_story_drift_ratio"},
            "max_rotation": {"label": "Maximum plastic rotation", "task": "regression", "field": "max_rotation"},
            "peak_base_shear": {"label": "Peak base shear", "task": "regression", "field": "peak_base_shear"},
        },
    },
    "supporting_performance_indicators": {
        "label": "Supporting Performance Indicators",
        "dataset": "directional",
        "targets": {
            "lscp_ratio": {
                "label": "LS-CP hinge-event ratio among all recorded hinge events",
                "task": "regression",
                "field": "lscp_ratio",
            },
            "cp_ratio": {
                "label": "CP-C or more severe hinge-event ratio among all recorded hinge events",
                "task": "regression",
                "field": "cp_ratio",
            },
            "has_lscp": {
                "label": "At least one LS-CP hinge event occurs",
                "task": "classification",
                "kind": "field_positive",
                "field": "has_lscp",
            },
            "has_cp": {
                "label": "At least one CP-C or more severe hinge event occurs",
                "task": "classification",
                "kind": "field_positive",
                "field": "has_cp",
            },
            "fema_target_capacity_ratio": {
                "label": "FEMA 440 displacement-modification target-to-capacity ratio",
                "task": "regression",
                "field": "fema_target_capacity_ratio",
            },
            "fema_capacity_status_exceeded": {
                "label": "FEMA 440 capacity status: target displacement exceeds capacity",
                "task": "classification",
                "kind": "field_equals",
                "field": "fema_capacity_status",
                "value": "capacity_exceeded",
            },
        },
    },
}

MODEL_TARGET_GROUPS: dict[str, dict[str, Any]] = {
    "directional_asymmetry": {
        "label": "Directional Asymmetry Behaviors",
        "dataset": "model",
        "targets": {
            "damage_rank_diff_ge_2": {"label": "X/Y critical damage level differs by at least two stages", "kind": "damage_rank_diff_ge", "value": 2},
            "one_lscp_plus_other_limited": {"label": "LS-CP+ in one direction, B-IO/IO-LS in the other", "kind": "one_high_one_limited"},
            "critical_type_diff": {"label": "X/Y critical element type differs", "kind": "critical_field_diff", "field": "critical_type"},
            "critical_story_diff": {"label": "X/Y critical story differs", "kind": "critical_field_diff", "field": "critical_story"},
            "critical_axis_diff": {"label": "X/Y critical axis location differs", "kind": "critical_field_diff", "field": "critical_axis"},
        },
    }
}

TARGET_GROUPS = {**DIRECTIONAL_TARGET_GROUPS, **MODEL_TARGET_GROUPS}


DIRECTIONAL_NUMERIC_FEATURES = [
    "story_count",
    "x_bay_count",
    "y_bay_count",
    "bay_count_dir",
    "avg_span_x",
    "avg_span_y",
    "avg_span_dir",
    "max_span_x",
    "max_span_y",
    "story_height",
    "total_height",
    "concrete_fck",
    "steel_fy",
    "column_width",
    "column_depth",
    "column_area",
    "beam_width",
    "beam_depth",
    "beam_area",
    "rho_col",
    "rho_beam_top",
    "rho_beam_bottom",
    "slab_thickness",
    "slab_rebar_ratio",
    "raft_thickness",
    "raft_rebar_ratio",
    "subgrade_modulus",
    "wall_count",
    "wall_area",
    "wall_thickness",
    "wall_length",
    "wall_rebar_ratio",
    "target_drift",
    "target_displacement",
    "peak_base_shear",
    "max_story_drift_ratio",
    "max_rotation",
]

DIRECTIONAL_CATEGORICAL_FEATURES = ["direction", "concrete_class", "steel_class", "soil_class", "has_shear_walls"]

MODEL_NUMERIC_FEATURES = [
    "story_count",
    "x_bay_count",
    "y_bay_count",
    "bay_count_diff_abs",
    "bay_count_ratio",
    "avg_span_x",
    "avg_span_y",
    "avg_span_diff_abs",
    "avg_span_ratio",
    "max_span_x",
    "max_span_y",
    "max_span_diff_abs",
    "plan_dim_x",
    "plan_dim_y",
    "plan_aspect_ratio",
    "plan_area",
    "story_height",
    "total_height",
    "concrete_fck",
    "steel_fy",
    "column_width",
    "column_depth",
    "column_area",
    "beam_width",
    "beam_depth",
    "beam_area",
    "rho_col",
    "rho_beam_top",
    "rho_beam_bottom",
    "slab_thickness",
    "slab_rebar_ratio",
    "raft_thickness",
    "raft_rebar_ratio",
    "subgrade_modulus",
    "wall_count",
    "wall_area",
    "wall_thickness",
    "wall_length",
    "wall_rebar_ratio",
    "target_drift",
    "target_displacement",
]

MODEL_CATEGORICAL_FEATURES = ["concrete_class", "steel_class", "soil_class", "has_shear_walls"]

FEATURE_LABELS = {
    "target_drift": "Target drift ratio",
    "target_displacement": "Target roof displacement",
    "rho_col": "Column reinforcement ratio",
    "column_width": "Column width",
    "column_depth": "Column depth",
    "column_area": "Column area",
    "beam_width": "Beam width",
    "beam_depth": "Beam depth",
    "beam_area": "Beam area",
    "rho_beam_top": "Beam top reinforcement ratio",
    "rho_beam_bottom": "Beam bottom reinforcement ratio",
    "wall_count": "Shear wall count",
    "wall_area": "Shear wall area",
    "wall_thickness": "Shear wall thickness",
    "wall_length": "Shear wall length",
    "wall_rebar_ratio": "Shear wall reinforcement ratio",
    "slab_thickness": "Slab thickness",
    "slab_rebar_ratio": "Slab reinforcement ratio",
    "raft_thickness": "Raft foundation thickness",
    "raft_rebar_ratio": "Raft foundation reinforcement ratio",
    "subgrade_modulus": "Subgrade modulus",
    "story_count": "Story count",
    "story_height": "Story height",
    "total_height": "Total height",
    "bay_count_dir": "Bay count in analysis direction",
    "x_bay_count": "X-direction bay count",
    "y_bay_count": "Y-direction bay count",
    "avg_span_dir": "Span length in analysis direction",
    "avg_span_x": "X-direction average span",
    "avg_span_y": "Y-direction average span",
    "max_span_x": "X-direction maximum span",
    "max_span_y": "Y-direction maximum span",
    "avg_span_diff_abs": "Absolute X/Y span difference",
    "avg_span_ratio": "X/Y span ratio",
    "bay_count_diff_abs": "Absolute X/Y bay count difference",
    "bay_count_ratio": "X/Y bay count ratio",
    "plan_area": "Plan area",
    "plan_aspect_ratio": "Plan aspect ratio",
    "plan_dim_x": "Plan dimension in X",
    "plan_dim_y": "Plan dimension in Y",
    "peak_base_shear": "Peak base shear",
    "max_story_drift_ratio": "Maximum interstory drift ratio",
    "max_rotation": "Maximum plastic rotation",
    "lscp_ratio": "LS-CP hinge-event ratio",
    "cp_ratio": "CP-C or more severe hinge-event ratio",
    "fema_target_capacity_ratio": "FEMA 440 target-to-capacity ratio",
    "first_story": "First plastic hinge story",
    "first_axis": "First plastic hinge plan zone",
    "critical_story": "Critical element story",
    "critical_axis": "Critical element plan zone",
    "critical_type": "Critical element type",
    "first_type": "First plastic hinge element type",
    "direction": "Analysis direction",
    "concrete_class": "Concrete class",
    "steel_class": "Steel class",
    "soil_class": "Soil class",
    "has_shear_walls": "Shear wall presence",
}


@dataclass(frozen=True)
class BehaviorDataset:
    """Parsed behavior dataset."""

    directional_rows: list[dict[str, Any]]
    model_rows: list[dict[str, Any]]
    source_count: int


def behavior_ml_metadata() -> dict[str, Any]:
    """Return available behavior target groups for the dashboard."""
    groups = []
    for group_id, group in TARGET_GROUPS.items():
        groups.append(
            {
                "id": group_id,
                "label": group["label"],
                "dataset": group["dataset"],
                "targets": [{"id": key, "label": value["label"]} for key, value in group["targets"].items()],
            }
        )
    return {"groups": groups, "algorithms": ["random_forest", "xgboost", "lightgbm"]}


def run_behavior_ml_analysis(
    output_dir: Path,
    group_id: str,
    target_id: str,
    algorithms: list[str] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train selected tree classifiers and return article-oriented tables."""
    if group_id not in TARGET_GROUPS:
        raise ValueError(f"Unknown behavior group: {group_id}")
    group = TARGET_GROUPS[group_id]
    target = group["targets"].get(target_id)
    if not target:
        raise ValueError(f"Unknown behavior target: {target_id}")
    algorithms = [normalize_algorithm(item) for item in (algorithms or ["random_forest", "xgboost", "lightgbm"])]
    algorithms = [item for item in algorithms if item in {"random_forest", "xgboost", "lightgbm"}]
    if not algorithms:
        raise ValueError("Select at least one algorithm.")

    dataset = build_behavior_dataset(output_dir)
    rows = dataset.model_rows if group["dataset"] == "model" else dataset.directional_rows
    if len(rows) < 30:
        raise ValueError("Not enough records were found for behavior analysis.")

    task = str(target.get("task") or "classification")
    if task == "regression":
        values = [as_float(row.get(str(target.get("field") or "")), math.nan) for row in rows]
        valid_pairs = [(row, value) for row, value in zip(rows, values) if math.isfinite(value)]
        if len(valid_pairs) < 30:
            raise ValueError("Not enough valid records were found for the selected numeric target.")
        rows = [row for row, _ in valid_pairs]
        values = [value for _, value in valid_pairs]
        positive_count = None
        positive_rate = None
        metrics, importance_rows, feature_profiles = train_tree_regressors(rows, values, group["dataset"], algorithms, random_seed)
        comments = build_regression_comments(target["label"], len(values), metrics, feature_profiles)
    else:
        labels = [1 if target_matches(row, target, group["dataset"]) else 0 for row in rows]
        positive_count = sum(labels)
        if positive_count < 2:
            raise ValueError("The selected behavior requires at least two positive records.")
        if positive_count >= len(labels) - 1:
            raise ValueError("The selected behavior does not have enough negative records.")
        positive_rate = positive_count / len(labels)
        metrics, importance_rows, feature_profiles = train_tree_models(rows, labels, group["dataset"], algorithms, random_seed)
        comments = build_comments(target["label"], positive_count, len(labels), metrics, feature_profiles)

    summary = {
        "group_id": group_id,
        "group_label": group["label"],
        "target_id": target_id,
        "target_label": target["label"],
        "task": task,
        "dataset_type": group["dataset"],
        "sample_count": len(rows),
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "source_metadata_count": dataset.source_count,
        "metrics": metrics,
        "feature_importance": importance_rows,
        "feature_profiles": feature_profiles,
        "comments": comments,
        "method_note": (
            "Classification targets are evaluated as one-vs-rest problems. Regression targets are evaluated directly as numeric outputs. "
            "Tree feature importances are aggregated from encoded variables back to the original structural parameters."
        ),
    }
    write_behavior_report(output_dir, group_id, target_id, summary)
    return summary


def normalize_algorithm(value: str) -> str:
    """Normalize algorithm key."""
    clean = str(value or "").strip().lower().replace("-", "_")
    if clean in {"rf", "randomforest"}:
        return "random_forest"
    if clean in {"xgb"}:
        return "xgboost"
    if clean in {"lgbm", "light_gbm"}:
        return "lightgbm"
    return clean


def build_behavior_dataset(output_dir: Path) -> BehaviorDataset:
    """Read JSON metadata and build directional/model rows."""
    directional_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    source_count = 0
    for path in metadata_json_paths(output_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("status") or "") != "success":
            continue
        source_count += 1
        per_direction: dict[str, dict[str, Any]] = {}
        for direction in ("X", "Y"):
            row = directional_row_from_metadata(data, direction, path)
            if row:
                per_direction[direction] = row
                directional_rows.append(row)
        if "X" in per_direction and "Y" in per_direction:
            model_rows.append(model_row_from_directional(data, path, per_direction["X"], per_direction["Y"]))
    return BehaviorDataset(deduplicate_rows(directional_rows, ("model_name", "direction")), deduplicate_rows(model_rows, ("model_name",)), source_count)


def metadata_json_paths(output_dir: Path) -> list[Path]:
    """Return metadata JSON files under an output directory or metadata folder."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    paths = [path for path in output_dir.glob("*_metadata.json") if path.is_file()]
    paths.extend(path for path in output_dir.glob("run_*/*_metadata.json") if path.is_file())
    if not paths:
        paths.extend(path for path in output_dir.rglob("*_metadata.json") if path.is_file())
    return sorted(set(paths))


def deduplicate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Keep one row for each key tuple."""
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(item) for item in keys)
        deduped[key] = row
    return list(deduped.values())


def directional_row_from_metadata(data: dict[str, Any], direction: str, path: Path) -> dict[str, Any] | None:
    """Build one row for one pushover direction."""
    candidate = data.get("candidate", {}) if isinstance(data.get("candidate"), dict) else {}
    summary = direction_summary(data, direction)
    if not summary:
        return None
    first = summary.get("first_plastic_hinge") if isinstance(summary.get("first_plastic_hinge"), dict) else {}
    critical = critical_event(summary)
    if not first or not critical:
        return None

    spans_x = [as_float(item) for item in candidate.get("spans_x", [])]
    spans_y = [as_float(item) for item in candidate.get("spans_y", [])]
    story_count = as_float(candidate.get("story_count"))
    story_height = as_float(candidate.get("story_height"))
    column = candidate.get("column_section", {}) if isinstance(candidate.get("column_section"), dict) else {}
    beam = candidate.get("beam_section", {}) if isinstance(candidate.get("beam_section"), dict) else {}
    column_width = as_float(column.get("width"))
    column_depth = as_float(column.get("depth"))
    beam_width = as_float(beam.get("width"))
    beam_depth = as_float(beam.get("depth"))
    has_walls = bool(candidate.get("has_shear_walls"))
    wall_count = as_float(candidate.get("wall_count")) if has_walls else 0.0
    wall_thickness = as_float(candidate.get("wall_thickness_m")) if has_walls else 0.0
    wall_length = as_float(candidate.get("wall_length_m")) if has_walls else 0.0
    pushover = data.get("pushover", {}) if isinstance(data.get("pushover"), dict) else {}
    pushover_results = pushover.get("results", {}) if isinstance(pushover.get("results"), dict) else {}
    curves = (pushover_results.get("curves", {}) or {}) if isinstance(pushover_results.get("curves", {}), dict) else {}
    curve = curves.get(direction, {}) if isinstance(curves.get(direction), dict) else {}
    story_drifts = pushover_results.get("story_drifts", {}) if isinstance(pushover_results.get("story_drifts"), dict) else {}
    fema = calculate_fema440_for_curve(curve, story_count=story_count, soil_class=str(candidate.get("soil_class") or ""))
    fema_dm = fema.get("displacement_modification", {}) if isinstance(fema.get("displacement_modification"), dict) else {}
    fema_target_displacement = as_float(fema_dm.get("target_displacement_proxy_m"), math.nan)
    final_capacity_displacement = as_float(curve.get("final_control_displacement_m"), math.nan)
    fema_ratio = safe_ratio_or_nan(fema_target_displacement, final_capacity_displacement)
    state_counts = summary.get("state_counts", {}) if isinstance(summary.get("state_counts"), dict) else {}
    event_count = as_float(summary.get("event_count"))
    lscp_count = as_float(state_counts.get("LS-CP"))
    cp_count = sum(as_float(state_counts.get(state)) for state in CP_OR_MORE_STATES)
    lscp_ratio = safe_divide(lscp_count, event_count) if event_count > 0 else 0.0
    cp_ratio = safe_divide(cp_count, event_count) if event_count > 0 else 0.0

    first_location = event_location(first, spans_x, spans_y)
    critical_location = event_location(critical, spans_x, spans_y)
    first_ls = summary.get("first_ls_level") if isinstance(summary.get("first_ls_level"), dict) else {}
    state = str(critical.get("hinge_state_level") or "")
    target_drift = as_float(pushover.get("target_drift_ratio") or candidate.get("pushover_target_drift_ratio"))
    direction_spans = spans_y if direction == "Y" else spans_x

    return {
        "model_name": path.stem.replace("_metadata", ""),
        "direction": direction,
        "story_count": story_count,
        "x_bay_count": as_float(candidate.get("x_bay_count")),
        "y_bay_count": as_float(candidate.get("y_bay_count")),
        "bay_count_dir": as_float(candidate.get("y_bay_count") if direction == "Y" else candidate.get("x_bay_count")),
        "avg_span_x": mean(spans_x),
        "avg_span_y": mean(spans_y),
        "avg_span_dir": mean(direction_spans),
        "max_span_x": max(spans_x, default=0.0),
        "max_span_y": max(spans_y, default=0.0),
        "story_height": story_height,
        "total_height": story_count * story_height,
        "concrete_class": str(candidate.get("concrete_class") or ""),
        "concrete_fck": concrete_fck(candidate.get("concrete_class")),
        "steel_class": str(candidate.get("steel_class") or ""),
        "steel_fy": steel_fy(candidate.get("steel_class")),
        "column_width": column_width,
        "column_depth": column_depth,
        "column_area": column_width * column_depth,
        "beam_width": beam_width,
        "beam_depth": beam_depth,
        "beam_area": beam_width * beam_depth,
        "rho_col": as_float(candidate.get("rho_col")),
        "rho_beam_top": as_float(candidate.get("beam_top_ratio_support")),
        "rho_beam_bottom": as_float(candidate.get("beam_bottom_ratio_span")),
        "slab_thickness": as_float(candidate.get("slab_thickness_m")),
        "slab_rebar_ratio": as_float(candidate.get("slab_rebar_ratio")),
        "raft_thickness": as_float(candidate.get("raft_thickness_m")),
        "raft_rebar_ratio": as_float(candidate.get("raft_rebar_ratio")),
        "soil_class": str(candidate.get("soil_class") or ""),
        "subgrade_modulus": as_float(candidate.get("subgrade_modulus_kn_m3")),
        "has_shear_walls": "yes" if has_walls else "no",
        "wall_count": wall_count,
        "wall_area": wall_count * wall_thickness * wall_length,
        "wall_thickness": wall_thickness,
        "wall_length": wall_length,
        "wall_rebar_ratio": as_float(candidate.get("wall_rebar_ratio")) if has_walls else 0.0,
        "target_drift": target_drift,
        "target_displacement": as_float(pushover.get("target_displacement_m")),
        "peak_base_shear": as_float(curve.get("peak_base_shear_kn")),
        "max_story_drift_ratio": max_story_drift_ratio_for_direction(story_drifts, direction),
        "max_rotation": max_rotation_from_summary(summary),
        "lscp_count": lscp_count,
        "cp_count": cp_count,
        "lscp_ratio": lscp_ratio,
        "cp_ratio": cp_ratio,
        "has_lscp": 1 if lscp_count > 0 else 0,
        "has_cp": 1 if cp_count > 0 else 0,
        "fema_target_capacity_ratio": fema_ratio,
        "fema_capacity_status": fema_capacity_status(fema_ratio),
        "first_type": str(first.get("element_type") or "unknown"),
        "first_story": story_from_element(first.get("element_name")),
        "first_axis": first_location["axis"],
        "critical_state": state,
        "critical_rank": STATE_RANK.get(state, -1),
        "critical_type": str(critical.get("element_type") or "unknown"),
        "critical_story": story_from_element(critical.get("element_name")),
        "critical_axis": critical_location["axis"],
        "first_ls_step": as_float(first_ls.get("step_number"), math.nan) if first_ls else math.nan,
        "first_ls_exists": bool(first_ls),
    }


def model_row_from_directional(data: dict[str, Any], path: Path, x_row: dict[str, Any], y_row: dict[str, Any]) -> dict[str, Any]:
    """Build one row comparing X and Y results for the same model."""
    candidate = data.get("candidate", {}) if isinstance(data.get("candidate"), dict) else {}
    spans_x = [as_float(item) for item in candidate.get("spans_x", [])]
    spans_y = [as_float(item) for item in candidate.get("spans_y", [])]
    row = {key: x_row.get(key) for key in DIRECTIONAL_NUMERIC_FEATURES + DIRECTIONAL_CATEGORICAL_FEATURES if key not in {"direction", "bay_count_dir", "avg_span_dir", "peak_base_shear"}}
    row.update(
        {
            "model_name": path.stem.replace("_metadata", ""),
            "x_bay_count": as_float(candidate.get("x_bay_count")),
            "y_bay_count": as_float(candidate.get("y_bay_count")),
            "bay_count_diff_abs": abs(as_float(candidate.get("x_bay_count")) - as_float(candidate.get("y_bay_count"))),
            "bay_count_ratio": safe_divide(as_float(candidate.get("x_bay_count")), max(as_float(candidate.get("y_bay_count")), 1.0)),
            "avg_span_x": mean(spans_x),
            "avg_span_y": mean(spans_y),
            "avg_span_diff_abs": abs(mean(spans_x) - mean(spans_y)),
            "avg_span_ratio": safe_divide(mean(spans_x), max(mean(spans_y), 1e-9)),
            "max_span_x": max(spans_x, default=0.0),
            "max_span_y": max(spans_y, default=0.0),
            "max_span_diff_abs": abs(max(spans_x, default=0.0) - max(spans_y, default=0.0)),
            "plan_dim_x": sum(spans_x),
            "plan_dim_y": sum(spans_y),
            "plan_aspect_ratio": safe_divide(sum(spans_x), max(sum(spans_y), 1e-9)),
            "plan_area": sum(spans_x) * sum(spans_y),
            "x_rank": x_row.get("critical_rank", -1),
            "y_rank": y_row.get("critical_rank", -1),
            "x_state": x_row.get("critical_state", ""),
            "y_state": y_row.get("critical_state", ""),
            "x_critical_type": x_row.get("critical_type", ""),
            "y_critical_type": y_row.get("critical_type", ""),
            "x_critical_story": x_row.get("critical_story", 0),
            "y_critical_story": y_row.get("critical_story", 0),
            "x_critical_axis": x_row.get("critical_axis", ""),
            "y_critical_axis": y_row.get("critical_axis", ""),
        }
    )
    return row


def direction_summary(data: dict[str, Any], direction: str) -> dict[str, Any] | None:
    """Return plastic hinge summary for a direction."""
    plastic = data.get("plastic_hinges", {}) if isinstance(data.get("plastic_hinges"), dict) else {}
    results = plastic.get("results", {}) if isinstance(plastic.get("results"), dict) else {}
    summaries = results.get("summary_by_direction", {}) if isinstance(results.get("summary_by_direction"), dict) else {}
    summary = summaries.get(direction)
    return summary if isinstance(summary, dict) else None


def critical_event(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the most critical event in a direction summary."""
    direct = summary.get("critical_event")
    if isinstance(direct, dict) and direct:
        return direct
    events = summary.get("critical_events")
    if isinstance(events, list) and events:
        valid = [item for item in events if isinstance(item, dict)]
        return max(
            valid,
            key=lambda item: (
                STATE_RANK.get(str(item.get("hinge_state_level") or ""), -1),
                abs(as_float(item.get("plastic_rotation_rad"))),
                abs(as_float(item.get("demand_capacity_ratio"))),
                abs(as_float(item.get("moment_knm"))),
            ),
            default={},
        )
    first = summary.get("first_plastic_hinge")
    return first if isinstance(first, dict) else {}


def max_rotation_from_summary(summary: dict[str, Any]) -> float:
    """Return maximum absolute plastic rotation from critical events."""
    events = summary.get("critical_events")
    if not isinstance(events, list):
        events = []
    rotations = [abs(as_float(event.get("plastic_rotation_rad"))) for event in events if isinstance(event, dict)]
    first = summary.get("first_plastic_hinge")
    if isinstance(first, dict):
        rotations.append(abs(as_float(first.get("plastic_rotation_rad"))))
    return max(rotations, default=0.0)


def max_story_drift_ratio_for_direction(story_drifts: dict[str, Any], direction: str) -> float:
    """Return maximum interstory drift ratio for one direction."""
    if not isinstance(story_drifts, dict):
        return 0.0
    by_direction = story_drifts.get("by_direction", {}) if isinstance(story_drifts.get("by_direction"), dict) else {}
    drift = by_direction.get(direction) if isinstance(by_direction.get(direction), dict) else story_drifts.get(direction)
    if not isinstance(drift, dict):
        return 0.0
    max_drift = drift.get("max_drift")
    if isinstance(max_drift, dict):
        return as_float(max_drift.get("drift_ratio"))
    rows = drift.get("stories")
    if isinstance(rows, list):
        return max((as_float(row.get("drift_ratio")) for row in rows if isinstance(row, dict)), default=0.0)
    return 0.0


def event_location(event: dict[str, Any], spans_x: list[float], spans_y: list[float]) -> dict[str, Any]:
    """Classify event plan location as edge/middle."""
    numbers = [float(item) for item in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", str(event.get("element_name") or ""))]
    if len(numbers) < 3:
        return {"axis": "unknown"}
    x_coord = numbers[-2]
    y_coord = numbers[-1]
    x_max = sum(spans_x)
    y_max = sum(spans_y)
    tol = 1e-5
    axis = "edge" if abs(x_coord) < tol or abs(x_coord - x_max) < tol or abs(y_coord) < tol or abs(y_coord - y_max) < tol else "middle"
    return {"axis": axis}


def story_from_element(name: object) -> int:
    """Parse story index from generated frame element name."""
    match = re.search(r"^[A-Z]+_([0-9]+)_", str(name or ""))
    return int(match.group(1)) if match else 1


def target_matches(row: dict[str, Any], target: dict[str, Any], dataset_type: str) -> bool:
    """Return whether a parsed row belongs to the selected target behavior."""
    kind = target.get("kind")
    if kind == "damage_state":
        return row.get("critical_state") == target.get("state")
    if kind == "first_axis":
        return row.get("first_axis") == target.get("value")
    if kind == "first_type":
        return row.get("first_type") == target.get("value")
    if kind == "critical_type":
        return row.get("critical_type") == target.get("value")
    if kind == "critical_axis":
        return row.get("critical_axis") == target.get("value")
    if kind == "field_positive":
        return as_float(row.get(str(target.get("field") or ""))) > 0
    if kind == "field_equals":
        return str(row.get(str(target.get("field") or ""))) == str(target.get("value") or "")
    if kind == "critical_upper_story":
        return int(row.get("critical_story") or 1) >= 2
    if kind == "first_story_ge":
        return int(row.get("first_story") or 1) >= int(target.get("value") or 3)
    if kind == "first_type_critical_type":
        return row.get("first_type") == target.get("first") and row.get("critical_type") == target.get("critical")
    if kind == "first_upper_critical_upper":
        return int(row.get("first_story") or 1) >= 2 and int(row.get("critical_story") or 1) >= 2
    if kind == "first_axis_critical_axis":
        return row.get("first_axis") == target.get("first") and row.get("critical_axis") == target.get("critical")
    if kind == "compound_transition":
        return (
            row.get("first_type") == target.get("first_type")
            and row.get("first_axis") == target.get("first_axis")
            and row.get("critical_type") == target.get("critical_type")
            and row.get("critical_axis") == target.get("critical_axis")
        )
    if kind == "low_drift_high_damage":
        return as_float(row.get("target_drift")) <= 0.020 and str(row.get("critical_state")) in HIGH_STATES
    if kind == "high_drift_limited_damage":
        return as_float(row.get("target_drift")) >= 0.035 and str(row.get("critical_state")) in LIMITED_STATES
    if kind == "early_ls_step":
        return bool(row.get("first_ls_exists")) and as_float(row.get("first_ls_step"), math.inf) <= as_float(target.get("value"), 5)
    if kind == "high_drift_only_bio":
        return as_float(row.get("target_drift")) >= 0.035 and row.get("critical_state") == "B-IO"
    if dataset_type == "model":
        if kind == "damage_rank_diff_ge":
            return abs(int(row.get("x_rank") or -1) - int(row.get("y_rank") or -1)) >= int(target.get("value") or 2)
        if kind == "one_high_one_limited":
            return (row.get("x_state") in HIGH_STATES and row.get("y_state") in LIMITED_STATES) or (
                row.get("y_state") in HIGH_STATES and row.get("x_state") in LIMITED_STATES
            )
        if kind == "critical_field_diff":
            field = str(target.get("field") or "")
            return row.get(f"x_{field}") != row.get(f"y_{field}")
    return False


def train_tree_models(
    rows: list[dict[str, Any]],
    labels: list[int],
    dataset_type: str,
    algorithms: list[str],
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Train tree classifiers and return metrics and parameter profiles."""
    try:
        import pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(f"Required ML package is not installed: {exc}") from exc

    warnings.filterwarnings("ignore")
    numeric_features = MODEL_NUMERIC_FEATURES if dataset_type == "model" else [item for item in DIRECTIONAL_NUMERIC_FEATURES if item not in OUTPUT_NUMERIC_FEATURES]
    categorical_features = MODEL_CATEGORICAL_FEATURES if dataset_type == "model" else DIRECTIONAL_CATEGORICAL_FEATURES
    features = numeric_features + categorical_features
    frame = pd.DataFrame(rows)
    for key in features:
        if key not in frame:
            frame[key] = "" if key in categorical_features else 0.0
    x_frame = frame[features].copy()
    y_values = pd.Series(labels, dtype=int)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        [("num", "passthrough", numeric_features), ("cat", encoder, categorical_features)],
        verbose_feature_names_out=False,
    )

    test_size = 0.25 if min(sum(labels), len(labels) - sum(labels)) >= 10 else 0.35
    x_train, x_test, y_train, y_test = train_test_split(x_frame, y_values, test_size=test_size, random_state=random_seed, stratify=y_values)
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=70,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=2,
        ),
        "lightgbm": LGBMClassifier(n_estimators=90, num_leaves=24, learning_rate=0.08, class_weight="balanced", random_state=random_seed, n_jobs=2, verbose=-1),
    }
    algorithm_labels = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}

    metrics: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    for algorithm in algorithms:
        classifier = classifiers[algorithm]
        pipeline = Pipeline([("pre", preprocessor), ("clf", classifier)])
        if algorithm == "xgboost":
            positive = int(y_train.sum())
            pipeline.set_params(clf__scale_pos_weight=(len(y_train) - positive) / max(positive, 1))
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics.append(
            {
                "algorithm": algorithm,
                "model": algorithm_labels[algorithm],
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "pr_auc": float(average_precision_score(y_test, probabilities)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            }
        )
        importances.extend(aggregate_feature_importance(pipeline, features, categorical_features, algorithm, algorithm_labels[algorithm]))

    profiles = build_feature_profiles(frame, labels, importances, numeric_features, categorical_features)
    return metrics, importances, profiles


def train_tree_regressors(
    rows: list[dict[str, Any]],
    values: list[float],
    dataset_type: str,
    algorithms: list[str],
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Train tree regressors and return metrics and parameter profiles."""
    try:
        import pandas as pd
        from lightgbm import LGBMRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(f"Gerekli ML paketi kurulu degil: {exc}") from exc

    warnings.filterwarnings("ignore")
    numeric_features = MODEL_NUMERIC_FEATURES if dataset_type == "model" else [item for item in DIRECTIONAL_NUMERIC_FEATURES if item not in OUTPUT_NUMERIC_FEATURES]
    categorical_features = MODEL_CATEGORICAL_FEATURES if dataset_type == "model" else DIRECTIONAL_CATEGORICAL_FEATURES
    features = numeric_features + categorical_features
    frame = pd.DataFrame(rows)
    for key in features:
        if key not in frame:
            frame[key] = "" if key in categorical_features else 0.0
    x_frame = frame[features].copy()
    y_values = pd.Series(values, dtype=float)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        [("num", "passthrough", numeric_features), ("cat", encoder, categorical_features)],
        verbose_feature_names_out=False,
    )

    x_train, x_test, y_train, y_test = train_test_split(x_frame, y_values, test_size=0.25, random_state=random_seed)
    regressors = {
        "random_forest": RandomForestRegressor(
            n_estimators=90,
            max_depth=16,
            min_samples_leaf=4,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=110,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=2,
        ),
        "lightgbm": LGBMRegressor(n_estimators=110, num_leaves=28, learning_rate=0.06, random_state=random_seed, n_jobs=2, verbose=-1),
    }
    algorithm_labels = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}

    metrics: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    for algorithm in algorithms:
        regressor = regressors[algorithm]
        pipeline = Pipeline([("pre", preprocessor), ("clf", regressor)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        mse = float(mean_squared_error(y_test, predictions))
        metrics.append(
            {
                "algorithm": algorithm,
                "model": algorithm_labels[algorithm],
                "r2": float(r2_score(y_test, predictions)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": math.sqrt(mse),
            }
        )
        importances.extend(aggregate_feature_importance(pipeline, features, categorical_features, algorithm, algorithm_labels[algorithm]))

    profiles = build_regression_feature_profiles(frame, values, importances, numeric_features, categorical_features)
    return metrics, importances, profiles


def aggregate_feature_importance(pipeline: Any, features: list[str], categorical_features: list[str], algorithm: str, model_label: str) -> list[dict[str, Any]]:
    """Aggregate encoded tree importances back to original feature names."""
    names = list(pipeline.named_steps["pre"].get_feature_names_out())
    raw = getattr(pipeline.named_steps["clf"], "feature_importances_", None)
    if raw is None:
        return []
    values = {feature: 0.0 for feature in features}
    for name, importance in zip(names, raw):
        clean = str(name)
        if clean in values:
            values[clean] += float(importance)
            continue
        for categorical in categorical_features:
            if clean.startswith(f"{categorical}_"):
                values[categorical] += float(importance)
                break
    total = sum(values.values()) or 1.0
    return [
        {
            "algorithm": algorithm,
            "model": model_label,
            "feature": key,
            "feature_label": FEATURE_LABELS.get(key, key),
            "importance": value,
            "importance_share": value / total,
        }
        for key, value in sorted(values.items(), key=lambda item: item[1], reverse=True)[:15]
    ]


def build_feature_profiles(
    frame: Any,
    labels: list[int],
    importances: list[dict[str, Any]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[dict[str, Any]]:
    """Build class-vs-rest profile for the most important features."""
    import pandas as pd

    label_series = pd.Series(labels, index=frame.index)
    positive = frame[label_series == 1]
    other = frame[label_series == 0]
    ranked_features: list[str] = []
    for item in sorted(importances, key=lambda row: row["importance_share"], reverse=True):
        feature = str(item["feature"])
        if feature not in ranked_features:
            ranked_features.append(feature)
        if len(ranked_features) >= 12:
            break

    rows: list[dict[str, Any]] = []
    for feature in ranked_features:
        if feature in numeric_features:
            pos_mean = float(positive[feature].astype(float).mean())
            other_mean = float(other[feature].astype(float).mean())
            general_mean = float(frame[feature].astype(float).mean())
            direction = "higher" if pos_mean > general_mean else "lower" if pos_mean < general_mean else "similar"
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "positive": format_feature_value(feature, pos_mean),
                    "others": format_feature_value(feature, other_mean),
                    "general": format_feature_value(feature, general_mean),
                    "direction": direction,
                    "comment": profile_comment(FEATURE_LABELS.get(feature, feature), direction),
                }
            )
        elif feature in categorical_features:
            pos_mode = mode_value(positive[feature].tolist())
            other_mode = mode_value(other[feature].tolist())
            general_mode = mode_value(frame[feature].tolist())
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "positive": str(pos_mode),
                    "others": str(other_mode),
                    "general": str(general_mode),
                    "direction": "class shift",
                    "comment": f"The selected behavior is mostly represented by {pos_mode} for {FEATURE_LABELS.get(feature, feature)}.",
                }
            )
    return rows


def build_regression_feature_profiles(
    frame: Any,
    values: list[float],
    importances: list[dict[str, Any]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[dict[str, Any]]:
    """Build high-target vs low-target profile for important regression features."""
    import pandas as pd

    target_series = pd.Series(values, index=frame.index, dtype=float)
    low_limit = float(target_series.quantile(0.25))
    high_limit = float(target_series.quantile(0.75))
    high = frame[target_series >= high_limit]
    low = frame[target_series <= low_limit]
    ranked_features: list[str] = []
    for item in sorted(importances, key=lambda row: row["importance_share"], reverse=True):
        feature = str(item["feature"])
        if feature not in ranked_features:
            ranked_features.append(feature)
        if len(ranked_features) >= 12:
            break

    rows: list[dict[str, Any]] = []
    for feature in ranked_features:
        if feature in numeric_features:
            high_mean = float(high[feature].astype(float).mean())
            low_mean = float(low[feature].astype(float).mean())
            general_mean = float(frame[feature].astype(float).mean())
            direction = "higher" if high_mean > low_mean else "lower" if high_mean < low_mean else "similar"
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "positive": format_feature_value(feature, high_mean),
                    "others": format_feature_value(feature, low_mean),
                    "general": format_feature_value(feature, general_mean),
                    "direction": direction,
                    "comment": regression_profile_comment(FEATURE_LABELS.get(feature, feature), direction),
                }
            )
        elif feature in categorical_features:
            high_mode = mode_value(high[feature].tolist())
            low_mode = mode_value(low[feature].tolist())
            general_mode = mode_value(frame[feature].tolist())
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "positive": str(high_mode),
                    "others": str(low_mode),
                    "general": str(general_mode),
                    "direction": "class shift",
                    "comment": f"High target records are mostly represented by {high_mode} for {FEATURE_LABELS.get(feature, feature)}.",
                }
            )
    return rows


def build_comments(target_label: str, positive_count: int, total_count: int, metrics: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> list[str]:
    """Create dashboard-friendly interpretation comments."""
    best_roc = max((row["roc_auc"] for row in metrics), default=0.0)
    best_pr = max((row["pr_auc"] for row in metrics), default=0.0)
    rate = positive_count / max(total_count, 1)
    if positive_count < 30:
        reliability = "The positive sample count is low; interpret the parameter trends as limited-confidence evidence."
    elif positive_count < 100:
        reliability = "The behavior is rare; parametric interpretation is possible, but the limited-confidence note should be retained."
    else:
        reliability = "The sample count is sufficient for parametric interpretation."
    if best_roc >= 0.90 and best_pr >= 0.40:
        separability = "The models separate the selected behavior strongly."
    elif best_roc >= 0.80:
        separability = "The models produce a moderate discriminative signal for the selected behavior."
    else:
        separability = "Model performance is limited; the behavior is not strongly separated by individual parameters."
    top = ", ".join(item["feature_label"] for item in profiles[:5]) or "no dominant parameter"
    return [
        f"For {target_label}, the positive class contains {positive_count}/{total_count} records ({rate * 100:.2f}%).",
        f"{separability} The highest ROC-AUC is {best_roc:.3f}, and the highest PR-AUC is {best_pr:.3f}.",
        f"Most representative parameters: {top}.",
        reliability,
    ]


def build_regression_comments(target_label: str, sample_count: int, metrics: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> list[str]:
    """Create dashboard comments for numeric structural response targets."""
    best_r2 = max((row["r2"] for row in metrics), default=0.0)
    best_rmse = min((row["rmse"] for row in metrics), default=0.0)
    if best_r2 >= 0.80:
        fit_text = "The numeric target is strongly predictable from the selected structural parameters."
    elif best_r2 >= 0.50:
        fit_text = "The numeric target has a moderate parametric signal; interpret trends rather than exact values."
    else:
        fit_text = "The numeric target is weakly predicted; use the results as exploratory parameter ranking."
    top = ", ".join(item["feature_label"] for item in profiles[:5]) or "no dominant parameter"
    return [
        f"{target_label} was evaluated as a regression target using {sample_count} valid directional observations.",
        f"{fit_text} Best R2 is {best_r2:.3f}; lowest RMSE is {best_rmse:.3g}.",
        f"Most representative parameters: {top}.",
        "The profile table compares the upper quartile of the target with the lower quartile, not a positive/negative class.",
    ]


def write_behavior_report(output_dir: Path, group_id: str, target_id: str, summary: dict[str, Any]) -> None:
    """Persist latest behavior report as JSON for article traceability."""
    report_dir = Path(output_dir) / "behavior_ml_reports"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{group_id}_{target_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def as_float(value: object, default: float = 0.0) -> float:
    """Parse a numeric value."""
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    """Return mean of values."""
    return sum(values) / len(values) if values else 0.0


def safe_divide(left: float, right: float) -> float:
    """Return safe division."""
    return left / right if abs(right) > 1e-12 else 0.0


def safe_ratio_or_nan(left: float, right: float) -> float:
    """Return ratio, preserving missing/invalid denominator as NaN."""
    if not math.isfinite(left) or not math.isfinite(right) or abs(right) <= 1e-12:
        return math.nan
    return left / right


def fema_capacity_status(ratio: float) -> str:
    """Return the same FEMA capacity status used in the dashboard charts."""
    if not math.isfinite(ratio):
        return "not_available"
    if ratio <= 0.9:
        return "capacity_ok"
    if ratio <= 1.1:
        return "near_capacity"
    return "capacity_exceeded"


def concrete_fck(value: object) -> float:
    """Extract concrete fck from class label."""
    match = re.search(r"(\d+)", str(value or ""))
    return float(match.group(1)) if match else 30.0


def steel_fy(value: object) -> float:
    """Return steel yield stress from class label."""
    return 500.0 if "500" in str(value or "") else 420.0


def mode_value(values: list[Any]) -> Any:
    """Return most common value."""
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "-"


def format_feature_value(feature: str, value: float) -> str:
    """Format a feature mean for the UI."""
    if feature in {"rho_col", "rho_beam_top", "rho_beam_bottom", "slab_rebar_ratio", "raft_rebar_ratio", "wall_rebar_ratio", "target_drift", "max_story_drift_ratio"}:
        return f"{value * 100:.2f}%"
    if feature.endswith("_area") or feature in {"plan_area"}:
        return f"{value:.3g} m^2"
    if "modulus" in feature:
        return f"{value:.0f}"
    if "count" in feature or feature == "story_count":
        return f"{value:.2f}"
    return f"{value:.3g}"


def profile_comment(label: str, direction: str) -> str:
    """Return short profile comment."""
    if direction == "similar":
        return f"{label} is close to the overall mean in the selected class."
    return f"{label} is {direction} in the selected class than in the overall dataset."


def regression_profile_comment(label: str, direction: str) -> str:
    """Return short high-target vs low-target profile comment."""
    if direction == "similar":
        return f"{label} is similar in the upper and lower target quartiles."
    if direction == "higher":
        return f"{label} is higher in the upper target quartile than in the lower quartile."
    return f"{label} is lower in the upper target quartile than in the lower quartile."
