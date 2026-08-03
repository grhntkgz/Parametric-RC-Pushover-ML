"""Extract paper-oriented capacity-curve metrics from metadata JSON files.

The script reads previously generated metadata files and converts X/Y pushover
capacity curves into generalized metrics that are easier to discuss in a
manuscript than thousands of individual curves.

Outputs are written as CSV, SVG, and a short Markdown report. The implementation
uses only the Python standard library so it can run in reviewer environments
without plotting/data-science dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


STATE_RANK = {"A-B": 0, "B-IO": 1, "IO-LS": 2, "LS-CP": 3, "CP-C": 4, "C-D": 5, "D-E": 6, "beyond E": 7, "Beyond E": 7}
CSV_FLOAT_PRECISION = 6


@dataclass(frozen=True)
class CapacityPoint:
    """One positive pushover capacity-curve point."""

    step_number: float
    displacement_m: float
    base_shear_kn: float
    source: str


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Extract capacity-curve metrics from SAP2000 metadata JSON files.")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path.home() / "Desktop" / "metadata",
        help="Directory containing *_metadata.json files. Default: Desktop/metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "capacity_curve_analysis",
        help="Directory where CSV/SVG/report outputs will be written.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Optional limit for quick tests. 0 means all files.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = metadata_json_paths(args.metadata_dir)
    if args.max_files > 0:
        paths = paths[: args.max_files]

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(rows_from_metadata_file(path))

    rows = dedupe_directional_rows(rows)
    model_rows = xy_asymmetry_rows(rows)
    damage_summary = summarize_rows(rows, "critical_state")
    type_summary = summarize_rows(rows, "critical_type")

    write_csv(output_dir / "capacity_curve_directional_metrics.csv", rows)
    write_csv(output_dir / "capacity_curve_xy_asymmetry.csv", model_rows)
    write_csv(output_dir / "capacity_curve_summary_by_damage_class.csv", damage_summary)
    write_csv(output_dir / "capacity_curve_summary_by_critical_type.csv", type_summary)
    write_group_curve_svg(output_dir / "normalized_capacity_curves_by_damage_class.svg", rows, "critical_state")
    write_group_curve_svg(output_dir / "normalized_capacity_curves_by_critical_type.svg", rows, "critical_type")
    write_report(output_dir / "capacity_curve_report.md", rows, model_rows, damage_summary, type_summary, args.metadata_dir)

    print(f"Read metadata files: {len(paths)}")
    print(f"Directional curve rows: {len(rows)}")
    print(f"Model X/Y asymmetry rows: {len(model_rows)}")
    print(f"Output directory: {output_dir}")


def metadata_json_paths(root: Path) -> list[Path]:
    """Return metadata JSON files from a flat folder or nested run folders."""

    root = Path(root)
    if not root.exists():
        return []
    paths = [path for path in root.glob("*_metadata.json") if path.is_file()]
    paths.extend(path for path in root.glob("run_*/*_metadata.json") if path.is_file())
    if not paths:
        paths.extend(path for path in root.rglob("*_metadata.json") if path.is_file())
    return sorted(set(paths))


def rows_from_metadata_file(path: Path) -> list[dict[str, Any]]:
    """Build X/Y capacity rows from one metadata file."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if str(data.get("status") or "") != "success":
        return []

    model_name = model_name_from_metadata(path, data)
    candidate = data.get("candidate", {}) if isinstance(data.get("candidate"), dict) else {}
    pushover = data.get("pushover", {}) if isinstance(data.get("pushover"), dict) else {}
    results = pushover.get("results", {}) if isinstance(pushover.get("results"), dict) else {}
    curves = results.get("curves", {}) if isinstance(results.get("curves"), dict) else {}

    rows: list[dict[str, Any]] = []
    for direction in ("X", "Y"):
        curve = curves.get(direction, {}) if isinstance(curves.get(direction), dict) else {}
        points = capacity_points(curve.get("points", []))
        if len(points) < 2:
            continue
        metrics = curve_metrics(points)
        summary = direction_summary(data, direction)
        critical = critical_event(summary)
        state_counts = summary.get("state_counts", {}) if isinstance(summary.get("state_counts"), dict) else {}
        row = {
            "model_name": model_name,
            "direction": direction,
            "metadata_file": str(path),
            "story_count": as_float(candidate.get("story_count")),
            "total_height_m": as_float(candidate.get("story_count")) * as_float(candidate.get("story_height")),
            "x_bay_count": as_float(candidate.get("x_bay_count")),
            "y_bay_count": as_float(candidate.get("y_bay_count")),
            "concrete_class": str(candidate.get("concrete_class") or ""),
            "column_section": section_label(candidate.get("column_section")),
            "beam_section": section_label(candidate.get("beam_section")),
            "rho_col": as_float(candidate.get("rho_col"), math.nan),
            "rho_beam_top": as_float(candidate.get("beam_top_ratio_support"), math.nan),
            "rho_beam_bottom": as_float(candidate.get("beam_bottom_ratio_span"), math.nan),
            "has_shear_walls": "yes" if candidate.get("has_shear_walls") else "no",
            "wall_area_m2": wall_area(candidate),
            "soil_class": str(candidate.get("soil_class") or ""),
            "subgrade_modulus_kn_m3": as_float(candidate.get("subgrade_modulus_kn_m3"), math.nan),
            "target_drift_ratio": as_float(pushover.get("target_drift_ratio") or candidate.get("pushover_target_drift_ratio"), math.nan),
            "target_displacement_m": as_float(pushover.get("target_displacement_m"), math.nan),
            "critical_state": str(critical.get("hinge_state_level") or "unknown"),
            "critical_rank": STATE_RANK.get(str(critical.get("hinge_state_level") or ""), -1),
            "critical_type": str(critical.get("element_type") or "unknown"),
            "lscp_event_count": as_float(state_counts.get("LS-CP")),
            "cp_event_count": sum(as_float(state_counts.get(item)) for item in ("CP-C", "C-D", "D-E", "beyond E", "Beyond E")),
            "curve_source_type": curve_source_type(curve),
            "curve_point_count": len(points),
            "curve_note": str(curve.get("curve_note") or curve.get("message") or ""),
        }
        row.update(metrics)
        row["normalized_curve_points"] = normalized_curve_serialized(points)
        rows.append(row)
    return rows


def model_name_from_metadata(path: Path, data: dict[str, Any]) -> str:
    """Return a stable model name."""

    model_path = str(data.get("model_path") or "")
    if model_path:
        return Path(model_path).stem
    return path.stem.replace("_metadata", "")


def capacity_points(raw_points: Any) -> list[CapacityPoint]:
    """Normalize raw capacity points to positive, sorted, unique displacement points."""

    if not isinstance(raw_points, list):
        return []
    points: list[CapacityPoint] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        displacement = abs(as_float(item.get("control_displacement_m"), math.nan))
        shear = abs(as_float(item.get("base_shear_kn"), math.nan))
        if not math.isfinite(displacement) or not math.isfinite(shear):
            continue
        points.append(
            CapacityPoint(
                step_number=as_float(item.get("step_number") if item.get("step_number") is not None else item.get("step")),
                displacement_m=displacement,
                base_shear_kn=shear,
                source=str(item.get("source") or item.get("load_step") or ""),
            )
        )
    points.sort(key=lambda point: (point.displacement_m, point.base_shear_kn))

    unique: list[CapacityPoint] = []
    for point in points:
        if unique and abs(point.displacement_m - unique[-1].displacement_m) < 1e-9:
            if point.base_shear_kn >= unique[-1].base_shear_kn:
                unique[-1] = point
        else:
            unique.append(point)
    if not unique or unique[0].displacement_m > 1e-9:
        unique.insert(0, CapacityPoint(0.0, 0.0, 0.0, "origin"))
    return unique


def curve_metrics(points: list[CapacityPoint]) -> dict[str, float]:
    """Calculate generalized capacity-curve metrics."""

    peak_point = max(points, key=lambda point: point.base_shear_kn)
    final_point = points[-1]
    final_disp = final_point.displacement_m
    peak_shear = peak_point.base_shear_kn
    area = trapezoid_area(points)
    initial_stiffness = initial_stiffness_kn_per_m(points)
    final_secant = safe_divide(final_point.base_shear_kn, final_disp)
    yield_disp = displacement_at_shear(points, 0.6 * peak_shear)
    ductility = safe_divide(final_disp, yield_disp)
    normalized_energy = safe_divide(area, peak_shear * final_disp)
    return {
        "initial_stiffness_kn_per_m": initial_stiffness,
        "peak_base_shear_kn": peak_shear,
        "peak_displacement_m": peak_point.displacement_m,
        "final_displacement_m": final_disp,
        "final_base_shear_kn": final_point.base_shear_kn,
        "post_peak_strength_ratio": safe_divide(final_point.base_shear_kn, peak_shear),
        "area_under_curve_kn_m": area,
        "normalized_energy": normalized_energy,
        "yield_displacement_proxy_m": yield_disp,
        "ductility_proxy": ductility,
        "final_secant_stiffness_kn_per_m": final_secant,
        "stiffness_degradation_proxy": safe_divide(final_secant, initial_stiffness),
    }


def initial_stiffness_kn_per_m(points: list[CapacityPoint]) -> float:
    """Estimate initial stiffness by fitting the early positive segment through the origin."""

    final_disp = points[-1].displacement_m
    if final_disp <= 0:
        return 0.0
    nonzero = [point for point in points if point.displacement_m > 1e-9 and point.base_shear_kn > 0]
    if not nonzero:
        return 0.0
    limit = max(0.15 * final_disp, nonzero[0].displacement_m)
    early = [point for point in nonzero if point.displacement_m <= limit]
    if len(early) < 2:
        early = nonzero[: min(3, len(nonzero))]
    denominator = sum(point.displacement_m**2 for point in early)
    if denominator <= 0:
        return safe_divide(early[0].base_shear_kn, early[0].displacement_m)
    return sum(point.displacement_m * point.base_shear_kn for point in early) / denominator


def displacement_at_shear(points: list[CapacityPoint], target_shear: float) -> float:
    """Interpolate the first displacement where the curve reaches a target shear."""

    if target_shear <= 0:
        return 0.0
    previous = points[0]
    for point in points[1:]:
        if point.base_shear_kn >= target_shear:
            span = point.base_shear_kn - previous.base_shear_kn
            if abs(span) < 1e-12:
                return point.displacement_m
            ratio = (target_shear - previous.base_shear_kn) / span
            return previous.displacement_m + ratio * (point.displacement_m - previous.displacement_m)
        previous = point
    return points[-1].displacement_m


def trapezoid_area(points: list[CapacityPoint]) -> float:
    """Return area under the capacity curve."""

    area = 0.0
    for left, right in zip(points, points[1:]):
        width = max(0.0, right.displacement_m - left.displacement_m)
        area += 0.5 * (left.base_shear_kn + right.base_shear_kn) * width
    return area


def direction_summary(data: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return plastic-hinge summary for one direction."""

    plastic = data.get("plastic_hinges", {}) if isinstance(data.get("plastic_hinges"), dict) else {}
    results = plastic.get("results", {}) if isinstance(plastic.get("results"), dict) else {}
    summaries = results.get("summary_by_direction", {}) if isinstance(results.get("summary_by_direction"), dict) else {}
    summary = summaries.get(direction)
    return summary if isinstance(summary, dict) else {}


def critical_event(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the most critical hinge event from a direction summary."""

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
                abs(as_float(item.get("moment_knm") or item.get("moment_kn_m"))),
            ),
            default={},
        )
    first = summary.get("first_plastic_hinge")
    return first if isinstance(first, dict) else {}


def curve_source_type(curve: dict[str, Any]) -> str:
    """Classify whether points are saved steps or envelope-derived estimates."""

    note = str(curve.get("curve_note") or curve.get("message") or "").lower()
    if "envelope" in note:
        return "envelope_monotonic_estimate"
    points = curve.get("points", [])
    if isinstance(points, list) and any(str(item.get("source") or "") == "matched_step" for item in points if isinstance(item, dict)):
        return "saved_step_history"
    return "unknown"


def normalized_curve_serialized(points: list[CapacityPoint], bins: int = 40) -> str:
    """Serialize a normalized V/Vmax versus Delta/Delta_final curve as JSON."""

    peak = max((point.base_shear_kn for point in points), default=0.0)
    final_disp = points[-1].displacement_m if points else 0.0
    if peak <= 0 or final_disp <= 0:
        return "[]"
    samples = []
    for index in range(bins + 1):
        x = index / bins
        displacement = x * final_disp
        shear = interpolate_shear(points, displacement)
        samples.append([round(x, 4), round(shear / peak, 4)])
    return json.dumps(samples, separators=(",", ":"))


def interpolate_shear(points: list[CapacityPoint], displacement: float) -> float:
    """Interpolate base shear at a displacement."""

    if not points:
        return 0.0
    if displacement <= points[0].displacement_m:
        return points[0].base_shear_kn
    previous = points[0]
    for point in points[1:]:
        if displacement <= point.displacement_m:
            span = point.displacement_m - previous.displacement_m
            if abs(span) < 1e-12:
                return point.base_shear_kn
            ratio = (displacement - previous.displacement_m) / span
            return previous.base_shear_kn + ratio * (point.base_shear_kn - previous.base_shear_kn)
        previous = point
    return points[-1].base_shear_kn


def xy_asymmetry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build model-level X/Y capacity-asymmetry metrics."""

    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model[str(row.get("model_name"))][str(row.get("direction"))] = row

    output: list[dict[str, Any]] = []
    for model_name, directions in by_model.items():
        if "X" not in directions or "Y" not in directions:
            continue
        x_row = directions["X"]
        y_row = directions["Y"]
        output.append(
            {
                "model_name": model_name,
                "peak_base_shear_x_kn": x_row["peak_base_shear_kn"],
                "peak_base_shear_y_kn": y_row["peak_base_shear_kn"],
                "peak_base_shear_xy_ratio": ratio_larger_to_smaller(x_row["peak_base_shear_kn"], y_row["peak_base_shear_kn"]),
                "initial_stiffness_x_kn_per_m": x_row["initial_stiffness_kn_per_m"],
                "initial_stiffness_y_kn_per_m": y_row["initial_stiffness_kn_per_m"],
                "initial_stiffness_xy_ratio": ratio_larger_to_smaller(x_row["initial_stiffness_kn_per_m"], y_row["initial_stiffness_kn_per_m"]),
                "normalized_energy_x": x_row["normalized_energy"],
                "normalized_energy_y": y_row["normalized_energy"],
                "normalized_energy_xy_ratio": ratio_larger_to_smaller(x_row["normalized_energy"], y_row["normalized_energy"]),
                "damage_rank_x": x_row["critical_rank"],
                "damage_rank_y": y_row["critical_rank"],
                "damage_rank_diff_abs": abs(as_float(x_row["critical_rank"]) - as_float(y_row["critical_rank"])),
                "critical_state_x": x_row["critical_state"],
                "critical_state_y": y_row["critical_state"],
                "critical_type_x": x_row["critical_type"],
                "critical_type_y": y_row["critical_type"],
            }
        )
    return output


def summarize_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    """Summarize capacity metrics by a categorical group."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        summary_rows.append(
            {
                group_key: group,
                "count": len(group_rows),
                "share": safe_divide(len(group_rows), len(rows)),
                "avg_initial_stiffness_kn_per_m": average(group_rows, "initial_stiffness_kn_per_m"),
                "avg_peak_base_shear_kn": average(group_rows, "peak_base_shear_kn"),
                "avg_final_displacement_m": average(group_rows, "final_displacement_m"),
                "avg_post_peak_strength_ratio": average(group_rows, "post_peak_strength_ratio"),
                "avg_normalized_energy": average(group_rows, "normalized_energy"),
                "avg_ductility_proxy": average(group_rows, "ductility_proxy"),
                "avg_stiffness_degradation_proxy": average(group_rows, "stiffness_degradation_proxy"),
                "envelope_estimate_share": share_equal(group_rows, "curve_source_type", "envelope_monotonic_estimate"),
            }
        )
    return summary_rows


def write_group_curve_svg(path: Path, rows: list[dict[str, Any]], group_key: str) -> None:
    """Write an SVG of mean normalized capacity curves by group."""

    curves = group_average_curves(rows, group_key)
    if not curves:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    width, height = 920, 560
    left, right, top, bottom = 78, 30, 40, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#176b87", "#e11d48", "#2f855a", "#f59e0b", "#6d28d9", "#0f766e", "#9f1239", "#475569"]

    def px(x: float) -> float:
        return left + x * plot_w

    def py(y: float) -> float:
        return top + (1.05 - y) / 1.05 * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Normalized capacity curves">',
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1.4}.grid{stroke:#e5e7eb}.curve{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.legend text{font-size:13px}.title{font-size:20px;font-weight:700}.label{font-size:14px;font-weight:700}</style>",
        f'<text x="{left}" y="24" class="title">Mean normalized capacity curves by {escape_xml(group_key.replace("_", " "))}</text>',
    ]
    for tick in range(0, 6):
        x_value = tick / 5
        y_value = tick / 5
        lines.append(f'<line x1="{px(x_value):.1f}" y1="{top}" x2="{px(x_value):.1f}" y2="{top + plot_h}" class="grid"/>')
        lines.append(f'<line x1="{left}" y1="{py(y_value):.1f}" x2="{left + plot_w}" y2="{py(y_value):.1f}" class="grid"/>')
        lines.append(f'<text x="{px(x_value) - 9:.1f}" y="{top + plot_h + 24}" font-size="12">{x_value:.1f}</text>')
        lines.append(f'<text x="{left - 42}" y="{py(y_value) + 4:.1f}" font-size="12">{y_value:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<text x="{left + plot_w / 2 - 80:.1f}" y="{height - 22}" class="label">Normalized roof displacement</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2 + 80:.1f}" class="label" transform="rotate(-90 18 {top + plot_h / 2 + 80:.1f})">Normalized base shear</text>')

    for index, (group, values) in enumerate(curves.items()):
        color = colors[index % len(colors)]
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in values["curve"])
        lines.append(f'<polyline points="{points}" class="curve" stroke="{color}"><title>{escape_xml(group)} / n={values["count"]}</title></polyline>')
        legend_y = 58 + index * 22
        lines.append(f'<g class="legend"><line x1="{width - 235}" y1="{legend_y}" x2="{width - 205}" y2="{legend_y}" stroke="{color}" stroke-width="3"/><text x="{width - 196}" y="{legend_y + 4}">{escape_xml(group)} (n={values["count"]})</text></g>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def group_average_curves(rows: list[dict[str, Any]], group_key: str, max_groups: int = 6) -> dict[str, dict[str, Any]]:
    """Return mean normalized curves for the largest groups."""

    grouped: dict[str, list[list[list[float]]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        try:
            curve = json.loads(str(row.get("normalized_curve_points") or "[]"))
        except json.JSONDecodeError:
            curve = []
        if not curve:
            continue
        group = str(row.get(group_key) or "unknown")
        grouped[group].append(curve)
        counts[group] += 1

    top_groups = sorted(grouped, key=lambda key: (-counts[key], key))[:max_groups]
    output: dict[str, dict[str, Any]] = {}
    for group in top_groups:
        curves = grouped[group]
        length = min(len(curve) for curve in curves)
        averaged = []
        for index in range(length):
            x_value = sum(curve[index][0] for curve in curves) / len(curves)
            y_value = sum(curve[index][1] for curve in curves) / len(curves)
            averaged.append((x_value, y_value))
        output[group] = {"count": counts[group], "curve": averaged}
    return output


def write_report(path: Path, rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], damage_summary: list[dict[str, Any]], type_summary: list[dict[str, Any]], metadata_dir: Path) -> None:
    """Write a compact Markdown report for manuscript drafting."""

    envelope_share = share_equal(rows, "curve_source_type", "envelope_monotonic_estimate")
    mean_peak = mean([as_float(row.get("peak_base_shear_kn"), math.nan) for row in rows])
    mean_energy = mean([as_float(row.get("normalized_energy"), math.nan) for row in rows])
    mean_ductility = mean([as_float(row.get("ductility_proxy"), math.nan) for row in rows])
    mean_shear_ratio = mean([as_float(row.get("peak_base_shear_xy_ratio"), math.nan) for row in model_rows])
    mean_stiffness_ratio = mean([as_float(row.get("initial_stiffness_xy_ratio"), math.nan) for row in model_rows])

    lines = [
        "# Capacity Curve Derived Metrics",
        "",
        f"- Metadata source: `{metadata_dir}`",
        f"- Directional observations: {len(rows)}",
        f"- Model-level X/Y pairs: {len(model_rows)}",
        f"- Envelope-derived monotonic estimate share: {format_percent(envelope_share)}",
        "",
        "## Overall Indicators",
        "",
        f"- Mean peak base shear: {mean_peak:.1f} kN",
        f"- Mean normalized energy: {mean_energy:.3f}",
        f"- Mean ductility proxy: {mean_ductility:.3f}",
        f"- Mean X/Y peak-shear asymmetry ratio: {mean_shear_ratio:.3f}",
        f"- Mean X/Y initial-stiffness asymmetry ratio: {mean_stiffness_ratio:.3f}",
        "",
        "## Damage-Class Summary",
        "",
        markdown_table(damage_summary[:8], ["critical_state", "count", "share", "avg_peak_base_shear_kn", "avg_initial_stiffness_kn_per_m", "avg_normalized_energy", "avg_ductility_proxy"]),
        "",
        "## Critical-Element-Type Summary",
        "",
        markdown_table(type_summary[:8], ["critical_type", "count", "share", "avg_peak_base_shear_kn", "avg_initial_stiffness_kn_per_m", "avg_normalized_energy", "avg_ductility_proxy"]),
        "",
        "## Manuscript Note",
        "",
        "These metrics convert individual pushover curves into comparable response indicators. "
        "Because some SAP2000 outputs are stored as envelope-derived monotonic estimates rather than full saved nonlinear state histories, the curve source type should be reported when interpreting energy, ductility, and post-peak behavior.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Return a simple Markdown table."""

    if not rows:
        return "_No data._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = format_float(value)
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def dedupe_directional_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row for each model/direction pair."""

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        deduped[(str(row.get("model_name")), str(row.get("direction")))] = row
    return list(deduped.values())


def average(rows: list[dict[str, Any]], key: str) -> float:
    """Average finite numeric values from dictionaries."""

    return mean([as_float(row.get(key), math.nan) for row in rows])


def share_equal(rows: list[dict[str, Any]], key: str, expected: str) -> float:
    """Return share of rows with key equal to expected."""

    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get(key)) == expected) / len(rows)


def wall_area(candidate: dict[str, Any]) -> float:
    """Approximate total wall area from metadata."""

    if not candidate.get("has_shear_walls"):
        return 0.0
    return as_float(candidate.get("wall_count")) * as_float(candidate.get("wall_thickness_m")) * as_float(candidate.get("wall_length_m"))


def section_label(section: Any) -> str:
    """Return a compact section label."""

    if not isinstance(section, dict):
        return ""
    width = as_float(section.get("width"), math.nan)
    depth = as_float(section.get("depth"), math.nan)
    if not math.isfinite(width) or not math.isfinite(depth):
        return ""
    return f"{int(round(width * 100))}x{int(round(depth * 100))}"


def ratio_larger_to_smaller(left: Any, right: Any) -> float:
    """Return larger/smaller ratio for two positive values."""

    a = abs(as_float(left, math.nan))
    b = abs(as_float(right, math.nan))
    if not math.isfinite(a) or not math.isfinite(b) or min(a, b) <= 1e-12:
        return math.nan
    return max(a, b) / min(a, b)


def safe_divide(left: float, right: float) -> float:
    """Return safe division."""

    if not math.isfinite(left) or not math.isfinite(right) or abs(right) <= 1e-12:
        return 0.0
    return left / right


def as_float(value: object, default: float = 0.0) -> float:
    """Parse a numeric value."""

    try:
        if value in (None, ""):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    """Return the mean of finite values."""

    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def csv_value(value: Any) -> Any:
    """Format CSV values consistently."""

    if isinstance(value, float):
        return format_float(value)
    return value


def format_float(value: float) -> str:
    """Format a float for CSV/Markdown output."""

    if not math.isfinite(value):
        return ""
    return f"{value:.{CSV_FLOAT_PRECISION}g}"


def format_percent(value: float) -> str:
    """Format a share as percentage."""

    if not math.isfinite(value):
        return ""
    return f"{100 * value:.2f}%"


def escape_xml(value: Any) -> str:
    """Escape text for SVG."""

    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
