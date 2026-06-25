"""Local web dashboard for SAP2000 model generator settings and execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import traceback
import csv
from dataclasses import asdict, fields, replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, GeneratorConfig, Section  # noqa: E402
from design_rules import validate_candidate_model  # noqa: E402
from fema440 import calculate_fema440_by_direction  # noqa: E402
from ml_model import ml_status, predict_ml, reset_ml_model, save_ml_snapshot, train_ml_model  # noqa: E402
from model_generator import _model_file_stem, _random_candidate, _refresh_successful_exact_export_models, _rewrite_exact_export_summary, generate_models  # noqa: E402
from sap_hinge_validation import calibrate_proxy, compare_proxy_with_exact, import_hinge_exports, select_validation_subset  # noqa: E402
from som_model import optimize_som_grid, som_status, train_som  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "static"


class JobState:
    """Thread-safe state for the current generator job."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.success_count = 0
        self.failed_count = 0
        self.total = 0
        self.current = 0
        self.cancel_requested = False
        self.rows: list[dict[str, Any]] = []
        self.logs: list[str] = []
        self.error: str = ""
        self.output_dir = str(CONFIG.output_dir)
        self.run_output_dir: str = ""
        self.active_sap_process_ids: set[int] = set()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe copy of the current job state."""
        with self.lock:
            live_sap_process_ids = sorted(pid for pid in self.active_sap_process_ids if _process_is_running(pid))
            if len(live_sap_process_ids) != len(self.active_sap_process_ids):
                self.active_sap_process_ids = set(live_sap_process_ids)
            effective_running = self.running or bool(live_sap_process_ids) or self.cancel_requested
            return {
                "running": self.running,
                "effective_running": effective_running,
                "sap_process_running": bool(live_sap_process_ids),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "total": self.total,
                "current": self.current,
                "cancel_requested": self.cancel_requested,
                "rows": self.rows[-25:],
                "logs": self.logs[-120:],
                "error": self.error,
                "output_dir": self.output_dir,
                "run_output_dir": self.run_output_dir,
                "metadata_csv": str(Path(self.run_output_dir or self.output_dir) / "models_metadata.csv"),
                "active_sap_process_ids": live_sap_process_ids,
            }

    def log(self, message: str) -> None:
        """Append a timestamped log line."""
        with self.lock:
            stamp = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"[{stamp}] {message}")


STATE = JobState()


def section_to_dict(section: Section) -> dict[str, Any]:
    """Convert Section dataclass to dashboard shape."""
    return {"width": section.width, "depth": section.depth, "label": section.label_cm}


def defaults_payload() -> dict[str, Any]:
    """Return dashboard defaults derived from config.py."""
    cfg = CONFIG
    payload = asdict(cfg)
    payload["output_dir"] = str(cfg.output_dir)
    payload["sap2000_exe_path"] = str(cfg.sap2000_exe_path)
    payload["column_sections"] = [section_to_dict(section) for section in cfg.column_sections]
    payload["beam_sections"] = [section_to_dict(section) for section in cfg.beam_sections]
    return payload


def parse_sections(raw_sections: list[dict[str, Any]]) -> tuple[Section, ...]:
    """Parse dashboard section rows into immutable Section objects."""
    sections: list[Section] = []
    for item in raw_sections:
        width = float(item["width"])
        depth = float(item["depth"])
        if width <= 0.0 or depth <= 0.0:
            raise ValueError("Section dimensions must be positive.")
        sections.append(Section(width, depth))
    if not sections:
        raise ValueError("At least one section is required.")
    return tuple(sections)


def config_from_payload(payload: dict[str, Any]) -> GeneratorConfig:
    """Build GeneratorConfig from dashboard JSON payload."""
    allowed = {field.name for field in fields(GeneratorConfig)}
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if key in allowed:
            clean[key] = value

    if "output_dir" in clean:
        clean["output_dir"] = Path(str(clean["output_dir"])).expanduser()
    if "sap2000_exe_path" in clean:
        clean["sap2000_exe_path"] = Path(str(clean["sap2000_exe_path"])).expanduser()
    if "random_seed" in clean and clean["random_seed"] in ("", None):
        clean["random_seed"] = None
    if "column_sections" in clean:
        clean["column_sections"] = parse_sections(clean["column_sections"])
    if "beam_sections" in clean:
        clean["beam_sections"] = parse_sections(clean["beam_sections"])
    if "concrete_classes" in clean:
        clean["concrete_classes"] = tuple(clean["concrete_classes"])
    if "steel_classes" in clean:
        clean["steel_classes"] = tuple(clean["steel_classes"])
    if "pushover_directions" in clean:
        clean["pushover_directions"] = tuple(clean["pushover_directions"])
    if "hinge_relative_distances" in clean:
        clean["hinge_relative_distances"] = tuple(float(value) for value in clean["hinge_relative_distances"])
    if "soil_classes" in clean:
        clean["soil_classes"] = tuple(clean["soil_classes"])
    for ratio_key in ("column_rebar_ratios", "beam_top_ratio_supports", "beam_bottom_ratio_spans", "wall_rebar_ratios", "slab_rebar_ratios", "raft_rebar_ratios"):
        if ratio_key in clean:
            clean[ratio_key] = tuple(float(value) for value in clean[ratio_key])
            if not clean[ratio_key]:
                raise ValueError(f"{ratio_key} must include at least one selected ratio.")
    return replace(CONFIG, **clean)


def preview_candidates(cfg: GeneratorConfig, count: int = 8) -> list[dict[str, Any]]:
    """Generate a small validated candidate preview without touching SAP2000."""
    import random

    rng = random.Random(cfg.random_seed)
    preview: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        candidate = _random_candidate(rng, cfg)
        validation = validate_candidate_model(candidate, cfg)
        preview.append(
            {
                "index": index,
                "name": _model_file_stem(index, candidate),
                "valid": validation.is_valid,
                "story_count": candidate["story_count"],
                "x_bays": candidate["x_bay_count"],
                "y_bays": candidate["y_bay_count"],
                "concrete": candidate["concrete_class"],
                "column": candidate["column_section"].label_cm,
                "beam": candidate["beam_section"].label_cm,
                "rho_col": candidate["rho_col"],
                "rho_beam": candidate["rho_beam"],
                "beam_top_ratio_support": candidate["beam_top_ratio_support"],
                "beam_bottom_ratio_span": candidate["beam_bottom_ratio_span"],
                "push_drift": candidate["pushover_target_drift_ratio"],
                "soil_class": candidate["soil_class"],
                "raft_thickness_m": candidate["raft_thickness_m"],
                "raft_rebar_ratio": candidate["raft_rebar_ratio"],
                "slab_thickness_m": candidate["slab_thickness_m"],
                "slab_rebar_ratio": candidate["slab_rebar_ratio"],
                "subgrade_modulus_kn_m3": candidate["subgrade_modulus_kn_m3"],
                "has_shear_walls": candidate["has_shear_walls"],
                "wall_thickness_m": candidate["wall_thickness_m"],
                "wall_length_m": candidate["wall_length_m"],
                "wall_rebar_ratio": candidate["wall_rebar_ratio"],
                "wall_placement": candidate["wall_placement"],
                "reasons": validation.rejection_reasons,
            }
        )
    return preview


def start_job(cfg: GeneratorConfig) -> bool:
    """Start a background generator job if none is running."""
    base_output_dir = cfg.output_dir
    run_dir = _new_run_output_dir(base_output_dir)
    run_cfg = replace(cfg, output_dir=run_dir)
    with STATE.lock:
        if STATE.running:
            return False
        STATE.running = True
        STATE.started_at = datetime.now().isoformat(timespec="seconds")
        STATE.finished_at = None
        STATE.success_count = 0
        STATE.failed_count = 0
        STATE.total = cfg.n_iter
        STATE.current = 0
        STATE.rows = []
        STATE.logs = []
        STATE.error = ""
        STATE.output_dir = str(base_output_dir)
        STATE.run_output_dir = str(run_dir)
        STATE.cancel_requested = False
        STATE.active_sap_process_ids = set()

    thread = threading.Thread(target=run_job, args=(run_cfg,), daemon=True)
    thread.start()
    return True


def _new_run_output_dir(base_output_dir: Path) -> Path:
    """Return a unique timestamped run directory under the configured output root."""
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_output_dir / stamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base_output_dir / f"{stamp}_{suffix:02d}"
    return candidate


def _process_is_running(process_id: int) -> bool:
    """Return whether a Windows process id is still alive."""
    if process_id <= 0:
        return False
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {process_id}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    return str(process_id) in completed.stdout


def stop_job() -> bool:
    """Request cancellation of the active generation job and stop owned SAP2000 processes."""
    with STATE.lock:
        if not STATE.running:
            return False
        STATE.cancel_requested = True
        active_process_ids = set(STATE.active_sap_process_ids)
    if active_process_ids:
        killed: list[int] = []
        failed: list[int] = []
        for process_id in sorted(active_process_ids):
            completed = subprocess.run(
                ["taskkill", "/PID", str(process_id), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                killed.append(process_id)
            else:
                failed.append(process_id)
        if killed:
            STATE.log(f"Stop requested. Terminated active SAP2000 process(es): {', '.join(str(pid) for pid in killed)}.")
        if failed:
            STATE.log(f"Stop requested, but SAP2000 process(es) could not be terminated immediately: {', '.join(str(pid) for pid in failed)}.")
    else:
        STATE.log("Stop requested. Waiting for the current SAP2000 step to finish because no owned SAP2000 process id is registered.")
    return True


def clear_artifacts(output_dir: Path) -> dict[str, Any]:
    """Delete generated model artifacts from the configured output directory."""
    with STATE.lock:
        if STATE.running:
            raise RuntimeError("Generation is running; artifacts cannot be cleared now.")

    target = output_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    allowed_external_names = {"generated_models", "sap2000_generated_models"}
    is_project_output = target != project_root and project_root in target.parents
    is_named_output_dir = target.name in allowed_external_names and target.parent != target
    if not (is_project_output or is_named_output_dir):
        raise RuntimeError("Refusing to clear a directory that is not a generated model output folder.")

    deleted: list[str] = []
    preserved: list[str] = []
    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():
        if item.name in {"ml_snapshots", "hinge_validation"} or item.name.startswith("ml_"):
            preserved.append(item.name)
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        deleted.append(item.name)

    with STATE.lock:
        STATE.rows = []
        STATE.logs = []
        STATE.success_count = 0
        STATE.failed_count = 0
        STATE.total = 0
        STATE.current = 0
        STATE.error = ""
        STATE.output_dir = str(target)
        STATE.run_output_dir = ""
        STATE.cancel_requested = False
        STATE.finished_at = None
        STATE.started_at = None
    STATE.log(f"Cleared {len(deleted)} artifact(s) from {target}. Preserved {len(preserved)} ML file(s).")
    return {"deleted_count": len(deleted), "deleted": deleted, "preserved_ml": preserved, "output_dir": str(target)}


def artifact_inventory(output_dir: Path, offset: int = 0, limit: int = 50, search: str = "", run_id: str = "") -> dict[str, Any]:
    """Return generated model artifacts, including models without exact export."""
    target = output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 50), 500))
    search_text = str(search or "").strip().lower()
    run_filter = str(run_id or "").strip()
    with STATE.lock:
        running = STATE.running
    for root in _output_scan_roots(target):
        _ensure_exact_metadata_refresh(root)
    files: list[dict[str, Any]] = []
    for item in sorted(_iter_artifact_files(target), key=lambda path: path.stat().st_mtime, reverse=True):
        stat = item.stat()
        rel_name = item.relative_to(target).as_posix()
        files.append(
            {
                "name": rel_name,
                "basename": item.name,
                "model_name": item.name.replace("_preview.svg", "") if item.name.endswith("_preview.svg") else item.stem,
                "run_id": item.parent.name if item.parent != target else "root",
                "path": str(item),
                "kind": artifact_kind(item),
                "is_preview": item.name.endswith("_preview.svg"),
                "is_sap_screenshot": "_sap_" in item.name and item.suffix.lower() == ".png",
                "size_kb": round(stat.st_size / 1024.0, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    previews = [file for file in files if file["is_preview"] and _preview_has_completed_metadata(Path(str(file["path"])))]
    runs = sorted({file["run_id"] for file in previews}, reverse=True)
    if run_filter:
        previews = [file for file in previews if file["run_id"] == run_filter]
    if search_text:
        previews = [
            file
            for file in previews
            if search_text in str(file.get("name", "")).lower()
            or search_text in str(file.get("model_name", "")).lower()
            or search_text in str(file.get("basename", "")).lower()
        ]
    page = previews[offset : offset + limit]
    return {
        "output_dir": str(target),
        "sdb_count": sum(1 for file in files if file["kind"] == "SAP2000"),
        "json_count": sum(1 for file in files if file["kind"] == "JSON"),
        "csv_count": sum(1 for file in files if file["kind"] == "CSV"),
        "preview_count": len(previews),
        "preview_total_count": len(previews),
        "preview_offset": offset,
        "preview_limit": limit,
        "preview_has_more": offset + limit < len(previews),
        "preview_runs": runs,
        "previews": page,
        "files": files[:100],
        "message": (
            "Uretim devam ediyor; tamamlanan model temsili gorselleri listeleniyor."
            if running and page
            else "" if page
            else "Henuz listelenecek model temsili gorseli yok."
        ),
}


def _preview_has_completed_metadata(preview_path: Path) -> bool:
    """Return true when a preview belongs to a completed metadata record."""
    if not preview_path.name.endswith("_preview.svg"):
        return False
    metadata_path = preview_path.with_name(preview_path.name.replace("_preview.svg", "_metadata.json"))
    if not metadata_path.exists():
        return False
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status = str(data.get("status") or data.get("uretim_durumu") or "").lower()
    if status not in {"success", "successful"}:
        return False
    candidate = data.get("candidate", {}) if isinstance(data.get("candidate"), dict) else {}
    plastic = data.get("plastic_hinges", {}) if isinstance(data.get("plastic_hinges"), dict) else {}
    results = plastic.get("results", {}) if isinstance(plastic.get("results"), dict) else {}
    summaries = results.get("summary_by_direction", {}) if isinstance(results, dict) else {}
    return bool(candidate and summaries)


def model_results_inventory(output_dir: Path) -> dict[str, Any]:
    """Return per-model pushover summaries, exact export status included."""
    target = output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with STATE.lock:
        running = STATE.running
    for root in _output_scan_roots(target):
        _ensure_exact_metadata_refresh(root)
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(_iter_metadata_files(target), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            data = json.loads(item.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - skip corrupt/incomplete metadata but report why.
            models.append({"name": item.stem, "metadata_file": item.relative_to(target).as_posix(), "available": False, "message": str(exc)})
            continue
        model_path = Path(str(data.get("model_path", "")))
        name = model_path.stem or item.stem.replace("_metadata", "")
        pushover = data.get("pushover", {}) if isinstance(data.get("pushover"), dict) else {}
        plastic = data.get("plastic_hinges", {}) if isinstance(data.get("plastic_hinges"), dict) else {}
        results = pushover.get("results", {}) if isinstance(pushover.get("results"), dict) else {}
        curves = results.get("curves", {}) if isinstance(results.get("curves"), dict) else {}
        story_drifts = results.get("story_drifts", {}) if isinstance(results.get("story_drifts"), dict) else {}
        if not story_drifts and isinstance(data.get("candidate"), dict):
            story_drifts = data["candidate"].get("story_drift_results", {}) if isinstance(data["candidate"].get("story_drift_results"), dict) else {}
        compact_curves: dict[str, Any] = {}
        for direction, curve in curves.items():
            if not isinstance(curve, dict):
                continue
            points = curve.get("points", [])
            compact_curves[direction] = {
                "available": bool(curve.get("available")),
                "point_count": curve.get("point_count", len(points) if isinstance(points, list) else 0),
                "peak_base_shear_kn": curve.get("peak_base_shear_kn"),
                "final_control_displacement_m": curve.get("final_control_displacement_m"),
                "final_base_shear_kn": curve.get("final_base_shear_kn"),
                "points": points if isinstance(points, list) else [],
                "message": friendly_result_message(curve.get("message", "")),
            }
        candidate = data.get("candidate", {}) if isinstance(data.get("candidate"), dict) else {}
        assignment = plastic.get("assignment", {}) if isinstance(plastic.get("assignment"), dict) else {}
        hinge_results = plastic.get("results", {}) if isinstance(plastic.get("results"), dict) else {}
        exact_export = plastic.get("exact_ui_export", {}) if isinstance(plastic.get("exact_ui_export"), dict) else {}
        exact_available = hinge_results.get("result_source") == "sap_exact_frame_hinge_states" and bool(hinge_results.get("available"))
        hinge_summary = hinge_results.get("summary_by_direction", {}) if isinstance(hinge_results.get("summary_by_direction"), dict) else {}
        if exact_available and isinstance(hinge_results.get("exact_events_by_direction"), dict):
            hinge_events = hinge_results.get("exact_events_by_direction", {})
        elif isinstance(hinge_results.get("proxy_events_by_direction"), dict):
            hinge_events = hinge_results.get("proxy_events_by_direction", {})
        else:
            hinge_events = hinge_results.get("events_by_direction", {}) if isinstance(hinge_results.get("events_by_direction"), dict) else {}
        result_source = hinge_results.get("result_source", "") if isinstance(hinge_results, dict) else ""
        if not result_source and hinge_summary:
            result_source = "proxy"
        hinge_samples = {
            direction: _critical_hinge_events(value.get("events", []), 8)
            for direction, value in hinge_events.items()
            if isinstance(value, dict) and isinstance(value.get("events"), list)
        }
        seen.add(name)
        model_record = {
                "name": name,
                "metadata_file": item.relative_to(target).as_posix(),
                "run_id": item.parent.name if item.parent != target else "root",
                "model_file": model_path.name if model_path.name else "",
                "status": data.get("status", ""),
                "story_count": candidate.get("story_count"),
                "x_bay_count": candidate.get("x_bay_count"),
                "y_bay_count": candidate.get("y_bay_count"),
                "spans_x": candidate.get("spans_x", []),
                "spans_y": candidate.get("spans_y", []),
                "story_height": candidate.get("story_height"),
                "raft_thickness_m": candidate.get("raft_thickness_m"),
                "raft_rebar_ratio": candidate.get("raft_rebar_ratio"),
                "slab_thickness_m": candidate.get("slab_thickness_m"),
                "slab_rebar_ratio": candidate.get("slab_rebar_ratio"),
                "target_drift_ratio": pushover.get("target_drift_ratio"),
                "target_displacement_m": pushover.get("target_displacement_m"),
                "analysis_run": pushover.get("analysis_run"),
                "analysis_status": pushover.get("analysis_status", {}),
                "hinge_assigned_count": assignment.get("assigned_count"),
                "hinge_expected_count": assignment.get("expected_hinges"),
                "hinge_result_available": hinge_results.get("available") if isinstance(hinge_results, dict) else None,
                "hinge_result_note": hinge_results.get("note", "") if isinstance(hinge_results, dict) else "",
                "hinge_result_source": result_source,
                "exact_export_available": exact_available,
                "exact_export_message": exact_export.get("message", "") or hinge_results.get("exact_export_message", ""),
                "hinge_summary": hinge_summary,
                "hinge_event_counts": {
                    direction: value.get("event_count", 0)
                    for direction, value in hinge_events.items()
                    if isinstance(value, dict)
                },
                "hinge_event_samples": hinge_samples,
                "curves": compact_curves,
                "story_drifts": story_drifts,
        }
        model_record["fema440"] = calculate_fema440_by_direction(model_record)
        models.append(model_record)
    return {
        "output_dir": str(target),
        "models": models[:500],
        "message": (
            "Uretim devam ediyor; tamamlanan model sonuclari listeleniyor."
            if running and models
            else "" if models
            else "Henuz listelenecek model sonucu yok."
        ),
    }


def _output_scan_roots(output_dir: Path) -> list[Path]:
    """Return output root and timestamped run folders that may contain models."""
    roots = [output_dir]
    if output_dir.exists():
        roots.extend(path for path in sorted(output_dir.glob("run_*")) if path.is_dir())
    return roots


def _iter_artifact_files(output_dir: Path) -> list[Path]:
    """Return generated artifact files from root and all run folders."""
    allowed_suffixes = {".sdb", ".json", ".csv", ".svg", ".png", ".out", ".msh", ".log", ".$2k", ".ico"}
    files: list[Path] = []
    for root in _output_scan_roots(output_dir):
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_file() and (item.suffix.lower() in allowed_suffixes or item.name == "models_metadata.csv"):
                files.append(item)
    return files


def _iter_metadata_files(output_dir: Path) -> list[Path]:
    """Return per-model metadata files from root and all run folders."""
    files: list[Path] = []
    for root in _output_scan_roots(output_dir):
        if root.exists():
            files.extend(root.glob("*_metadata.json"))
    return files


def _verified_exact_model_stems(output_dir: Path) -> set[str]:
    """Return models whose X/Y exact SAP hinge exports passed worker validation."""
    summary_path = output_dir / "hinge_validation" / "export_worker_summary.json"
    if not summary_path.exists():
        return set()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    min_rows = int(summary.get("min_rows_per_direction") or CONFIG.exact_hinge_export_min_rows_per_direction)
    verified: set[str] = set()
    for record in summary.get("records", []):
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        validation = record.get("export_validation", {})
        stem = str(record.get("model_stem") or "")
        if stem and isinstance(validation, dict) and _exact_validation_has_minimum_rows(validation, min_rows):
            verified.add(stem)
    return verified


def _exact_validation_has_minimum_rows(validation: dict[str, Any], min_rows: int) -> bool:
    """Check exact SAP hinge validation with the current dashboard row threshold."""
    if not validation.get("is_valid"):
        return False
    files = validation.get("files", [])
    if not isinstance(files, list) or len(files) < 2:
        return False
    return all(
        isinstance(file, dict)
        and str(file.get("table_name", "")).lower() == "table:  frame hinge states"
        and bool(file.get("has_required_columns"))
        and int(file.get("row_count") or 0) >= max(1, min_rows)
        for file in files
    )


def _ensure_exact_metadata_refresh(output_dir: Path) -> None:
    """Upgrade successful older worker summaries to exact metadata and SVG once."""
    summary_path = output_dir / "hinge_validation" / "export_worker_summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    min_rows = int(summary.get("min_rows_per_direction") or CONFIG.exact_hinge_export_min_rows_per_direction)
    pending = []
    for record in summary.get("records", []):
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        validation = record.get("export_validation", {})
        if not isinstance(validation, dict) or not _exact_validation_has_minimum_rows(validation, min_rows):
            continue
        stem = str(record.get("model_stem") or "")
        metadata_path = output_dir / f"{stem}_metadata.json"
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = data.get("plastic_hinges", {}).get("results", {})
        if not isinstance(results, dict) or results.get("result_source") != "sap_exact_frame_hinge_states":
            pending.append(stem)
    if not pending:
        return
    _refresh_successful_exact_export_models(output_dir, summary)
    _rewrite_exact_export_summary(output_dir, summary)


def _csv_float(value: object) -> float | None:
    """Parse dashboard CSV numeric values."""
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _csv_json_list(value: object) -> list[float]:
    """Parse JSON list values stored in the metadata CSV."""
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    clean: list[float] = []
    for item in parsed:
        try:
            clean.append(float(item))
        except (TypeError, ValueError):
            continue
    return clean


def _critical_hinge_events(events: list[Any], limit: int) -> list[Any]:
    """Return the most critical hinge events for dashboard preview."""
    if not isinstance(events, list):
        return []
    rows = [
        event
        for event in events
        if isinstance(event, dict) and _hinge_state_rank(event.get("hinge_state_level")) >= _hinge_state_rank("B-IO")
    ]
    return sorted(
        rows,
        key=lambda event: (
            _csv_float(event.get("plastic_rotation_rad")) or 0.0,
            _csv_float(event.get("demand_capacity_ratio")) or 0.0,
            _csv_float(event.get("step_number")) or 0.0,
        ),
        reverse=True,
    )[:limit]


def _hinge_state_rank(level: object) -> int:
    """Return dashboard ordering rank for one hinge-state label."""
    levels = {"A-B": 0, "B-IO": 1, "IO-LS": 2, "LS-CP": 3, "CP-C": 4, "C-D": 5, "D-E": 6, "beyond E": 7}
    return levels.get(str(level or ""), -1)


def _csv_int(value: object) -> int | None:
    """Parse dashboard CSV integer values."""
    parsed = _csv_float(value)
    return int(parsed) if parsed is not None else None


def friendly_result_message(message: object) -> str:
    """Return a dashboard-friendly message for known historical result read errors."""
    text = str(message or "")
    if "must be real number, not list" in text:
        return "Eski sonuc okuma hatasi. Kod duzeltildi; modeli yeniden analiz edince kapasite egrisi okunacak."
    return text


def artifact_kind(path: Path) -> str:
    """Classify generated artifact by extension/name."""
    if path.suffix.lower() == ".sdb":
        return "SAP2000"
    if path.suffix.lower() == ".json":
        return "JSON"
    if path.suffix.lower() == ".csv":
        return "CSV"
    if path.suffix.lower() == ".svg":
        return "Preview"
    if path.suffix.lower() == ".png" and "_sap_" in path.name:
        return "SAP Screenshot"
    if path.suffix.lower() == ".png":
        return "PNG"
    return path.suffix.lower().lstrip(".").upper() or "FILE"


def open_sap_model(output_dir: Path, sap_exe: Path, name: str) -> dict[str, Any]:
    """Open a generated SAP2000 .sdb model with the Windows file association."""
    if not name:
        raise RuntimeError("Invalid SAP2000 model file name.")
    target = resolve_sap_model_path(output_dir, name)
    output_root = output_dir.resolve()
    try:
        target.relative_to(output_root)
    except ValueError as exc:
        raise RuntimeError("Refusing to open a file outside the output directory.")
    if not target.exists():
        raise RuntimeError(f"Model file not found: {target}")
    if not sap_exe.exists():
        raise RuntimeError(f"SAP2000 executable not found: {sap_exe}")
    subprocess.Popen([str(sap_exe), str(target)], close_fds=True)
    return {"opened": True, "path": str(target), "sap2000_exe": str(sap_exe)}


def resolve_sap_model_path(output_dir: Path, requested_name: str) -> Path:
    """Resolve a preview or model file name to an existing .sdb file."""
    output_root = output_dir.resolve()
    raw = Path(str(requested_name).replace("\\", "/"))
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise RuntimeError("Invalid SAP2000 model file name.")
    name = raw.name
    requested_parent = output_root / raw.parent
    candidates: list[Path] = []
    lower_name = name.lower()
    if lower_name.endswith("_preview.svg"):
        candidates.append(requested_parent / name.replace("_preview.svg", ".sdb"))
    elif lower_name.endswith(".sdb"):
        candidates.append(requested_parent / name)
    else:
        candidates.append(requested_parent / f"{name}.sdb")

    stem = name
    for suffix in ("_preview.svg", ".sdb", ".svg"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates.extend(requested_parent.glob(f"{stem}*.sdb"))
    candidates.extend(output_root.glob(f"run_*/{stem}*.sdb"))
    candidates.extend(output_root.glob(f"{stem}*.sdb"))

    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(output_root)
        except ValueError:
            continue
        if resolved.exists():
            return resolved
    raise RuntimeError(f"SAP2000 model file not found for: {requested_name}")


def run_job(cfg: GeneratorConfig) -> None:
    """Run model generation and update shared state."""
    temp_dir = cfg.output_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    old_temp = os.environ.get("TEMP")
    old_tmp = os.environ.get("TMP")
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)

    def should_stop() -> bool:
        with STATE.lock:
            return STATE.cancel_requested

    def progress(event: str, payload: dict[str, Any]) -> None:
        if event == "iteration_started":
            with STATE.lock:
                STATE.current = int(payload["index"])
            STATE.log(f"Iteration {payload['index']}/{payload['total']} started.")
        elif event == "iteration_finished":
            STATE.log(f"Iteration {payload['index']} finished: {payload['status']} {payload.get('model_path', '')}")
        elif event == "sap_started":
            process_ids = {int(pid) for pid in payload.get("sap_process_ids", []) if str(pid).isdigit()}
            with STATE.lock:
                STATE.active_sap_process_ids = process_ids
            suffix = f" PID: {', '.join(str(pid) for pid in sorted(process_ids))}" if process_ids else ""
            STATE.log(f"{payload.get('message', event)}{suffix}")
        elif event == "sap_closed":
            with STATE.lock:
                STATE.active_sap_process_ids = set()
            STATE.log(str(payload.get("message", event)))
        elif event == "sap_session_error":
            with STATE.lock:
                STATE.active_sap_process_ids = set()
            STATE.log(str(payload.get("message", event)))
        elif event == "exact_hinge_export_attempt_started":
            STATE.log(
                "Exact SAP hinge export "
                f"{payload['index']}/{payload['total']}, attempt {payload['attempt']}/{payload['max_attempts']}: "
                f"{Path(str(payload['model_path'])).stem}"
            )
        elif event == "exact_hinge_export_retrying":
            STATE.log(
                "Exact SAP hinge export will retry in a fresh SAP session "
                f"({payload['next_attempt']}/{payload['max_attempts']}): {payload.get('message', '')}"
            )
        elif event == "stopped":
            STATE.log(str(payload.get("message", event)))
        else:
            STATE.log(str(payload.get("message", event)))

    try:
        rows = generate_models(cfg, progress_callback=progress, stop_callback=should_stop)
        with STATE.lock:
            STATE.rows = [asdict(row) for row in rows]
            STATE.success_count = sum(row.uretim_durumu == "success" for row in rows)
            STATE.failed_count = len(rows) - STATE.success_count
            STATE.current = len(rows)
            if STATE.cancel_requested:
                STATE.error = ""
        if cfg.ml_auto_train_after_generation and any(row.uretim_durumu == "success" for row in rows):
            algorithm = cfg.ml_auto_train_algorithm or "random_forest"
            STATE.log(f"ML auto-train started with {algorithm}; previous weights are kept until the new model is written.")
            result = train_ml_model(cfg.output_dir, algorithm, cfg.ml_random_seed, cfg.ml_preserve_existing_weights, cfg.ml_use_som_features)
            STATE.log(f"ML auto-train finished: {result.get('model_path', '')}")
    except Exception as exc:  # noqa: BLE001 - dashboard should surface any generator failure.
        with STATE.lock:
            STATE.error = f"{exc}\n{traceback.format_exc()}"
            STATE.failed_count = max(STATE.failed_count, cfg.n_iter - len(STATE.rows))
        STATE.log(f"Generation failed: {exc}")
    finally:
        if old_temp is not None:
            os.environ["TEMP"] = old_temp
        if old_tmp is not None:
            os.environ["TMP"] = old_tmp
        with STATE.lock:
            STATE.running = False
            STATE.finished_at = datetime.now().isoformat(timespec="seconds")
            was_cancelled = STATE.cancel_requested
            STATE.cancel_requested = False
            STATE.active_sap_process_ids = set()
        if was_cancelled:
            STATE.log("Generation stopped by user.")


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP API and static file handler for the local dashboard."""

    def do_GET(self) -> None:
        """Serve static files and read-only API endpoints."""
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/defaults":
            self._send_json(defaults_payload())
        elif path == "/api/status":
            self._send_json(STATE.snapshot())
        elif path == "/api/artifacts":
            query = parse_qs(parsed.query)
            self._send_json(
                artifact_inventory(
                    Path(STATE.output_dir),
                    int(query.get("offset", ["0"])[0] or 0),
                    int(query.get("limit", ["50"])[0] or 50),
                    query.get("search", [""])[0],
                    query.get("run", [""])[0],
                )
            )
        elif path == "/api/model-results":
            self._send_json(model_results_inventory(Path(STATE.output_dir)))
        elif path == "/api/ml/status":
            self._send_json(ml_status(Path(STATE.output_dir)))
        elif path == "/api/som/status":
            self._send_json(som_status(Path(STATE.output_dir)))
        elif path == "/api/artifact":
            query = parse_qs(parsed.query)
            self._serve_artifact(query.get("name", [""])[0])
        elif path.startswith("/static/"):
            self._serve_static(path.removeprefix("/static/"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        """Handle preview and generation requests."""
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            cfg = config_from_payload(payload.get("config", payload))
            if path == "/api/preview":
                self._send_json({"candidates": preview_candidates(cfg, cfg.n_iter)})
            elif path == "/api/run":
                started = start_job(cfg)
                status = HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
                if not started:
                    STATE.log("Run request ignored because a generation job is already running.")
                self._send_json({"started": started, "status": STATE.snapshot()}, status=status)
            elif path == "/api/stop":
                stopped = stop_job()
                status = HTTPStatus.ACCEPTED if stopped else HTTPStatus.CONFLICT
                self._send_json({"stop_requested": stopped, "status": STATE.snapshot()}, status=status)
            elif path == "/api/clear-artifacts":
                result = clear_artifacts(cfg.output_dir)
                self._send_json({"cleared": True, "result": result, "status": STATE.snapshot()})
            elif path == "/api/open-model":
                result = open_sap_model(cfg.output_dir, cfg.sap2000_exe_path, str(payload.get("name", "")))
                self._send_json({"opened": True, "result": result})
            elif path == "/api/ml/train":
                preserve_existing = bool(payload.get("preserve_existing", cfg.ml_preserve_existing_weights))
                train_result = train_ml_model(
                    cfg.output_dir,
                    str(payload.get("algorithm", "knn")),
                    int(payload.get("random_seed", 42)),
                    preserve_existing,
                    bool(payload.get("use_som_features", cfg.ml_use_som_features)),
                    payload.get("selected_x_columns") if isinstance(payload.get("selected_x_columns"), list) else None,
                )
                self._send_json({"trained": True, "result": ml_status(cfg.output_dir), "train_result": train_result})
            elif path == "/api/ml/predict":
                result = predict_ml(cfg.output_dir, payload.get("params", {}), str(payload.get("algorithm", "") or ""))
                self._send_json({"predicted": True, "result": result})
            elif path == "/api/ml/save-snapshot":
                result = save_ml_snapshot(cfg.output_dir, str(payload.get("algorithm", "") or ""))
                self._send_json({"saved": True, "result": result})
            elif path == "/api/ml/reset":
                result = reset_ml_model(cfg.output_dir, str(payload.get("algorithm", "") or ""))
                self._send_json({"reset": True, "result": result})
            elif path == "/api/som/train":
                result = train_som(
                    cfg.output_dir,
                    int(payload.get("width", 8) or 8),
                    int(payload.get("height", 8) or 8),
                    int(payload.get("iterations", 800) or 800),
                    str(payload.get("result_metric", "max_story_drift_ratio") or "max_story_drift_ratio"),
                    int(payload.get("random_seed", cfg.ml_random_seed) or cfg.ml_random_seed),
                    payload.get("selected_x_columns") if isinstance(payload.get("selected_x_columns"), list) else None,
                )
                self._send_json({"trained": True, "result": result})
            elif path == "/api/som/optimize":
                result = optimize_som_grid(
                    cfg.output_dir,
                    int(payload.get("start_size", 3) or 3),
                    int(payload.get("max_size", 8) or 8),
                    int(payload.get("iterations", 800) or 800),
                    str(payload.get("result_metric", "max_story_drift_ratio") or "max_story_drift_ratio"),
                    int(payload.get("random_seed", cfg.ml_random_seed) or cfg.ml_random_seed),
                    payload.get("selected_x_columns") if isinstance(payload.get("selected_x_columns"), list) else None,
                )
                self._send_json({"optimized": True, "result": result})
            elif path == "/api/hinge-validation/select":
                result = select_validation_subset(cfg.output_dir, int(payload.get("count", 30)))
                self._send_json({"selected": True, "result": result})
            elif path == "/api/hinge-validation/calibrate":
                imported = import_hinge_exports(cfg.output_dir)
                report = compare_proxy_with_exact(cfg.output_dir)
                calibration = calibrate_proxy(cfg.output_dir)
                self._send_json({"calibrated": True, "result": {"import": imported, "report": report, "calibration": calibration}})
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - API should return useful validation messages.
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep terminal output compact."""
        return

    def _read_json(self) -> dict[str, Any]:
        """Read request JSON body."""
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send JSON response."""
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative_path: str) -> None:
        """Serve a file from the dashboard static directory."""
        target = (STATIC_DIR / relative_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        content_type = "text/plain; charset=utf-8"
        if target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        self._serve_file(target, content_type)

    def _serve_artifact(self, name: str) -> None:
        """Serve one generated artifact by file name from the output directory."""
        if not name:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid artifact name")
            return
        requested = Path(str(name).replace("\\", "/"))
        if requested.is_absolute() or any(part == ".." for part in requested.parts):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid artifact name")
            return
        target = (Path(STATE.output_dir) / requested).resolve()
        output_dir = Path(STATE.output_dir).resolve()
        try:
            target.relative_to(output_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        content_type = "application/octet-stream"
        if target.suffix.lower() == ".svg":
            content_type = "image/svg+xml; charset=utf-8"
        elif target.suffix.lower() == ".png":
            content_type = "image/png"
        elif target.suffix.lower() == ".json":
            content_type = "application/json; charset=utf-8"
        elif target.suffix.lower() == ".csv":
            content_type = "text/csv; charset=utf-8"
        self._serve_file(target, content_type)

    def _serve_file(self, path: Path, content_type: str) -> None:
        """Serve a local file."""
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Start the local dashboard server."""
    server = ThreadingHTTPServer(("127.0.0.1", 8765), DashboardHandler)
    print("SAP2000 generator dashboard: http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
