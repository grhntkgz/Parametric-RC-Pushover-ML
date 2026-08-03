"""Random candidate generation, validation, SAP2000 model creation, and metadata."""

from __future__ import annotations

import csv
import json
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from config import CONFIG, GeneratorConfig, Section
from design_rules import CONCRETE_FCK_MPA, ValidationResult, concrete_elastic_modulus_mpa, steel_yield_strength_mpa, validate_candidate_model
from sap_api import Point3D, Sap2000Api, SapApiError
from sap_hinge_export_worker import export_models_in_isolated_workers
from sap_hinge_ui_export import PUSHOVER_OUTPUT_CASES, _windows_process_ids
from sap_hinge_validation import normalize_export_file

MAX_STORED_HINGE_EVENTS_PER_DIRECTION = 40
MAX_STORED_CURVE_POINTS_PER_DIRECTION = 80
PROXY_CALIBRATION_RELATIVE_PATH = Path("hinge_validation") / "proxy_calibration.json"


@dataclass
class ModelMetadata:
    """Metadata row saved to CSV for each successful or failed iteration."""

    model_id: str
    kat_sayisi: int | None
    x_yonu_aciklik_sayisi: int | None
    y_yonu_aciklik_sayisi: int | None
    aciklik_uzunluklari_x: str
    aciklik_uzunluklari_y: str
    kat_yuksekligi: float | None
    toplam_yukseklik: float | None
    temel_tipi: str
    perde_var: bool
    perde_sayisi: int | None
    perde_kalinligi_m: float | None
    perde_uzunlugu_m: float | None
    perde_donati_orani: float | None
    perde_yerlesimi: str
    x_yonu_perde_sayisi: int | None
    y_yonu_perde_sayisi: int | None
    perde_lw_tw_kontrolu: bool | None
    perde_min_kalinlik_kontrolu: bool | None
    radye_kalinligi_m: float | None
    radye_donati_orani: float | None
    radye_donati_orani_kontrolu: bool | None
    doseme_kalinligi_m: float | None
    doseme_donati_orani: float | None
    doseme_kalinligi_kontrolu: bool | None
    doseme_donati_orani_kontrolu: bool | None
    doseme_duzlem_ici_gerilme_kontrolu_placeholder: str
    zemin_sinifi: str
    zemin_yatak_katsayisi_kn_m3: float | None
    zemin_yayi_modeli: str
    beton_sinifi: str
    beton_fck_mpa: float | None
    beton_elastisite_modulu_mpa: float | None
    celik_sinifi: str
    kolon_kesitleri: str
    kiris_kesitleri: str
    kolon_donati_orani: float | None
    kiris_donati_orani: float | None
    kiris_mesnet_ust_donati_orani: float | None
    kiris_aciklik_alt_donati_orani: float | None
    kolon_min_donati_kontrolu: bool | None
    kolon_max_donati_kontrolu: bool | None
    kiris_min_donati_kontrolu: bool | None
    kiris_max_donati_kontrolu: bool | None
    guclu_kolon_zayif_kiris_kontrolu: bool | None
    goreli_kat_otelemesi_kontrolu_placeholder: str
    pushover_aktif: bool
    pushover_yonleri: str
    pushover_hedef_otelemesi_orani: float | None
    pushover_hedef_deplasman_x_m: float | None
    pushover_hedef_deplasman_y_m: float | None
    pushover_yuk_dagilimi: str
    pushover_analiz_calistirildi: bool
    plastik_mafsal_aktif: bool
    kolon_mafsal_tipi: str
    kiris_mafsal_tipi: str
    kesme_mafsali_aktif: bool
    plastik_mafsal_atanan_eleman_sayisi: int | None
    plastik_mafsal_atama_uyarilari: str
    mafsal_sonuc_okuma_durumu: str
    model_kayit_yolu: str
    uretim_durumu: str
    varsa_elenme_nedeni: str


ProgressCallback = Callable[[str, dict[str, Any]], None]
StopCallback = Callable[[], bool]


def generate_models(
    cfg: GeneratorConfig = CONFIG,
    progress_callback: ProgressCallback | None = None,
    stop_callback: StopCallback | None = None,
) -> list[ModelMetadata]:
    """Generate requested number of valid SAP2000 models and metadata files."""
    rng = random.Random(cfg.random_seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _invalidate_exact_export_summary(cfg.output_dir)
    metadata_path = cfg.output_dir / "models_metadata.csv"
    rows: list[ModelMetadata] = []
    inline_exact_export = _inline_exact_export_enabled(cfg)

    if inline_exact_export:
        for index in range(1, cfg.n_iter + 1):
            if stop_callback is not None and stop_callback():
                _emit_progress(progress_callback, "stopped", {"message": "Stop requested; no new iterations will be started."})
                break
            _emit_progress(progress_callback, "iteration_started", {"index": index, "total": cfg.n_iter})
            sap, owned_sap_processes = _start_sap_session(progress_callback)
            try:
                try:
                    row = _generate_one_model(index, rng, cfg, sap, owned_sap_processes)
                except Exception as exc:  # noqa: BLE001 - isolate SAP/COM crashes to one iteration.
                    _emit_progress(progress_callback, "sap_session_error", {"message": f"SAP2000 session error on iteration {index}: {exc}"})
                    row = _metadata_from_candidate(index, None, None, Path(""), "failed", f"SAP2000 session error: {exc}", cfg)
            finally:
                _close_sap_session(sap, owned_sap_processes, force_owned_processes=False)
                _emit_progress(progress_callback, "sap_closed", {"message": "SAP2000 COM session closed."})
            rows.append(row)
            _write_metadata_csv(metadata_path, rows)
            _emit_progress(
                progress_callback,
                "iteration_finished",
                {
                    "index": index,
                    "total": cfg.n_iter,
                    "status": row.uretim_durumu,
                    "model_path": row.model_kayit_yolu,
                    "reason": row.varsa_elenme_nedeni,
                },
            )
    else:
        sap, owned_sap_processes = _start_sap_session(progress_callback)
        try:
            for index in range(1, cfg.n_iter + 1):
                if stop_callback is not None and stop_callback():
                    _emit_progress(progress_callback, "stopped", {"message": "Stop requested; no new iterations will be started."})
                    break
                _emit_progress(progress_callback, "iteration_started", {"index": index, "total": cfg.n_iter})
                try:
                    row = _generate_one_model(index, rng, cfg, sap, owned_sap_processes)
                except Exception as exc:  # noqa: BLE001 - SAP2000 COM can drop its IPC pipe; restart for the next iteration.
                    _emit_progress(progress_callback, "sap_session_error", {"message": f"SAP2000 session error on iteration {index}; restarting SAP2000: {exc}"})
                    row = _metadata_from_candidate(index, None, None, Path(""), "failed", f"SAP2000 session error: {exc}", cfg)
                    _close_sap_session(sap, owned_sap_processes, force_owned_processes=True)
                    if index < cfg.n_iter and (stop_callback is None or not stop_callback()):
                        sap, owned_sap_processes = _start_sap_session(progress_callback)
                rows.append(row)
                _write_metadata_csv(metadata_path, rows)
                _emit_progress(
                    progress_callback,
                    "iteration_finished",
                    {
                        "index": index,
                        "total": cfg.n_iter,
                        "status": row.uretim_durumu,
                        "model_path": row.model_kayit_yolu,
                        "reason": row.varsa_elenme_nedeni,
                    },
                )
        finally:
            if not cfg.keep_sap_open:
                _close_sap_session(sap, owned_sap_processes, force_owned_processes=False)
                _emit_progress(progress_callback, "sap_closed", {"message": "SAP2000 COM session closed."})

    if not inline_exact_export and cfg.enable_exact_hinge_ui_export and cfg.enable_plastic_hinges and not cfg.keep_sap_open:
        if stop_callback is not None and stop_callback():
            _emit_progress(progress_callback, "stopped", {"message": "Stop requested; exact SAP hinge export batch will not be started."})
            return rows
        model_paths = [
            Path(row.model_kayit_yolu)
            for row in rows
            if row.uretim_durumu == "success" and row.model_kayit_yolu and Path(row.model_kayit_yolu).exists()
        ]
        _emit_progress(
            progress_callback,
            "exact_hinge_export_batch_started",
            {"total": len(model_paths), "message": "Starting isolated SAP2000 exact-hinge reanalysis and export workers."},
        )
        summary = export_models_in_isolated_workers(
            cfg.output_dir,
            model_paths,
            max_attempts_per_model=max(1, cfg.exact_hinge_export_max_attempts_per_model),
            min_rows_per_direction=max(1, cfg.exact_hinge_export_min_rows_per_direction),
            progress_callback=progress_callback,
            stop_callback=stop_callback,
        )
        if summary.get("cancelled"):
            _emit_progress(progress_callback, "stopped", {"message": "Stop requested; active exact SAP hinge export worker was terminated."})
            return rows
        _refresh_successful_exact_export_models(cfg.output_dir, summary)
        _mark_failed_exact_export_models(cfg.output_dir, summary)
        _write_metadata_csv(metadata_path, rows)
        _emit_progress(progress_callback, "exact_hinge_export_batch_finished", summary)

    return rows


def _inline_exact_export_enabled(cfg: GeneratorConfig) -> bool:
    """Return whether exact hinge export should run inside each first-solve SAP session."""
    return bool(
        cfg.enable_inline_exact_hinge_export
        and cfg.enable_exact_hinge_ui_export
        and cfg.enable_plastic_hinges
        and cfg.run_analysis_after_save
        and not cfg.keep_sap_open
    )


def _invalidate_exact_export_summary(output_dir: Path) -> None:
    """Remove the previous batch marker so unfinished models stay hidden."""
    summary_path = output_dir / "hinge_validation" / "export_worker_summary.json"
    if summary_path.exists():
        summary_path.unlink()


def _refresh_successful_exact_export_models(output_dir: Path, summary: dict[str, Any]) -> None:
    """Replace proxy dashboard summaries and SVG highlights with exact SAP rows."""
    refreshed: list[str] = []
    for record in summary.get("records", []):
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        stem = str(record.get("model_stem") or "")
        metadata_path = output_dir / f"{stem}_metadata.json"
        if not stem or not metadata_path.exists():
            continue
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = []
        for exported_file in record.get("exported_files", []):
            path = Path(str(exported_file))
            if path.exists():
                rows.extend(normalize_export_file(path))
        exact_results = _exact_hinge_results(rows, record)
        if not exact_results.get("available"):
            continue
        plastic_hinges = data.setdefault("plastic_hinges", {})
        plastic_hinges["results"] = exact_results
        plastic_hinges["exact_ui_export"] = {
            "available": True,
            "source": "sap_exact_frame_hinge_states",
            "exported_files": record.get("exported_files", []),
        }
        metadata_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate = dict(data.get("candidate", {}))
        candidate["column_section"] = Section(**candidate["column_section"])
        candidate["beam_section"] = Section(**candidate["beam_section"])
        candidate["plastic_hinge_results"] = exact_results
        _write_model_preview_svg(output_dir / f"{stem}_preview.svg", candidate)
        record["metadata_refreshed_from_exact_hinges"] = True
        refreshed.append(stem)
    summary["exact_metadata_refreshed_models"] = refreshed


def _exact_hinge_results(rows: list[Any], record: dict[str, Any]) -> dict[str, Any]:
    """Build dashboard-ready exact hinge summaries from normalized SAP rows."""
    events_by_direction: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for direction in ("X", "Y"):
        events = [
            {
                "step_number": row.step_number,
                "load_step": row.load_step,
                "element_name": row.element_name,
                "element_type": row.element_type,
                "hinge_name": row.hinge_name,
                "hinge_location": row.hinge_location,
                "relative_distance": row.relative_distance,
                "hinge_state_level": row.hinge_state_level,
                "plastic_rotation_rad": row.plastic_rotation_rad,
                "moment_kn_m": row.moment_kn_m,
                "axial_force_kn": row.axial_force_kn,
                "shear_v2_kn": row.shear_v2_kn,
                "shear_v3_kn": row.shear_v3_kn,
                "result_source": "sap_exact_frame_hinge_states",
            }
            for row in rows
            if row.direction == direction
        ]
        if not events:
            continue
        summaries[direction] = _summarize_exact_hinge_events(events)
        events_by_direction[direction] = {
            "case": f"PUSHOVER_{direction}",
            "available": True,
            "event_count": len(events),
            "stored_event_count": min(len(events), MAX_STORED_HINGE_EVENTS_PER_DIRECTION),
            "events": _ranked_exact_hinge_events(events, MAX_STORED_HINGE_EVENTS_PER_DIRECTION),
        }
    return {
        "available": set(events_by_direction) == {"X", "Y"},
        "result_source": "sap_exact_frame_hinge_states",
        "exact_export_validation": record.get("export_validation", {}),
        "exact_events_by_direction": events_by_direction,
        "summary_by_direction": summaries,
        "note": "Dashboard hinge states and SVG highlights are read from SAP2000 Frame Hinge States exports.",
    }


def _normalized_rows_from_exact_record(record: dict[str, Any]) -> list[Any]:
    """Read normalized SAP hinge rows from one exact-export record."""
    rows: list[Any] = []
    for exported_file in record.get("exported_files", []):
        path = Path(str(exported_file))
        if path.exists():
            rows.extend(normalize_export_file(path))
    return rows


def _export_exact_hinges_from_active_session(
    candidate: dict[str, Any],
    model_path: Path,
    cfg: GeneratorConfig,
    sap_process_ids: set[int],
) -> dict[str, Any]:
    """Export exact SAP hinge tables immediately after the first pushover solve."""
    process_id = next(iter(sap_process_ids), None)
    requested_directions = set(cfg.pushover_directions)
    requested_cases = tuple(
        case_name for case_name in PUSHOVER_OUTPUT_CASES
        if case_name.rsplit("_", 1)[-1] in requested_directions
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("sap_hinge_export_worker.py")),
        "active",
        "--model",
        str(model_path),
        "--output-dir",
        str(cfg.output_dir),
        "--process-id",
        str(process_id or 0),
        "--wait-seconds",
        "300",
        "--min-rows-per-direction",
        str(max(1, cfg.exact_hinge_export_min_rows_per_direction)),
        "--output-cases",
        *requested_cases,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=420)
        record = _last_json_line(completed.stdout)
        stderr_text = completed.stderr.strip()
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        record = None
        stderr_text = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        return_code = None
    if record is None:
        record = {
            "model_path": str(model_path.resolve()),
            "model_stem": model_path.stem,
            "status": "failed",
            "duration_seconds": 0.0,
            "message": stderr_text or (
                "Exact hinge active-export subprocess timed out after 420 seconds."
                if return_code is None else f"Exact hinge active-export subprocess exited with code {return_code}."
            ),
            "exported_files": [],
            "attempt_count": 1,
            "inline_first_solve_export": True,
            "active_session_export_subprocess": True,
        }
    _append_exact_export_record(cfg.output_dir, record, cfg)
    candidate["exact_hinge_ui_export"] = {
        "available": record.get("status") == "success",
        "source": "sap_exact_frame_hinge_states",
        "mode": "inline_first_solve_export",
        "exported_files": record.get("exported_files", []),
        "validation": record.get("export_validation", {}),
        "message": record.get("message", ""),
    }
    return record


def _last_json_line(output: str) -> dict[str, Any] | None:
    """Read the final JSON object emitted by a helper subprocess."""
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _append_exact_export_record(output_dir: Path, record: dict[str, Any], cfg: GeneratorConfig) -> None:
    """Append one inline exact-export result to the dashboard validation summary."""
    summary_path = output_dir / "hinge_validation" / "export_worker_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    else:
        summary = {}
    records = [row for row in summary.get("records", []) if isinstance(row, dict)]
    records = [row for row in records if row.get("model_stem") != record.get("model_stem")]
    records.append(record)
    summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "model_count": len(records),
            "min_rows_per_direction": max(1, cfg.exact_hinge_export_min_rows_per_direction),
            "success_count": sum(row.get("status") == "success" for row in records),
            "failed_count": sum(row.get("status") != "success" for row in records),
            "cancelled": False,
            "mode": "inline_first_solve_export",
            "records": records,
            "summary_path": str(summary_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_exact_hinge_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize milestone and critical events from exact SAP hinge rows."""
    ranked_by_step = sorted(events, key=lambda event: (float(event.get("step_number") or 0.0), _level_rank(str(event.get("hinge_state_level", "A-B")))))
    plastic_events = [
        event
        for event in events
        if _level_rank(str(event.get("hinge_state_level", "A-B"))) >= _level_rank("B-IO")
    ]
    state_counts: dict[str, int] = {}
    for event in events:
        level = str(event.get("hinge_state_level", ""))
        state_counts[level] = state_counts.get(level, 0) + 1
    return {
        "event_count": len(events),
        "state_counts": state_counts,
        "first_plastic_hinge": _first_event_at_or_above(ranked_by_step, "B-IO"),
        "first_column_hinge": _first_event_at_or_above([event for event in ranked_by_step if event.get("element_type") == "column"], "B-IO"),
        "first_ls_level": _first_event_at_or_above(ranked_by_step, "LS-CP"),
        "first_cp_level": _first_event_at_or_above(ranked_by_step, "CP-C"),
        "critical_events": _ranked_exact_hinge_events(plastic_events, 8),
    }


def _ranked_exact_hinge_events(events: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return exact SAP rows ordered by state, rotation, and analysis step."""
    return sorted(
        events,
        key=lambda event: (
            _level_rank(str(event.get("hinge_state_level", "A-B"))),
            abs(float(event.get("plastic_rotation_rad") or 0.0)),
            float(event.get("step_number") or 0.0),
        ),
        reverse=True,
    )[:limit]


def _discard_failed_exact_export_models(
    output_dir: Path,
    rows: list[ModelMetadata],
    summary: dict[str, Any],
) -> list[ModelMetadata]:
    """Delete models without exact SAP hinge exports and exclude them from metadata."""
    failed_records = [
        record
        for record in summary.get("records", [])
        if isinstance(record, dict) and record.get("status") != "success" and record.get("model_stem")
    ]
    if not failed_records:
        summary["discarded_models"] = []
        _rewrite_exact_export_summary(output_dir, summary)
        return rows

    failed_stems = {str(record["model_stem"]) for record in failed_records}
    discarded_models: list[dict[str, Any]] = []
    for record in failed_records:
        stem = str(record["model_stem"])
        discarded_models.append(
            {
                "model_stem": stem,
                "message": str(record.get("message", "")),
                "deleted_artifacts": _delete_model_artifacts(output_dir, stem),
            }
        )
    retained_rows = [
        row
        for row in rows
        if not row.model_kayit_yolu or Path(row.model_kayit_yolu).stem not in failed_stems
    ]
    summary["discarded_models"] = discarded_models
    summary["retained_model_count"] = len(retained_rows)
    _rewrite_exact_export_summary(output_dir, summary)
    return retained_rows


def _mark_failed_exact_export_models(output_dir: Path, summary: dict[str, Any]) -> None:
    """Keep models without exact SAP hinge exports and mark their metadata."""
    failed_records = [
        record
        for record in summary.get("records", [])
        if isinstance(record, dict) and record.get("status") != "success" and record.get("model_stem")
    ]
    marked_models: list[dict[str, Any]] = []
    for record in failed_records:
        stem = str(record["model_stem"])
        metadata_path = output_dir / f"{stem}_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        plastic_hinges = data.setdefault("plastic_hinges", {})
        plastic_hinges["exact_ui_export"] = {
            "available": False,
            "source": "sap_exact_frame_hinge_states",
            "message": str(record.get("message", "Exact SAP hinge export failed.")),
            "exported_files": record.get("exported_files", []),
            "validation": record.get("export_validation", {}),
        }
        results = plastic_hinges.get("results", {})
        if isinstance(results, dict):
            results["exact_export_available"] = False
            results["exact_export_message"] = str(record.get("message", "Exact SAP hinge export failed."))
        metadata_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate = dict(data.get("candidate", {}))
        if isinstance(candidate.get("column_section"), dict) and isinstance(candidate.get("beam_section"), dict):
            candidate["column_section"] = Section(**candidate["column_section"])
            candidate["beam_section"] = Section(**candidate["beam_section"])
            if isinstance(results, dict):
                candidate["plastic_hinge_results"] = results
                candidate["preview_hinge_source_note"] = "Proxy hinge isaretleri: moment/kapasite ve kalibre edilebilir rotasyon backbone'u."
            _write_model_preview_svg(output_dir / f"{stem}_preview.svg", candidate)
        marked_models.append({"model_stem": stem, "message": str(record.get("message", ""))})
    summary["discarded_models"] = []
    summary["marked_models_without_exact_export"] = marked_models
    summary["retained_model_count"] = len(list(output_dir.glob("Model_*.sdb")))
    _rewrite_exact_export_summary(output_dir, summary)


def _delete_model_artifacts(output_dir: Path, model_stem: str) -> list[str]:
    """Remove one rejected model's root and raw-export files inside output_dir."""
    target = output_dir.resolve()
    if target == Path(target.anchor):
        raise RuntimeError(f"Refusing to delete model artifacts from filesystem root: {target}")
    deleted: list[str] = []
    folders = (target, target / "hinge_validation" / "raw_exports")
    for folder in folders:
        resolved_folder = folder.resolve()
        resolved_folder.relative_to(target)
        if not resolved_folder.exists():
            continue
        for artifact in resolved_folder.glob(f"{model_stem}*"):
            resolved_artifact = artifact.resolve()
            resolved_artifact.relative_to(target)
            if resolved_artifact.is_file():
                resolved_artifact.unlink()
                deleted.append(str(resolved_artifact))
    return deleted


def _rewrite_exact_export_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    """Persist exact-export cleanup decisions for audit without retaining rejected models."""
    summary_path = Path(str(summary.get("summary_path") or output_dir / "hinge_validation" / "export_worker_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _start_sap_session(progress_callback: ProgressCallback | None) -> tuple[Sap2000Api, set[int]]:
    """Start one SAP2000 COM session and remember only its newly created processes."""
    existing_processes = _windows_process_ids("SAP2000.EXE")
    sap = Sap2000Api(visible=True)
    _emit_progress(progress_callback, "sap_starting", {"message": "Starting SAP2000 COM session."})
    sap.start()
    owned_processes = _windows_process_ids("SAP2000.EXE") - existing_processes
    _emit_progress(
        progress_callback,
        "sap_started",
        {
            "message": "SAP2000 COM session started.",
            "sap_process_ids": sorted(owned_processes),
        },
    )
    return sap, owned_processes


def _close_sap_session(sap: Sap2000Api, owned_processes: set[int], force_owned_processes: bool) -> None:
    """Close one generator SAP session without terminating pre-existing user sessions."""
    if force_owned_processes and owned_processes:
        for process_id in owned_processes:
            subprocess.run(
                ["taskkill", "/PID", str(process_id), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        sap.sap_object = None
        sap.sap_model = None
        return
    try:
        sap.close(save=False)
    except Exception:  # noqa: BLE001 - a SAP v22 table viewer may terminate its own process while closing.
        for process_id in owned_processes:
            subprocess.run(
                ["taskkill", "/PID", str(process_id), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )


def _emit_progress(callback: ProgressCallback | None, event: str, payload: dict[str, Any]) -> None:
    """Notify optional callers about generation progress."""
    if callback is not None:
        callback(event, payload)


def _generate_one_model(index: int, rng: random.Random, cfg: GeneratorConfig, sap: Sap2000Api, sap_process_ids: set[int]) -> ModelMetadata:
    """Try candidates until one passes screening and can be saved in SAP2000."""
    last_candidate: dict[str, Any] | None = None
    last_validation: ValidationResult | None = None
    for _attempt in range(1, cfg.max_attempts_per_model + 1):
        candidate = _random_candidate(rng, cfg)
        validation = validate_candidate_model(candidate, cfg)
        last_candidate = candidate
        last_validation = validation
        if not validation.is_valid:
            continue

        model_name = _model_file_stem(index, candidate)
        model_path = cfg.output_dir / f"{model_name}.sdb"
        preview_path = cfg.output_dir / f"{model_name}_preview.svg"
        _write_model_preview_svg(preview_path, candidate)
        try:
            _build_sap_model(sap, candidate, model_path, cfg, sap_process_ids)
        except SapApiError as exc:
            _delete_model_artifacts(cfg.output_dir, model_path.stem)
            return _metadata_from_candidate(index, candidate, validation, model_path, "failed", f"SAP2000 error: {exc}", cfg)

        _write_model_preview_svg(preview_path, candidate)
        json_path = cfg.output_dir / f"{model_name}_metadata.json"
        _write_json_metadata(json_path, candidate, validation, model_path, "success", "", cfg)
        return _metadata_from_candidate(index, candidate, validation, model_path, "success", "", cfg)

    return _metadata_from_candidate(
        index,
        last_candidate,
        last_validation,
        Path(""),
        "failed",
        "Maximum candidate generation attempts exceeded.",
        cfg,
    )


def _random_candidate(rng: random.Random, cfg: GeneratorConfig) -> dict[str, Any]:
    """Create one randomized model candidate within configured parameter bounds."""
    story_count = rng.randint(cfg.story_count_min, cfg.story_count_max)
    x_bays = rng.randint(cfg.x_bay_count_min, cfg.x_bay_count_max)
    y_bays = rng.randint(cfg.y_bay_count_min, cfg.y_bay_count_max)
    candidate = {
        "story_count": story_count,
        "x_bay_count": x_bays,
        "y_bay_count": y_bays,
        "spans_x": [_rounded_step(rng, cfg.bay_length_min_m, cfg.bay_length_max_m, cfg.bay_length_step_m) for _ in range(x_bays)],
        "spans_y": [_rounded_step(rng, cfg.bay_length_min_m, cfg.bay_length_max_m, cfg.bay_length_step_m) for _ in range(y_bays)],
        "story_height": _rounded_step(rng, cfg.story_height_min_m, cfg.story_height_max_m, cfg.story_height_step_m),
        "concrete_class": rng.choice(cfg.concrete_classes),
        "steel_class": rng.choice(cfg.steel_classes),
        "column_section": rng.choice(cfg.column_sections),
        "beam_section": rng.choice(cfg.beam_sections),
        "rho_col": rng.choice(cfg.column_rebar_ratios),
        "beam_top_ratio_support": rng.choice(cfg.beam_top_ratio_supports),
        "beam_bottom_ratio_span": rng.choice(cfg.beam_bottom_ratio_spans),
        "pushover_target_drift_ratio": _select_pushover_target_drift(rng, cfg),
        "soil_class": rng.choice(cfg.soil_classes),
        "subgrade_modulus_kn_m3": 0.0,
        "raft_thickness_m": _rounded_step(rng, cfg.raft_thickness_min_m, cfg.raft_thickness_max_m, cfg.raft_thickness_step_m),
        "raft_rebar_ratio": rng.choice(cfg.raft_rebar_ratios),
        "slab_thickness_m": _rounded_step(rng, cfg.slab_thickness_min_m, cfg.slab_thickness_max_m, cfg.slab_thickness_step_m),
        "slab_rebar_ratio": rng.choice(cfg.slab_rebar_ratios),
        "has_shear_walls": cfg.enable_shear_walls,
        "wall_thickness_m": _rounded_step(rng, cfg.wall_thickness_min_m, cfg.wall_thickness_max_m, cfg.wall_thickness_step_m),
        "wall_length_m": _rounded_step(rng, cfg.wall_length_min_m, cfg.wall_length_max_m, cfg.wall_length_step_m),
        "wall_rebar_ratio": rng.choice(cfg.wall_rebar_ratios),
        "wall_placement": cfg.wall_placement,
        "wall_count": 0,
        "wall_count_x": 0,
        "wall_count_y": 0,
    }
    candidate["subgrade_modulus_kn_m3"] = cfg.soil_subgrade_modulus_kn_m3[candidate["soil_class"]]
    candidate["rho_beam"] = max(candidate["beam_top_ratio_support"], candidate["beam_bottom_ratio_span"])
    return candidate


def _rounded_step(rng: random.Random, min_value: float, max_value: float, step: float) -> float:
    """Return a random value on a fixed decimal step."""
    count = round((max_value - min_value) / step)
    return round(min_value + step * rng.randint(0, count), 3)


def _select_pushover_target_drift(rng: random.Random, cfg: GeneratorConfig) -> float:
    """Return fixed or randomized pushover target drift ratio."""
    if cfg.fix_pushover_target_drift_ratio:
        return round(cfg.pushover_target_drift_ratio_fixed, 3)
    return _rounded_step(
        rng,
        cfg.pushover_target_drift_ratio_min,
        cfg.pushover_target_drift_ratio_max,
        cfg.pushover_target_drift_ratio_step,
    )


def _build_sap_model(sap: Sap2000Api, candidate: dict[str, Any], model_path: Path, cfg: GeneratorConfig, sap_process_ids: set[int]) -> None:
    """Create 3D RC frame geometry, loads, diaphragms, and save the SAP2000 file."""
    sap.initialize_new_model()
    concrete = candidate["concrete_class"]
    steel = candidate["steel_class"]
    column_section: Section = candidate["column_section"]
    beam_section: Section = candidate["beam_section"]

    sap.define_concrete_material(concrete)
    sap.define_rebar_material(steel, steel_yield_strength_mpa(steel, cfg))
    col_prop = f"COL_{column_section.label_cm}"
    beam_prop = f"BEAM_{beam_section.label_cm}"
    sap.define_rectangular_frame_section(col_prop, concrete, column_section)
    sap.define_rectangular_frame_section(beam_prop, concrete, beam_section)
    raft_prop = f"RAFT_{round(candidate['raft_thickness_m'] * 100):03d}cm"
    if cfg.enable_raft_foundation:
        sap.define_raft_shell_section(raft_prop, concrete, candidate["raft_thickness_m"])
    slab_prop = f"SLAB_{round(candidate['slab_thickness_m'] * 100):03d}cm"
    if cfg.enable_floor_slabs:
        sap.define_slab_shell_section(slab_prop, concrete, candidate["slab_thickness_m"])
    wall_prop = f"WALL_{round(candidate['wall_thickness_m'] * 100):03d}cm"
    if candidate.get("has_shear_walls"):
        sap.define_wall_shell_section(wall_prop, concrete, candidate["wall_thickness_m"])
    sap.define_load_patterns()
    if cfg.enable_pushover_cases:
        sap.define_pushover_load_patterns(cfg.pushover_directions)
    sap.set_mass_source()

    x_coords = _cumulative_coordinates(candidate["spans_x"])
    y_coords = _cumulative_coordinates(candidate["spans_y"])
    z_coords = [round(i * candidate["story_height"], 3) for i in range(candidate["story_count"] + 1)]

    base_points = [Point3D(x, y, 0.0) for x in x_coords for y in y_coords]
    floor_points: dict[int, list[Point3D]] = {story: [] for story in range(1, candidate["story_count"] + 1)}
    column_names: list[str] = []
    beam_names: list[str] = []

    for story in range(candidate["story_count"]):
        z_bot = z_coords[story]
        z_top = z_coords[story + 1]
        for x in x_coords:
            for y in y_coords:
                column_names.append(sap.add_frame(Point3D(x, y, z_bot), Point3D(x, y, z_top), col_prop, f"C_{story + 1}_{x:.2f}_{y:.2f}"))
                floor_points[story + 1].append(Point3D(x, y, z_top))

    for story in range(1, candidate["story_count"] + 1):
        z = z_coords[story]
        for y in y_coords:
            for i in range(len(x_coords) - 1):
                beam_names.append(sap.add_frame(Point3D(x_coords[i], y, z), Point3D(x_coords[i + 1], y, z), beam_prop, f"BX_{story}_{i}_{y:.2f}"))
        for x in x_coords:
            for j in range(len(y_coords) - 1):
                beam_names.append(sap.add_frame(Point3D(x, y_coords[j], z), Point3D(x, y_coords[j + 1], z), beam_prop, f"BY_{story}_{j}_{x:.2f}"))

    if cfg.enable_floor_slabs:
        candidate["slab_panel_count"] = _add_floor_slabs(sap, x_coords, y_coords, z_coords, slab_prop)
    else:
        candidate["slab_panel_count"] = 0

    if candidate.get("has_shear_walls"):
        wall_summary = _add_shear_walls(sap, candidate, x_coords, y_coords, z_coords, wall_prop)
        candidate["wall_count"] = wall_summary["total"]
        candidate["wall_count_x"] = wall_summary["x"]
        candidate["wall_count_y"] = wall_summary["y"]

    if cfg.enable_raft_foundation:
        sap.add_raft_area(_raft_corner_points(x_coords, y_coords, cfg.foundation_edge_offset_m), raft_prop, "RAFT_FOUNDATION")
        if cfg.assign_soil_vertical_springs:
            sap.set_base_restraints_for_soil_springs(base_points)
            sap.assign_vertical_soil_springs(_base_point_spring_stiffness(x_coords, y_coords, candidate["subgrade_modulus_kn_m3"]))
        else:
            sap.set_base_restraints(base_points)
    else:
        sap.set_base_restraints(base_points)
    diaphragms = sap.define_rigid_diaphragms(candidate["story_count"])
    sap.assign_story_diaphragms(floor_points, diaphragms)
    sap.assign_uniform_frame_loads(beam_names, cfg.dead_load_kn_m, cfg.live_load_kn_m)
    sap.define_modal_case()
    if cfg.enable_plastic_hinges:
        candidate["plastic_hinge_assignment"] = _assign_plastic_hinges(sap, candidate, model_path, column_names, beam_names, cfg)
    if cfg.enable_pushover_cases:
        sap.assign_pushover_story_loads(
            floor_points,
            candidate["story_height"],
            cfg.pushover_base_shear_proxy_kn,
            cfg.pushover_directions,
        )
        _define_pushover_cases(sap, candidate, x_coords, y_coords, z_coords, cfg)
    sap.save(model_path)
    if cfg.enable_plastic_hinges:
        candidate["plastic_hinge_assignment"].update(
            _export_patch_import_hinges(sap, candidate, model_path, column_names, beam_names, cfg)
        )
        candidate["hinge_inventory"] = _build_hinge_inventory(candidate, column_names, beam_names, cfg)
        _raise_if_required_hinges_missing(candidate["plastic_hinge_assignment"], cfg)
    if cfg.enable_sap_screenshots:
        sap.capture_model_screenshot(model_path.with_name(f"{model_path.stem}_sap_model.png"), cfg.sap_screenshot_delay_s)
    if cfg.run_analysis_after_save:
        candidate["analysis_status"] = sap.run_analysis(_analysis_case_names(cfg))
        if cfg.enable_pushover_cases:
            candidate["pushover_results"] = _read_pushover_results(sap, candidate, cfg)
            candidate["story_drift_results"] = _read_story_drift_results(sap, candidate, x_coords, y_coords, z_coords, cfg)
        if cfg.enable_plastic_hinges:
            candidate["plastic_hinge_results"] = _read_plastic_hinge_results(sap, candidate, cfg)
            _write_proxy_hinge_events_csv(cfg.output_dir, model_path.stem, candidate["plastic_hinge_results"])
        sap.save_current()
        if _inline_exact_export_enabled(cfg):
            exact_record = _export_exact_hinges_from_active_session(candidate, model_path, cfg, sap_process_ids)
            exact_results = _exact_hinge_results(_normalized_rows_from_exact_record(exact_record), exact_record)
            if not exact_results.get("available"):
                raise SapApiError("Exact SAP hinge export produced no usable X/Y hinge events.")
            candidate["plastic_hinge_results"] = exact_results
            candidate["exact_hinge_export_record"] = exact_record
        if cfg.enable_sap_screenshots:
            sap.capture_deformed_screenshot(model_path.with_name(f"{model_path.stem}_sap_deformed.png"), "PUSHOVER_X", cfg.sap_screenshot_delay_s)


def _cumulative_coordinates(spans: list[float]) -> list[float]:
    """Return grid coordinates from span lengths."""
    coords = [0.0]
    for span in spans:
        coords.append(round(coords[-1] + span, 3))
    return coords


def _raft_corner_points(x_coords: list[float], y_coords: list[float], edge_offset: float) -> list[Point3D]:
    """Return raft boundary corner points with a small edge projection."""
    x_min = x_coords[0] - edge_offset
    x_max = x_coords[-1] + edge_offset
    y_min = y_coords[0] - edge_offset
    y_max = y_coords[-1] + edge_offset
    return [
        Point3D(x_min, y_min, 0.0),
        Point3D(x_max, y_min, 0.0),
        Point3D(x_max, y_max, 0.0),
        Point3D(x_min, y_max, 0.0),
    ]


def _add_floor_slabs(sap: Sap2000Api, x_coords: list[float], y_coords: list[float], z_coords: list[float], slab_prop: str) -> int:
    """Add one horizontal shell slab panel per bay at every elevated story."""
    count = 0
    for story, z in enumerate(z_coords[1:], start=1):
        for ix in range(len(x_coords) - 1):
            for iy in range(len(y_coords) - 1):
                sap.add_slab_area(
                    [
                        Point3D(x_coords[ix], y_coords[iy], z),
                        Point3D(x_coords[ix + 1], y_coords[iy], z),
                        Point3D(x_coords[ix + 1], y_coords[iy + 1], z),
                        Point3D(x_coords[ix], y_coords[iy + 1], z),
                    ],
                    slab_prop,
                    f"SLAB_{story}_{ix}_{iy}",
                )
                count += 1
    return count


def _base_point_spring_stiffness(x_coords: list[float], y_coords: list[float], subgrade_modulus_kn_m3: float) -> dict[Point3D, float]:
    """Return vertical Winkler point spring stiffness from tributary areas.

    This k_point = k_s * A_trib model is a preliminary synthetic-data proxy,
    not a final geotechnical raft-soil interaction design.
    """
    stiffness: dict[Point3D, float] = {}
    for i, x in enumerate(x_coords):
        tributary_x = _tributary_length(x_coords, i)
        for j, y in enumerate(y_coords):
            tributary_y = _tributary_length(y_coords, j)
            stiffness[Point3D(x, y, 0.0)] = round(subgrade_modulus_kn_m3 * tributary_x * tributary_y, 3)
    return stiffness


def _tributary_length(coords: list[float], index: int) -> float:
    """Return tributary length for one grid coordinate."""
    if len(coords) == 1:
        return 1.0
    if index == 0:
        return 0.5 * (coords[1] - coords[0])
    if index == len(coords) - 1:
        return 0.5 * (coords[-1] - coords[-2])
    return 0.5 * (coords[index + 1] - coords[index - 1])


def _add_shear_walls(
    sap: Sap2000Api,
    candidate: dict[str, Any],
    x_coords: list[float],
    y_coords: list[float],
    z_coords: list[float],
    wall_prop: str,
) -> dict[str, int]:
    """Add vertical shear wall shell panels in both X and Y plan directions."""
    placements = _wall_segments(candidate, x_coords, y_coords)
    wall_names: list[str] = []
    for story in range(candidate["story_count"]):
        z_bot = z_coords[story]
        z_top = z_coords[story + 1]
        for idx, (direction, start, end) in enumerate(placements, start=1):
            corners = [
                Point3D(start.x, start.y, z_bot),
                Point3D(end.x, end.y, z_bot),
                Point3D(end.x, end.y, z_top),
                Point3D(start.x, start.y, z_top),
            ]
            wall_names.append(sap.add_wall_area(corners, wall_prop, f"W{direction}_{story + 1}_{idx}"))
    x_count_per_story = sum(1 for direction, _, _ in placements if direction == "X")
    y_count_per_story = sum(1 for direction, _, _ in placements if direction == "Y")
    return {
        "total": len(wall_names),
        "x": x_count_per_story * candidate["story_count"],
        "y": y_count_per_story * candidate["story_count"],
    }


def _wall_segments(candidate: dict[str, Any], x_coords: list[float], y_coords: list[float]) -> list[tuple[str, Point3D, Point3D]]:
    """Return plan wall centerline segments in X and Y directions.

    A wall segment tagged X has its long edge parallel to global X and primarily
    contributes in-plane stiffness against Y-direction lateral action. A segment
    tagged Y has its long edge parallel to global Y and contributes against
    X-direction lateral action. Both are generated for balanced wall layouts.
    """
    wall_length = candidate["wall_length_m"]
    segments: list[tuple[str, Point3D, Point3D]] = []
    x_mid_i = max(0, (len(x_coords) - 2) // 2)
    y_mid_j = max(0, (len(y_coords) - 2) // 2)

    def limited_segment_x(y: float, i: int) -> tuple[Point3D, Point3D]:
        x1, x2 = x_coords[i], x_coords[i + 1]
        length = min(wall_length, x2 - x1)
        center = 0.5 * (x1 + x2)
        return Point3D(center - 0.5 * length, y, 0.0), Point3D(center + 0.5 * length, y, 0.0)

    def limited_segment_y(x: float, j: int) -> tuple[Point3D, Point3D]:
        y1, y2 = y_coords[j], y_coords[j + 1]
        length = min(wall_length, y2 - y1)
        center = 0.5 * (y1 + y2)
        return Point3D(x, center - 0.5 * length, 0.0), Point3D(x, center + 0.5 * length, 0.0)

    x_segments = [limited_segment_x(y_coords[0], x_mid_i)]
    if len(y_coords) > 1:
        x_segments.append(limited_segment_x(y_coords[-1], x_mid_i))
    y_segments = [limited_segment_y(x_coords[0], y_mid_j)]
    if len(x_coords) > 1:
        y_segments.append(limited_segment_y(x_coords[-1], y_mid_j))
    segments.extend(("X", start, end) for start, end in x_segments)
    segments.extend(("Y", start, end) for start, end in y_segments)
    return segments


def _assign_plastic_hinges(
    sap: Sap2000Api,
    candidate: dict[str, Any],
    model_path: Path,
    column_names: list[str],
    beam_names: list[str],
    cfg: GeneratorConfig,
) -> dict[str, Any]:
    """Initialize preliminary plastic hinge assignment metadata."""
    column_target_hinges = len(column_names) * len(cfg.hinge_relative_distances) * 2
    summaries: dict[str, Any] = {
        "enabled": True,
        "method": "sap2000_text_patch",
        "hinge_model": "Explicit user-defined M2/M3 hinges for columns; M3 hinges for beams",
        "column_property": _column_hinge_names()[0] + "," + _column_hinge_names()[1],
        "beam_property": _beam_hinge_name(),
        "shear_v2_property": cfg.shear_hinge_v2_property,
        "shear_v3_property": cfg.shear_hinge_v3_property,
        "assigned_count": 0,
        "failed_count": 0,
        "warnings": [],
        "text_model_path": str(model_path.with_suffix(".$2k")),
        "patched_text_model_path": str(model_path.with_name(f"{model_path.stem}_hinges.$2k")),
    }
    if cfg.assign_column_pmm_hinges:
        summaries["columns"] = {
            "target_frames": len(column_names),
            "target_hinges": column_target_hinges,
            "property": _column_hinge_names()[0] + "," + _column_hinge_names()[1],
            "note": "SAP2000 v22 text import for Default-PMM auto hinges needs the auto-hinge assignment table format. Until that table is mapped, columns use explicit user-defined M2 and M3 end hinges so pushover can run without import errors.",
        }
    if cfg.assign_beam_m3_hinges:
        summaries["beams"] = {
            "target_frames": len(beam_names),
            "target_hinges": len(beam_names) * len(cfg.hinge_relative_distances),
            "property": _beam_hinge_name(),
        }
    if cfg.assign_shear_hinges:
        summaries["warnings"].append("Shear hinge text patch is not implemented yet; V2/V3 hinge option is ignored in this stage.")
    return summaries


def _merge_hinge_summary(target: dict[str, Any], label: str, summary: object) -> None:
    """Merge one SAP hinge assignment summary into a JSON-safe dict."""
    assigned = int(getattr(summary, "assigned_count", 0))
    failed = int(getattr(summary, "failed_count", 0))
    warnings = list(getattr(summary, "warnings", []))
    target[label] = {"assigned_count": assigned, "failed_count": failed, "warnings": warnings[:10]}
    target["assigned_count"] += assigned
    target["failed_count"] += failed
    target["warnings"].extend(warnings)


def _export_patch_import_hinges(
    sap: Sap2000Api,
    candidate: dict[str, Any],
    model_path: Path,
    column_names: list[str],
    beam_names: list[str],
    cfg: GeneratorConfig,
) -> dict[str, Any]:
    """Assign frame hinges by patching SAP2000 text tables and reopening them.

    SAP2000 v22's local OAPI type library exposes hinge read APIs but no
    FrameObj.SetHingeAssign writer. The text model format, however, contains
    explicit hinge definition and assignment tables. This path exports the
    current model to .$2k, inserts user-defined hinge definitions and
    assignments, reopens the patched text model, then saves the .sdb again.
    """
    text_path = model_path.with_suffix(".$2k")
    patched_path = model_path.with_name(f"{model_path.stem}_hinges.$2k")
    sap.save_text(text_path)
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    patched = _patch_sap_text_hinges(text, candidate, column_names, beam_names, cfg)
    patched_path.write_text(patched, encoding="utf-8")
    sap._start_import_log_autoclose(timeout_s=180)
    sap.open_file(patched_path)
    sap._start_import_log_autoclose(timeout_s=30)
    verification = sap.count_frame_hinge_assignments([*column_names, *beam_names])
    sap.save(model_path)

    expected = 0
    if cfg.assign_column_pmm_hinges:
        expected += len(column_names) * len(cfg.hinge_relative_distances) * 2
    if cfg.assign_beam_m3_hinges:
        expected += len(beam_names) * len(cfg.hinge_relative_distances)
    assigned = int(verification.get("total_hinges", 0))
    return {
        "method": "sap2000_text_patch",
        "text_model_path": str(text_path),
        "patched_text_model_path": str(patched_path),
        "expected_hinges": expected,
        "assigned_count": assigned,
        "failed_count": max(expected - assigned, 0),
        "verification": verification,
        "warnings": list(verification.get("warnings", [])),
    }


def _patch_sap_text_hinges(
    text: str,
    candidate: dict[str, Any],
    column_names: list[str],
    beam_names: list[str],
    cfg: GeneratorConfig,
) -> str:
    """Insert hinge definition and assignment tables into a SAP2000 .$2k file."""
    tables = _sap_text_hinge_tables(candidate, column_names, beam_names, cfg)
    marker = 'TABLE:  "JOINT COORDINATES"'
    if marker not in text:
        raise SapApiError("Could not find JOINT COORDINATES table while patching hinge tables.")
    return text.replace(marker, tables + "\n" + marker, 1)


def _sap_text_hinge_tables(candidate: dict[str, Any], column_names: list[str], beam_names: list[str], cfg: GeneratorConfig) -> str:
    """Build SAP2000 text tables for explicit user-defined frame hinges.

    SAP2000's Default-PMM is an auto hinge and does not belong in the
    user-defined hinge assignment table. Until the exact auto-hinge table
    schema is mapped, this writer uses explicit M2/M3 column hinges and M3 beam
    hinges so text import and nonlinear analysis remain reliable.
    """
    col_m2, col_m3 = _column_hinge_names()
    beam_m3 = _beam_hinge_name()
    definitions: list[tuple[str, float, float]] = []
    col_capacity = _section_moment_capacity_kn_m(candidate["column_section"], candidate["rho_col"], candidate["steel_class"], cfg)
    beam_capacity = _section_moment_capacity_kn_m(candidate["beam_section"], candidate["rho_beam"], candidate["steel_class"], cfg)
    if cfg.assign_column_pmm_hinges:
        definitions.extend([(col_m2, col_capacity, 0.020), (col_m3, col_capacity, 0.020)])
    if cfg.assign_beam_m3_hinges:
        definitions.append((beam_m3, beam_capacity, 0.030))

    lines = [
        'TABLE:  "FRAME HINGE ASSIGNS 02 - USER DEFINED PROPERTIES"',
    ]
    if cfg.assign_column_pmm_hinges:
        for frame in column_names:
            for hinge_name in (col_m2, col_m3):
                for rel_dist in cfg.hinge_relative_distances:
                    lines.append(f"   Frame={_q(frame)}   AssignProp={_q(hinge_name)}   DistType=RelDist   RelDist={_sap_num(rel_dist)}")
    if cfg.assign_beam_m3_hinges:
        for frame in beam_names:
            for rel_dist in cfg.hinge_relative_distances:
                lines.append(f"   Frame={_q(frame)}   AssignProp={_q(beam_m3)}   DistType=RelDist   RelDist={_sap_num(rel_dist)}")

    lines.extend([
        "",
        'TABLE:  "FRAME HINGE ASSIGNS 09 - HINGE OVERWRITES"',
    ])
    for frame in [*column_names, *beam_names]:
        lines.append(f"   Frame={_q(frame)}   AutoDivide=No   NoLoadDrop=No   LimNegStiff=0,1")

    if not definitions:
        return "\n".join(lines) + "\n"

    lines.extend(["", 'TABLE:  "HINGES DEF 02 - NONINTERACTING - DEFORM CONTROL - GENERAL"'])
    for hinge_name, _, _ in definitions:
        dof = "Moment M2" if hinge_name.endswith("M2") else "Moment M3"
        lines.append(
            f"   HingeName={_q(hinge_name)}   DOFType={_q(dof)}   Symmetric=Yes   BeyondE=\"To Zero\"   "
            "FDType=Moment-Rot   UseYldForce=No   UseYldDispl=No   MRPosMoSF=1   MRPosRoSF=1   "
            "MRNegMoSF=1   MRNegRoSF=1   LengthType=Absolute   SSAbsLen=1   HysType=Isotropic"
        )

    lines.extend(["", 'TABLE:  "HINGES DEF 03 - NONINTERACTING - DEFORM CONTROL - FORCE-DEFORM"'])
    for hinge_name, moment_capacity, theta_y in definitions:
        for point_id, force, rotation in _hinge_force_deformation_points(moment_capacity, theta_y):
            lines.append(f"   HingeName={_q(hinge_name)}   FDPoint={point_id}   Force={_sap_num(force)}   Displ={_sap_num(rotation)}")

    lines.extend(["", 'TABLE:  "HINGES DEF 04 - NONINTERACTING - DEFORM CONTROL - ACCEPTANCE"'])
    for hinge_name, _, theta_y in definitions:
        lines.append(f"   HingeName={_q(hinge_name)}   ACPoint=IO   ACPos={_sap_num(2 * theta_y)}   ACNeg={_sap_num(-2 * theta_y)}")
        lines.append(f"   HingeName={_q(hinge_name)}   ACPoint=LS   ACPos={_sap_num(4 * theta_y)}   ACNeg={_sap_num(-4 * theta_y)}")
        lines.append(f"   HingeName={_q(hinge_name)}   ACPoint=CP   ACPos={_sap_num(6 * theta_y)}   ACNeg={_sap_num(-6 * theta_y)}")
    return "\n".join(lines) + "\n"


def _build_hinge_inventory(candidate: dict[str, Any], column_names: list[str], beam_names: list[str], cfg: GeneratorConfig) -> list[dict[str, Any]]:
    """Return generated hinge assignment records used for result post-processing."""
    inventory: list[dict[str, Any]] = []
    col_m2, col_m3 = _column_hinge_names()
    beam_m3 = _beam_hinge_name()
    col_capacity = _section_moment_capacity_kn_m(candidate["column_section"], candidate["rho_col"], candidate["steel_class"], cfg)
    beam_capacity = _section_moment_capacity_kn_m(candidate["beam_section"], candidate["rho_beam"], candidate["steel_class"], cfg)
    if cfg.assign_column_pmm_hinges:
        for frame in column_names:
            for hinge_name, dof in ((col_m2, "M2"), (col_m3, "M3")):
                for rel_dist in cfg.hinge_relative_distances:
                    inventory.append(_hinge_inventory_row(frame, "column", hinge_name, dof, rel_dist, col_capacity, 0.020))
    if cfg.assign_beam_m3_hinges:
        for frame in beam_names:
            for rel_dist in cfg.hinge_relative_distances:
                inventory.append(_hinge_inventory_row(frame, "beam", beam_m3, "M3", rel_dist, beam_capacity, 0.030))
    return inventory


def _hinge_inventory_row(
    frame: str,
    element_type: str,
    hinge_name: str,
    dof: str,
    relative_distance: float,
    moment_capacity_kn_m: float,
    yield_rotation_rad: float,
) -> dict[str, Any]:
    """Create one hinge inventory row."""
    if relative_distance <= 0.001:
        location = "i-end"
    elif relative_distance >= 0.999:
        location = "j-end"
    else:
        location = "relative distance"
    return {
        "element_name": frame,
        "element_type": element_type,
        "hinge_property": hinge_name,
        "hinge_dof": dof,
        "hinge_location": location,
        "relative_distance": float(relative_distance),
        "moment_capacity_kn_m": moment_capacity_kn_m,
        "yield_rotation_rad": yield_rotation_rad,
    }


def _hinge_force_deformation_points(moment_capacity: float, theta_y: float) -> list[tuple[str, float, float]]:
    """Return symmetric A-B-C-D-E points for a simple moment-rotation hinge."""
    my = max(moment_capacity, 1.0)
    return [
        ("-E", -0.20 * my, -8.0 * theta_y),
        ("-D", -0.20 * my, -4.0 * theta_y),
        ("-C", -1.20 * my, -4.0 * theta_y),
        ("-B", -1.00 * my, 0.0),
        ("A", 0.0, 0.0),
        ("B", 1.00 * my, 0.0),
        ("C", 1.20 * my, 4.0 * theta_y),
        ("D", 0.20 * my, 4.0 * theta_y),
        ("E", 0.20 * my, 8.0 * theta_y),
    ]


def _section_moment_capacity_kn_m(section: Section, rho: float, steel_class: str, cfg: GeneratorConfig) -> float:
    """Approximate plastic moment capacity used only for synthetic hinge curves."""
    fy_kn_m2 = steel_yield_strength_mpa(steel_class, cfg) * 1000.0
    return round(max(rho * section.width * section.depth**2 * fy_kn_m2, 1.0), 3)


def _column_hinge_names() -> tuple[str, str]:
    """Return generated column hinge property names."""
    return ("GEN_COL_M2", "GEN_COL_M3")


def _beam_hinge_name() -> str:
    """Return generated beam hinge property name."""
    return "GEN_BEAM_M3"


def _q(value: str) -> str:
    """Quote SAP text table values when needed."""
    return f'"{value}"' if any(char in value for char in (" ", "-", ".")) else value


def _sap_num(value: float) -> str:
    """Format a number for SAP2000 text tables in the current locale style."""
    text = f"{float(value):.8g}"
    return text.replace(".", ",")


def _raise_if_required_hinges_missing(summary: dict[str, Any], cfg: GeneratorConfig) -> None:
    """Fail the model when mandatory hinge assignment could not be verified."""
    if not cfg.require_plastic_hinge_assignment:
        return
    expected = int(summary.get("expected_hinges", 0))
    assigned = int(summary.get("assigned_count", 0))
    if expected <= 0 or assigned < expected:
        raise SapApiError(f"Plastic hinge assignment could not be verified. Expected {expected}, assigned {assigned}.")


def _define_pushover_cases(
    sap: Sap2000Api,
    candidate: dict[str, Any],
    x_coords: list[float],
    y_coords: list[float],
    z_coords: list[float],
    cfg: GeneratorConfig,
) -> None:
    """Define X/Y pushover cases using roof control joint and target drift."""
    control_point = _nearest_roof_control_point(x_coords, y_coords, z_coords[-1])
    candidate["pushover_control_point"] = asdict(control_point)
    total_height = candidate["story_count"] * candidate["story_height"]
    target_displacement = round(total_height * candidate["pushover_target_drift_ratio"], 4)
    candidate["pushover_saved_state_settings"] = {}
    for direction in cfg.pushover_directions:
        direction_upper = direction.upper()
        point_name = sap.define_pushover_case(
            name=f"PUSHOVER_{direction_upper}",
            load_pattern=f"PUSH_{direction_upper}",
            control_point=control_point,
            direction=direction_upper,
            target_displacement_m=target_displacement,
        )
        candidate["pushover_control_point_name"] = point_name
        candidate["pushover_saved_state_settings"][direction_upper] = sap.get_static_nonlinear_results_saved(f"PUSHOVER_{direction_upper}")


def _story_drift_control_points(
    x_coords: list[float],
    y_coords: list[float],
    z_coords: list[float],
) -> dict[int, Point3D]:
    """Return one plan-center control joint per story for drift calculations."""
    x_mid = 0.5 * (x_coords[0] + x_coords[-1])
    y_mid = 0.5 * (y_coords[0] + y_coords[-1])
    x = min(x_coords, key=lambda value: abs(value - x_mid))
    y = min(y_coords, key=lambda value: abs(value - y_mid))
    return {story: Point3D(x, y, z_coords[story]) for story in range(1, len(z_coords))}


def _read_pushover_results(sap: Sap2000Api, candidate: dict[str, Any], cfg: GeneratorConfig) -> dict[str, object]:
    """Read X/Y pushover capacity curve summaries after nonlinear analysis."""
    raw_point = candidate.get("pushover_control_point", {})
    control_point = Point3D(float(raw_point.get("x", 0.0)), float(raw_point.get("y", 0.0)), float(raw_point.get("z", 0.0)))
    control_point_name = str(candidate.get("pushover_control_point_name") or "")
    curves: dict[str, object] = {}
    for direction in cfg.pushover_directions:
        direction_upper = direction.upper()
        case_name = f"PUSHOVER_{direction_upper}"
        curves[direction_upper] = sap.read_pushover_curve(case_name, control_point, direction_upper, control_point_name or None)
    return {
        "control_point": asdict(control_point),
        "control_point_name": control_point_name,
        "saved_state_settings": candidate.get("pushover_saved_state_settings", {}),
        "curves": curves,
        "note": "Capacity curve uses SAP2000 base reaction and roof control-point displacement at saved nonlinear steps.",
    }


def _read_story_drift_results(
    sap: Sap2000Api,
    candidate: dict[str, Any],
    x_coords: list[float],
    y_coords: list[float],
    z_coords: list[float],
    cfg: GeneratorConfig,
) -> dict[str, object]:
    """Read interstory drift histories for each pushover direction."""
    story_points = _story_drift_control_points(x_coords, y_coords, z_coords)
    point_names = {story: sap._get_point_name(point) for story, point in story_points.items()}
    by_direction: dict[str, object] = {}
    for direction in cfg.pushover_directions:
        direction_upper = direction.upper()
        by_direction[direction_upper] = sap.read_story_drift_history(
            f"PUSHOVER_{direction_upper}",
            story_points,
            float(candidate["story_height"]),
            direction_upper,
            point_names,
        )
    max_rows = [
        value.get("max_drift")
        for value in by_direction.values()
        if isinstance(value, dict) and isinstance(value.get("max_drift"), dict)
    ]
    return {
        "available": bool(max_rows),
        "control_points": {story: asdict(point) for story, point in story_points.items()},
        "control_point_names": point_names,
        "by_direction": by_direction,
        "max_drift": max(max_rows, key=lambda row: float(row.get("drift_ratio", 0.0)), default=None),
        "note": "Interstory drift is computed from SAP2000 JointDispl results at the nearest plan-center diaphragm joint of each story.",
    }


def _read_plastic_hinge_results(sap: Sap2000Api, candidate: dict[str, Any], cfg: GeneratorConfig) -> dict[str, object]:
    """Read or derive per-hinge pushover result records.

    SAP2000 v22's local OAPI exposes frame force histories but not a direct
    frame-hinge result table in the generated type library. Therefore this
    function preserves the direct hinge-result attempt and also creates a
    transparent demand/capacity proxy from frame-end forces. Plastic rotation is
    left null unless a future SAP result table reader supplies it.
    """
    exact_result = sap.read_frame_hinge_state_summary(cfg.hinge_result_case)
    inventory = candidate.get("hinge_inventory", [])
    if not isinstance(inventory, list) or not inventory:
        return {
            "exact_result": exact_result,
            "available": False,
            "message": "No hinge inventory was available for post-processing.",
            "proxy_events_by_direction": {},
            "summary_by_direction": {},
        }

    frames = sorted({str(item.get("element_name")) for item in inventory if item.get("element_name")})
    curves = {}
    pushover_results = candidate.get("pushover_results", {})
    if isinstance(pushover_results, dict):
        curves = pushover_results.get("curves", {}) if isinstance(pushover_results.get("curves"), dict) else {}

    events_by_direction: dict[str, Any] = {}
    summary_by_direction: dict[str, Any] = {}
    calibration = _load_proxy_calibration(cfg.output_dir)
    for direction in cfg.pushover_directions:
        direction_upper = direction.upper()
        case_name = f"PUSHOVER_{direction_upper}"
        force_history = sap.read_frame_force_history(case_name, frames)
        curve = curves.get(direction_upper, {}) if isinstance(curves, dict) else {}
        events = _derive_hinge_proxy_events(inventory, force_history, curve if isinstance(curve, dict) else {}, calibration)
        summary = _summarize_hinge_proxy_events(events)
        events_by_direction[direction_upper] = {
            "case": case_name,
            "available": bool(events),
            "event_count": len(events),
            "events": events,
            "frame_force_record_count": force_history.get("record_count", 0),
            "warnings": force_history.get("warnings", []),
        }
        summary_by_direction[direction_upper] = summary

    return {
        "available": any(value.get("available") for value in events_by_direction.values()),
        "exact_result": exact_result,
        "inventory_count": len(inventory),
        "proxy_note": "Hinge states and plastic rotations are estimated from SAP2000 FrameForce results using the generated hinge moment-rotation backbone because this SAP2000 v22 OAPI type library does not expose direct plastic hinge result tables.",
        "proxy_calibration": calibration,
        "level_definition": {
            "A-B": "plastic rotation = 0",
            "B-IO": "0 < theta_p < IO",
            "IO-LS": "IO <= theta_p < LS",
            "LS-CP": "LS <= theta_p < CP",
            "CP-C": "CP <= theta_p < C",
            "C-D": "C <= theta_p < D",
            "D-E": "D <= theta_p < E",
            "beyond E": "theta_p >= E",
        },
        "proxy_events_by_direction": events_by_direction,
        "summary_by_direction": summary_by_direction,
    }


def _derive_hinge_proxy_events(
    inventory: list[dict[str, Any]],
    force_history: dict[str, object],
    curve: dict[str, Any],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create per-hinge state rows from frame force history."""
    records = force_history.get("records", [])
    if not isinstance(records, list) or not records:
        return []

    frame_lengths: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        frame = str(record.get("frame", ""))
        station = _float_or_none(record.get("object_station"))
        if frame and station is not None:
            frame_lengths[frame] = max(frame_lengths.get(frame, 0.0), station)

    roof_base_by_step = _roof_base_by_step(curve)
    events: list[dict[str, Any]] = []
    for hinge in inventory:
        frame = str(hinge.get("element_name", ""))
        if not frame:
            continue
        target_station = float(hinge.get("relative_distance", 0.0)) * max(frame_lengths.get(frame, 0.0), 0.0)
        hinge_records = [record for record in records if isinstance(record, dict) and str(record.get("frame", "")) == frame]
        closest_by_step: dict[tuple[float, str], dict[str, Any]] = {}
        for record in hinge_records:
            station = _float_or_none(record.get("object_station"))
            step = _float_or_none(record.get("step_number"))
            load_step = str(record.get("load_step", ""))
            if station is None or step is None:
                continue
            key = (round(step, 8), load_step)
            existing = closest_by_step.get(key)
            if existing is None or abs(station - target_station) < abs(float(existing.get("object_station", 0.0)) - target_station):
                closest_by_step[key] = record
        for (_, _), record in sorted(closest_by_step.items(), key=lambda item: (item[0][0], item[0][1])):
            event = _hinge_event_from_force_record(hinge, record, roof_base_by_step, calibration)
            if event is not None:
                events.append(event)
    return events


def _hinge_event_from_force_record(
    hinge: dict[str, Any],
    record: dict[str, Any],
    roof_base_by_step: dict[float, dict[str, Any]],
    calibration: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert one frame-force record to one hinge proxy event."""
    dof = str(hinge.get("hinge_dof", "M3"))
    moment = _float_or_none(record.get(dof))
    capacity = _float_or_none(hinge.get("moment_capacity_kn_m"))
    step = _float_or_none(record.get("step_number"))
    if moment is None or capacity is None or capacity <= 0.0 or step is None:
        return None
    ratio = abs(moment) / capacity
    yield_rotation = _float_or_none(hinge.get("yield_rotation_rad")) or 0.020
    uncalibrated_rotation = _plastic_rotation_from_ratio(ratio, yield_rotation)
    rotation_scale = _proxy_rotation_scale(calibration, str(hinge.get("element_type", "")), dof)
    plastic_rotation = uncalibrated_rotation * rotation_scale
    step_data = roof_base_by_step.get(round(step, 8), {})
    return {
        "step_number": step,
        "load_step": str(record.get("load_step", "")),
        "roof_displacement_m": step_data.get("roof_displacement_m"),
        "base_shear_kn": step_data.get("base_shear_kn"),
        "element_name": hinge.get("element_name"),
        "element_type": hinge.get("element_type"),
        "hinge_property": hinge.get("hinge_property"),
        "hinge_dof": dof,
        "hinge_location": hinge.get("hinge_location"),
        "relative_distance": hinge.get("relative_distance"),
        "hinge_state_level": _hinge_state_from_rotation(
            plastic_rotation,
            yield_rotation,
            calibration,
            str(hinge.get("element_type", "")),
            dof,
        ),
        "demand_capacity_ratio": round(ratio, 4),
        "plastic_rotation_rad": round(plastic_rotation, 6),
        "plastic_rotation_source": "estimated from frame force demand and calibrated generated hinge moment-rotation backbone",
        "plastic_rotation_uncalibrated_rad": round(uncalibrated_rotation, 6),
        "proxy_rotation_scale": round(rotation_scale, 6),
        "rotation_io_rad": round(2.0 * yield_rotation, 6),
        "rotation_ls_rad": round(4.0 * yield_rotation, 6),
        "rotation_cp_rad": round(6.0 * yield_rotation, 6),
        "rotation_c_rad": round(4.0 * yield_rotation, 6),
        "rotation_d_rad": round(6.0 * yield_rotation, 6),
        "rotation_e_rad": round(8.0 * yield_rotation, 6),
        "moment_kn_m": round(moment, 4),
        "axial_force_kn": _round_optional(record.get("P"), 4),
        "shear_v2_kn": _round_optional(record.get("V2"), 4),
        "shear_v3_kn": _round_optional(record.get("V3"), 4),
    }


def _roof_base_by_step(curve: dict[str, Any]) -> dict[float, dict[str, Any]]:
    """Index capacity curve values by step number for hinge event enrichment."""
    points = curve.get("points", [])
    indexed: dict[float, dict[str, Any]] = {}
    if not isinstance(points, list):
        return indexed
    for point in points:
        if not isinstance(point, dict):
            continue
        step = _float_or_none(point.get("step_number", point.get("step")))
        if step is None:
            continue
        indexed[round(step, 8)] = {
            "roof_displacement_m": point.get("control_displacement_m"),
            "base_shear_kn": point.get("base_shear_kn"),
        }
    return indexed


def _summarize_hinge_proxy_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize first hinge-state milestones from proxy event rows."""
    is_envelope_only = _is_envelope_only(events)
    real_step_events = [] if is_envelope_only else events
    ranked = sorted(real_step_events, key=lambda event: (float(event.get("step_number") or 0.0), _level_rank(str(event.get("hinge_state_level", "A-B")))))
    first_plastic = _first_event_at_or_above(ranked, "B-IO")
    first_column = _first_event_at_or_above([event for event in ranked if event.get("element_type") == "column"], "B-IO")
    first_ls = _first_event_at_or_above(ranked, "LS-CP")
    first_cp = _first_event_at_or_above(ranked, "CP-C")
    critical_events = sorted(
        events,
        key=lambda event: float(event.get("demand_capacity_ratio") or 0.0),
        reverse=True,
    )[:8]
    counts: dict[str, int] = {}
    for event in events:
        level = str(event.get("hinge_state_level", ""))
        counts[level] = counts.get(level, 0) + 1
    return {
        "event_count": len(events),
        "is_envelope_only": is_envelope_only,
        "summary_note": "SAP2000 returned Max/Min envelope rows only; first-hinge step milestones are not available." if is_envelope_only else "",
        "state_counts": counts,
        "first_plastic_hinge": first_plastic,
        "first_column_hinge": first_column,
        "first_ls_level": first_ls,
        "first_cp_level": first_cp,
        "critical_events": critical_events,
    }


def _is_envelope_only(events: list[dict[str, Any]]) -> bool:
    """Return true when all hinge events are SAP Max/Min envelope rows."""
    if not events:
        return False
    envelope_steps = {"max", "min", "envelope"}
    return all(
        abs(float(event.get("step_number") or 0.0)) <= 1.0e-9
        and str(event.get("load_step", "")).lower() in envelope_steps
        for event in events
    )


def _first_event_at_or_above(events: list[dict[str, Any]], level: str) -> dict[str, Any] | None:
    """Return the first event at or above a state rank."""
    threshold = _level_rank(level)
    for event in events:
        if _level_rank(str(event.get("hinge_state_level", "A-B"))) >= threshold:
            return event
    return None


def _plastic_rotation_from_ratio(ratio: float, yield_rotation: float) -> float:
    """Estimate plastic rotation from demand/capacity ratio and hinge backbone."""
    theta_y = max(yield_rotation, 1.0e-9)
    if ratio <= 1.0:
        return 0.0
    if ratio <= 1.2:
        return ((ratio - 1.0) / 0.2) * 4.0 * theta_y
    if ratio <= 2.0:
        return 4.0 * theta_y + ((ratio - 1.2) / 0.8) * 4.0 * theta_y
    return 8.0 * theta_y + (ratio - 2.0) * theta_y


def _load_proxy_calibration(output_dir: Path) -> dict[str, Any]:
    """Load optional SAP-validated proxy rotation scale factors."""
    path = output_dir / PROXY_CALIBRATION_RELATIVE_PATH
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "path": str(path), "warning": "Calibration file could not be read."}
    return {"available": True, "path": str(path), **payload}


def _proxy_rotation_scale(calibration: dict[str, Any], element_type: str, hinge_dof: str) -> float:
    """Return the most specific calibrated plastic-rotation multiplier."""
    if not calibration.get("available"):
        return 1.0
    factors = calibration.get("factors", {})
    element = str(element_type or "all").lower()
    dof = str(hinge_dof or "all").upper()
    for key in (f"{element}:{dof}", f"{element}:all"):
        value = factors.get(key, {}) if isinstance(factors, dict) else {}
        scale = _float_or_none(value.get("rotation_scale")) if isinstance(value, dict) else None
        if scale is not None and scale > 0.0:
            return scale
    default = calibration.get("default", {})
    scale = _float_or_none(default.get("rotation_scale")) if isinstance(default, dict) else None
    return scale if scale is not None and scale > 0.0 else 1.0


def _hinge_state_from_rotation(
    plastic_rotation: float,
    yield_rotation: float,
    calibration: dict[str, Any] | None = None,
    element_type: str = "",
    hinge_dof: str = "",
) -> str:
    """Classify an estimated plastic rotation using calibrated or generated IO/LS/CP limits."""
    theta_y = max(yield_rotation, 1.0e-9)
    theta = abs(plastic_rotation)
    levels = ("B-IO", "IO-LS", "LS-CP", "CP-C", "C-D", "D-E", "beyond E")
    default_thresholds = {
        "B-IO": 1.0e-12,
        "IO-LS": 2.0 * theta_y,
        "LS-CP": 4.0 * theta_y,
        "CP-C": 6.0 * theta_y,
        "C-D": 8.0 * theta_y,
        "D-E": 10.0 * theta_y,
        "beyond E": 12.0 * theta_y,
    }
    thresholds: dict[str, float] = {}
    previous = 0.0
    for level in levels:
        value = _proxy_state_boundary(calibration or {}, element_type, hinge_dof, level)
        threshold = value if value is not None and value >= 0.0 else default_thresholds[level]
        if level == "B-IO":
            threshold = max(1.0e-12, threshold)
        else:
            threshold = max(previous + 1.0e-9, threshold)
        thresholds[level] = threshold
        previous = threshold
    if theta < thresholds["B-IO"]:
        return "A-B"
    if theta < thresholds["IO-LS"]:
        return "B-IO"
    if theta < thresholds["LS-CP"]:
        return "IO-LS"
    if theta < thresholds["CP-C"]:
        return "LS-CP"
    if theta < thresholds["C-D"]:
        return "CP-C"
    if theta < thresholds["D-E"]:
        return "C-D"
    if theta < thresholds["beyond E"]:
        return "D-E"
    return "beyond E"


def _proxy_state_boundary(calibration: dict[str, Any], element_type: str, hinge_dof: str, level: str) -> float | None:
    """Return a calibrated absolute plastic-rotation threshold for one state."""
    if not calibration.get("available"):
        return None
    boundaries = calibration.get("state_boundaries", {})
    if not isinstance(boundaries, dict):
        return None
    element = str(element_type or "all").lower()
    dof = str(hinge_dof or "all").upper()
    for key in (f"{element}:{dof}:{level}", f"{element}:all:{level}", f"all:all:{level}"):
        value = boundaries.get(key, {})
        threshold = _float_or_none(value.get("rotation_threshold_rad")) if isinstance(value, dict) else None
        if threshold is not None and threshold >= 0.0:
            return threshold
    return None


def _level_rank(level: str) -> int:
    """Return sortable rank for requested hinge state labels."""
    order = {"A-B": 0, "B-IO": 1, "IO-LS": 2, "LS-CP": 3, "CP-C": 4, "C-D": 5, "D-E": 6, "beyond E": 7}
    return order.get(level, 0)


def _float_or_none(value: object) -> float | None:
    """Parse an optional float."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: object, digits: int) -> float | None:
    """Round optional numeric values for JSON output."""
    parsed = _float_or_none(value)
    return round(parsed, digits) if parsed is not None else None


def _analysis_case_names(cfg: GeneratorConfig) -> list[str]:
    """Return SAP2000 analysis cases that should be solved for this model."""
    cases: list[str] = []
    if cfg.enable_pushover_cases:
        cases.extend(f"PUSHOVER_{direction.upper()}" for direction in cfg.pushover_directions)
    else:
        cases.append("MODAL")
    return cases


def _nearest_roof_control_point(x_coords: list[float], y_coords: list[float], roof_z: float) -> Point3D:
    """Select the existing roof grid joint nearest to the plan centroid."""
    centroid_x = 0.5 * (x_coords[0] + x_coords[-1])
    centroid_y = 0.5 * (y_coords[0] + y_coords[-1])
    x = min(x_coords, key=lambda value: abs(value - centroid_x))
    y = min(y_coords, key=lambda value: abs(value - centroid_y))
    return Point3D(x, y, roof_z)


def _model_file_stem(index: int, candidate: dict[str, Any]) -> str:
    """Build a descriptive SAP2000 model filename stem."""
    col: Section = candidate["column_section"]
    beam: Section = candidate["beam_section"]
    rho_c = round(candidate["rho_col"] * 1000)
    rho_bt = round(candidate["beam_top_ratio_support"] * 1000)
    rho_bb = round(candidate["beam_bottom_ratio_span"] * 1000)
    push = round(candidate["pushover_target_drift_ratio"] * 1000)
    rho_s = round(candidate["slab_rebar_ratio"] * 10000)
    rho_r = round(candidate["raft_rebar_ratio"] * 10000)
    return (
        f"Model_{index:04d}_{candidate['story_count']}Story_"
        f"X{candidate['x_bay_count']}Bay_Y{candidate['y_bay_count']}Bay_"
        f"{candidate['concrete_class']}_Col{col.label_cm}_Beam{beam.label_cm}_"
        f"RhoC{rho_c:03d}_RhoBT{rho_bt:03d}_RhoBB{rho_bb:03d}_PushD{push:03d}_"
        f"Soil{candidate['soil_class']}_Raft{round(candidate['raft_thickness_m'] * 100):03d}_RhoR{rho_r:03d}_"
        f"Slab{round(candidate['slab_thickness_m'] * 100):03d}_RhoS{rho_s:03d}"
        f"{'_WallT' + str(round(candidate['wall_thickness_m'] * 100)).zfill(3) + '_L' + str(round(candidate['wall_length_m'] * 100)).zfill(3) if candidate.get('has_shear_walls') else ''}"
    )


def _metadata_from_candidate(
    index: int,
    candidate: dict[str, Any] | None,
    validation: ValidationResult | None,
    model_path: Path,
    status: str,
    reason: str,
    cfg: GeneratorConfig,
) -> ModelMetadata:
    """Convert candidate and validation objects to a CSV metadata row."""
    if candidate is None:
        return ModelMetadata(
            model_id=f"Model_{index:04d}",
            kat_sayisi=None,
            x_yonu_aciklik_sayisi=None,
            y_yonu_aciklik_sayisi=None,
            aciklik_uzunluklari_x="",
            aciklik_uzunluklari_y="",
            kat_yuksekligi=None,
            toplam_yukseklik=None,
            temel_tipi="",
            perde_var=False,
            perde_sayisi=None,
            perde_kalinligi_m=None,
            perde_uzunlugu_m=None,
            perde_donati_orani=None,
            perde_yerlesimi="",
            x_yonu_perde_sayisi=None,
            y_yonu_perde_sayisi=None,
            perde_lw_tw_kontrolu=None,
            perde_min_kalinlik_kontrolu=None,
            radye_kalinligi_m=None,
            radye_donati_orani=None,
            radye_donati_orani_kontrolu=None,
            doseme_kalinligi_m=None,
            doseme_donati_orani=None,
            doseme_kalinligi_kontrolu=None,
            doseme_donati_orani_kontrolu=None,
            doseme_duzlem_ici_gerilme_kontrolu_placeholder="placeholder_not_run",
            zemin_sinifi="",
            zemin_yatak_katsayisi_kn_m3=None,
            zemin_yayi_modeli="",
            beton_sinifi="",
            beton_fck_mpa=None,
            beton_elastisite_modulu_mpa=None,
            celik_sinifi="",
            kolon_kesitleri="",
            kiris_kesitleri="",
            kolon_donati_orani=None,
            kiris_donati_orani=None,
            kiris_mesnet_ust_donati_orani=None,
            kiris_aciklik_alt_donati_orani=None,
            kolon_min_donati_kontrolu=None,
            kolon_max_donati_kontrolu=None,
            kiris_min_donati_kontrolu=None,
            kiris_max_donati_kontrolu=None,
            guclu_kolon_zayif_kiris_kontrolu=None,
            goreli_kat_otelemesi_kontrolu_placeholder="placeholder_not_run",
            pushover_aktif=False,
            pushover_yonleri="",
            pushover_hedef_otelemesi_orani=None,
            pushover_hedef_deplasman_x_m=None,
            pushover_hedef_deplasman_y_m=None,
            pushover_yuk_dagilimi="",
            pushover_analiz_calistirildi=False,
            plastik_mafsal_aktif=False,
            kolon_mafsal_tipi="",
            kiris_mafsal_tipi="",
            kesme_mafsali_aktif=False,
            plastik_mafsal_atanan_eleman_sayisi=None,
            plastik_mafsal_atama_uyarilari="",
            mafsal_sonuc_okuma_durumu="",
            model_kayit_yolu=str(model_path),
            uretim_durumu=status,
            varsa_elenme_nedeni=reason,
        )

    checks = validation.checks if validation else {}
    concrete = candidate["concrete_class"]
    total_height = round(candidate["story_count"] * candidate["story_height"], 3)
    target_displacement = round(total_height * candidate["pushover_target_drift_ratio"], 4)
    hinge_assignment = candidate.get("plastic_hinge_assignment", {})
    hinge_results = _compact_hinge_results(candidate.get("plastic_hinge_results", {}))
    return ModelMetadata(
        model_id=f"Model_{index:04d}",
        kat_sayisi=candidate["story_count"],
        x_yonu_aciklik_sayisi=candidate["x_bay_count"],
        y_yonu_aciklik_sayisi=candidate["y_bay_count"],
        aciklik_uzunluklari_x=json.dumps(candidate["spans_x"], ensure_ascii=False),
        aciklik_uzunluklari_y=json.dumps(candidate["spans_y"], ensure_ascii=False),
        kat_yuksekligi=candidate["story_height"],
        toplam_yukseklik=total_height,
        temel_tipi="radye" if cfg.enable_raft_foundation else "ankastre_taban",
        perde_var=bool(candidate.get("has_shear_walls")),
        perde_sayisi=candidate.get("wall_count") if candidate.get("has_shear_walls") else 0,
        perde_kalinligi_m=candidate["wall_thickness_m"] if candidate.get("has_shear_walls") else None,
        perde_uzunlugu_m=candidate["wall_length_m"] if candidate.get("has_shear_walls") else None,
        perde_donati_orani=candidate["wall_rebar_ratio"] if candidate.get("has_shear_walls") else None,
        perde_yerlesimi=candidate["wall_placement"] if candidate.get("has_shear_walls") else "none",
        x_yonu_perde_sayisi=candidate.get("wall_count_x") if candidate.get("has_shear_walls") else 0,
        y_yonu_perde_sayisi=candidate.get("wall_count_y") if candidate.get("has_shear_walls") else 0,
        perde_lw_tw_kontrolu=checks.get("perde_lw_tw"),
        perde_min_kalinlik_kontrolu=checks.get("perde_kalinligi"),
        radye_kalinligi_m=candidate["raft_thickness_m"] if cfg.enable_raft_foundation else None,
        radye_donati_orani=candidate["raft_rebar_ratio"] if cfg.enable_raft_foundation else None,
        radye_donati_orani_kontrolu=checks.get("radye_donati_orani"),
        doseme_kalinligi_m=candidate["slab_thickness_m"] if cfg.enable_floor_slabs else None,
        doseme_donati_orani=candidate["slab_rebar_ratio"] if cfg.enable_floor_slabs else None,
        doseme_kalinligi_kontrolu=checks.get("doseme_kalinligi"),
        doseme_donati_orani_kontrolu=checks.get("doseme_donati_orani"),
        doseme_duzlem_ici_gerilme_kontrolu_placeholder="placeholder_not_run",
        zemin_sinifi=candidate["soil_class"],
        zemin_yatak_katsayisi_kn_m3=candidate["subgrade_modulus_kn_m3"],
        zemin_yayi_modeli="Winkler vertical point springs" if cfg.enable_raft_foundation and cfg.assign_soil_vertical_springs else "not_assigned",
        beton_sinifi=concrete,
        beton_fck_mpa=CONCRETE_FCK_MPA[concrete],
        beton_elastisite_modulu_mpa=round(concrete_elastic_modulus_mpa(concrete), 1),
        celik_sinifi=candidate["steel_class"],
        kolon_kesitleri=f"{candidate['column_section'].label_cm} cm",
        kiris_kesitleri=f"{candidate['beam_section'].label_cm} cm",
        kolon_donati_orani=candidate["rho_col"],
        kiris_donati_orani=candidate["rho_beam"],
        kiris_mesnet_ust_donati_orani=candidate["beam_top_ratio_support"],
        kiris_aciklik_alt_donati_orani=candidate["beam_bottom_ratio_span"],
        kolon_min_donati_kontrolu=checks.get("kolon_min_donati"),
        kolon_max_donati_kontrolu=checks.get("kolon_max_donati"),
        kiris_min_donati_kontrolu=checks.get("kiris_min_donati"),
        kiris_max_donati_kontrolu=checks.get("kiris_max_donati"),
        guclu_kolon_zayif_kiris_kontrolu=checks.get("guclu_kolon_zayif_kiris"),
        goreli_kat_otelemesi_kontrolu_placeholder="analysis_not_run",
        pushover_aktif=cfg.enable_pushover_cases,
        pushover_yonleri=",".join(cfg.pushover_directions),
        pushover_hedef_otelemesi_orani=candidate["pushover_target_drift_ratio"],
        pushover_hedef_deplasman_x_m=target_displacement if "X" in cfg.pushover_directions else None,
        pushover_hedef_deplasman_y_m=target_displacement if "Y" in cfg.pushover_directions else None,
        pushover_yuk_dagilimi=cfg.pushover_load_distribution,
        pushover_analiz_calistirildi=cfg.run_analysis_after_save and status == "success",
        plastik_mafsal_aktif=cfg.enable_plastic_hinges,
        kolon_mafsal_tipi=cfg.column_hinge_property if cfg.assign_column_pmm_hinges else "",
        kiris_mafsal_tipi=cfg.beam_hinge_property if cfg.assign_beam_m3_hinges else "",
        kesme_mafsali_aktif=cfg.assign_shear_hinges,
        plastik_mafsal_atanan_eleman_sayisi=hinge_assignment.get("assigned_count") if hinge_assignment else None,
        plastik_mafsal_atama_uyarilari="; ".join((hinge_assignment.get("warnings") or [])[:5]) if hinge_assignment else "",
        mafsal_sonuc_okuma_durumu=json.dumps(_hinge_results_csv_summary(hinge_results), ensure_ascii=False) if hinge_results else "not_read",
        model_kayit_yolu=str(model_path),
        uretim_durumu=status,
        varsa_elenme_nedeni=reason or (validation.rejection_text if validation else ""),
    )


def _write_json_metadata(path: Path, candidate: dict[str, Any], validation: ValidationResult, model_path: Path, status: str, reason: str, cfg: GeneratorConfig) -> None:
    """Write per-model JSON metadata next to the SAP2000 model."""
    payload = {
        "candidate": _json_safe_candidate(candidate),
        "validation": {
            "is_valid": validation.is_valid,
            "checks": validation.checks,
            "rejection_reasons": validation.rejection_reasons,
        },
        "model_path": str(model_path),
        "status": status,
        "reason": reason,
        "pushover": {
            "enabled": cfg.enable_pushover_cases,
            "directions": list(cfg.pushover_directions),
            "target_drift_ratio": candidate["pushover_target_drift_ratio"],
            "target_displacement_m": round(candidate["story_count"] * candidate["story_height"] * candidate["pushover_target_drift_ratio"], 4),
            "base_shear_proxy_kn": cfg.pushover_base_shear_proxy_kn,
            "load_distribution": cfg.pushover_load_distribution,
            "analysis_run": cfg.run_analysis_after_save and status == "success",
            "analysis_status": candidate.get("analysis_status", {}),
            "results": _compact_pushover_results(
                {
                    **(candidate.get("pushover_results", {}) if isinstance(candidate.get("pushover_results"), dict) else {}),
                    "story_drifts": candidate.get("story_drift_results", {}),
                }
            ),
        },
        "plastic_hinges": {
            "enabled": cfg.enable_plastic_hinges,
            "require_assignment": cfg.require_plastic_hinge_assignment,
            "column_pmm_enabled": cfg.assign_column_pmm_hinges,
            "beam_m3_enabled": cfg.assign_beam_m3_hinges,
            "shear_hinges_enabled": cfg.assign_shear_hinges,
            "column_hinge_property": cfg.column_hinge_property,
            "beam_hinge_property": cfg.beam_hinge_property,
            "shear_hinge_v2_property": cfg.shear_hinge_v2_property,
            "shear_hinge_v3_property": cfg.shear_hinge_v3_property,
            "relative_distances": list(cfg.hinge_relative_distances),
            "assignment": candidate.get("plastic_hinge_assignment", {}),
            "results": _compact_hinge_results(candidate.get("plastic_hinge_results", {})),
            "exact_ui_export": candidate.get("exact_hinge_ui_export", {}),
            "note": "Plastic hinge assignment is written through SAP2000 .$2k text tables because this SAP2000 v22 OAPI type library does not expose a hinge assignment writer.",
        },
        "reinforcement": {
            "column_rebar_ratio": candidate["rho_col"],
            "beam_top_ratio_support": candidate["beam_top_ratio_support"],
            "beam_bottom_ratio_span": candidate["beam_bottom_ratio_span"],
            "beam_envelope_ratio": candidate["rho_beam"],
            "raft_rebar_ratio": candidate["raft_rebar_ratio"],
            "slab_rebar_ratio": candidate["slab_rebar_ratio"],
            "wall_rebar_ratio": candidate["wall_rebar_ratio"] if candidate.get("has_shear_walls") else None,
            "note": "Ratios are preliminary automation parameters, not detailed bar layouts.",
        },
        "foundation": {
            "type": "raft" if cfg.enable_raft_foundation else "fixed_base",
            "raft_thickness_m": candidate["raft_thickness_m"],
            "raft_rebar_ratio": candidate["raft_rebar_ratio"],
            "raft_rebar_ratio_check": validation.checks.get("radye_donati_orani"),
            "soil_class": candidate["soil_class"],
            "subgrade_modulus_kn_m3": candidate["subgrade_modulus_kn_m3"],
            "spring_model": "Winkler vertical point springs" if cfg.assign_soil_vertical_springs else "not_assigned",
            "edge_offset_m": cfg.foundation_edge_offset_m,
        },
        "floor_slabs": {
            "enabled": cfg.enable_floor_slabs,
            "modeling": "horizontal shell panels between beam grid lines" if cfg.enable_floor_slabs else "not modeled",
            "panel_count": candidate.get("slab_panel_count", 0),
            "thickness_m": candidate["slab_thickness_m"],
            "rebar_ratio_each_direction": candidate["slab_rebar_ratio"],
            "thickness_check": validation.checks.get("doseme_kalinligi"),
            "rebar_ratio_check": validation.checks.get("doseme_donati_orani"),
            "in_plane_stress_check": "placeholder_not_run",
            "note": "The same preliminary reinforcement ratio is assigned as metadata for both orthogonal slab directions; final bar detailing is outside this generator.",
        },
        "shear_walls": {
            "enabled": bool(candidate.get("has_shear_walls")),
            "count": candidate.get("wall_count", 0),
            "x_direction_wall_count": candidate.get("wall_count_x", 0),
            "y_direction_wall_count": candidate.get("wall_count_y", 0),
            "thickness_m": candidate["wall_thickness_m"],
            "length_m": candidate["wall_length_m"],
            "rebar_ratio": candidate["wall_rebar_ratio"],
            "placement": candidate["wall_placement"],
            "min_thickness_check": validation.checks.get("perde_kalinligi"),
            "lw_tw_check": validation.checks.get("perde_lw_tw"),
            "rebar_ratio_check": validation.checks.get("perde_donati_orani"),
            "modeling": "vertical shell areas between selected column grid lines" if candidate.get("has_shear_walls") else "not modeled",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_model_preview_svg(path: Path, candidate: dict[str, Any]) -> None:
    """Write a simple plan/elevation SVG preview for dashboard monitoring."""
    width = 960
    height = 610
    margin = 52
    plan_x = margin
    plan_y = 96
    plan_w = 360
    plan_h = 320
    elev_x = 540
    elev_y = 96
    elev_w = 330
    elev_h = 320
    x_coords = _scaled_coords(candidate["spans_x"], plan_x, plan_w)
    y_coords = _scaled_coords(candidate["spans_y"], plan_y, plan_h)
    story_count = candidate["story_count"]
    story_height = candidate["story_height"]
    total_height = story_count * story_height
    col: Section = candidate["column_section"]
    beam: Section = candidate["beam_section"]

    plan_lines = []
    for x in x_coords:
        plan_lines.append(f'<line x1="{x:.1f}" y1="{plan_y}" x2="{x:.1f}" y2="{plan_y + plan_h}" class="grid"/>')
    for y in y_coords:
        plan_lines.append(f'<line x1="{plan_x}" y1="{y:.1f}" x2="{plan_x + plan_w}" y2="{y:.1f}" class="grid"/>')
    for x in x_coords:
        for y in y_coords:
            plan_lines.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" class="joint"/>')

    elev_lines = []
    for i in range(candidate["x_bay_count"] + 1):
        x = elev_x + elev_w * i / candidate["x_bay_count"]
        elev_lines.append(f'<line x1="{x:.1f}" y1="{elev_y}" x2="{x:.1f}" y2="{elev_y + elev_h}" class="grid"/>')
    for story in range(story_count + 1):
        y = elev_y + elev_h - elev_h * story / story_count
        elev_lines.append(f'<line x1="{elev_x}" y1="{y:.1f}" x2="{elev_x + elev_w}" y2="{y:.1f}" class="grid"/>')

    highlights = _preview_highlights(candidate)
    highlight_lines = [
        _preview_highlight_svg(item, candidate, plan_x, plan_y, plan_w, plan_h, elev_x, elev_y, elev_w, elev_h)
        for item in highlights
    ]
    hinge_source_note = str(candidate.get("preview_hinge_source_note", "") or "").strip()
    if highlights:
        source_note_svg = (
            f'<text x="54" y="558" class="meta">{_svg_escape(hinge_source_note)}</text>'
            if hinge_source_note
            else ""
        )
        legend = """
  <g class="legend">
    <line x1="54" y1="518" x2="92" y2="518" class="highlight x-critical"/>
    <text x="106" y="525">X kritik</text>
    <line x1="245" y1="518" x2="283" y2="518" class="highlight x-first"/>
    <text x="297" y="525">X ilk</text>
    <line x1="430" y1="518" x2="468" y2="518" class="highlight y-critical"/>
    <text x="482" y="525">Y kritik</text>
    <line x1="620" y1="518" x2="658" y2="518" class="highlight y-first"/>
    <text x="672" y="525">Y ilk</text>
  </g>""" + source_note_svg
    else:
        legend = '<text x="54" y="535" class="meta">No hinge summary</text>'

    foundation_text = f"Soil {candidate['soil_class']} | ks={candidate['subgrade_modulus_kn_m3']:.0f} kN/m3 | Raft={candidate['raft_thickness_m']:.2f} m rhoR={candidate['raft_rebar_ratio']:.4f} | Slab={candidate['slab_thickness_m']:.2f} m rhoS={candidate['slab_rebar_ratio']:.4f}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .bg {{ fill: #f7f9fb; }}
    .panel {{ fill: #ffffff; stroke: #d9dee7; stroke-width: 1.4; }}
    .grid {{ stroke: #176b87; stroke-width: 3; stroke-linecap: round; }}
    .joint {{ fill: #2d7d46; }}
    .push {{ stroke: #a44d22; stroke-width: 4; fill: none; stroke-linecap: round; stroke-linejoin: round; }}
    .title {{ font: 700 28px Segoe UI, Arial, sans-serif; fill: #172033; }}
    .label {{ font: 700 18px Segoe UI, Arial, sans-serif; fill: #172033; }}
    .meta {{ font: 15px Segoe UI, Arial, sans-serif; fill: #687386; }}
    .badge {{ fill: #eaf4f6; stroke: #c9e2e8; }}
    .highlight {{ stroke-width: 8; stroke-linecap: round; fill: none; }}
    .marker {{ stroke: #ffffff; stroke-width: 3; }}
    .x-critical {{ stroke: #dc2626; fill: #dc2626; }}
    .x-first {{ stroke: #f59e0b; fill: #f59e0b; stroke-dasharray: 10 7; }}
    .y-critical {{ stroke: #2563eb; fill: #2563eb; }}
    .y-first {{ stroke: #38bdf8; fill: #38bdf8; stroke-dasharray: 10 7; }}
    .legend text {{ font: 700 19px Segoe UI, Arial, sans-serif; fill: #172033; }}
  </style>
  <rect width="100%" height="100%" class="bg"/>
  <text x="{margin}" y="48" class="title">{_svg_escape(_model_preview_title(candidate))}</text>
  <rect x="{margin}" y="70" width="856" height="420" rx="8" class="panel"/>
  <text x="{plan_x}" y="86" class="label">Plan</text>
  <text x="{elev_x}" y="86" class="label">Elevasyon</text>
  {''.join(plan_lines)}
  {''.join(elev_lines)}
  {''.join(highlight_lines)}
  <path d="M {elev_x + elev_w + 34:.1f} {elev_y + elev_h:.1f} L {elev_x + elev_w + 34:.1f} {elev_y + 54:.1f} L {elev_x + elev_w + 22:.1f} {elev_y + 74:.1f} M {elev_x + elev_w + 34:.1f} {elev_y + 54:.1f} L {elev_x + elev_w + 46:.1f} {elev_y + 74:.1f}" class="push"/>
  {legend}
  <rect x="{margin}" y="568" width="856" height="34" rx="6" class="badge"/>
  <text x="{margin + 18}" y="591" class="meta">
    {_svg_escape(f"{story_count} stories, total H={total_height:.2f} m | X{candidate['x_bay_count']} Y{candidate['y_bay_count']} | {candidate['concrete_class']} | Column {col.label_cm} cm | Beam {beam.label_cm} cm | rhoC={candidate['rho_col']:.4f}, top={candidate['beam_top_ratio_support']:.4f}, bottom={candidate['beam_bottom_ratio_span']:.4f} | Push={candidate['pushover_target_drift_ratio']:.3f} | {foundation_text}")}
  </text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _preview_highlights(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return X/Y critical and first hinge elements for the saved SVG preview."""
    results = candidate.get("plastic_hinge_results", {})
    summaries = results.get("summary_by_direction", {}) if isinstance(results, dict) else {}
    if not isinstance(summaries, dict):
        return []
    x_coords = _cumulative_coordinates(candidate["spans_x"])
    y_coords = _cumulative_coordinates(candidate["spans_y"])
    story_count = int(candidate["story_count"])
    story_height = float(candidate["story_height"])
    items: list[dict[str, Any]] = []
    for direction, summary in summaries.items():
        if not isinstance(summary, dict):
            continue
        direction_key = "Y" if str(direction).upper() == "Y" else "X"
        critical_events = summary.get("critical_events", [])
        critical = _ranked_hinge_events(critical_events, 1)[0] if isinstance(critical_events, list) and critical_events else None
        critical_segment = _preview_element_segment(
            str(critical.get("element_name", "")) if isinstance(critical, dict) else "",
            x_coords,
            y_coords,
            story_count,
            story_height,
        )
        if critical_segment:
            items.append({**critical_segment, "role": "critical", "direction": direction_key})
        first = summary.get("first_plastic_hinge", {})
        first_segment = _preview_element_segment(
            str(first.get("element_name", "")) if isinstance(first, dict) else "",
            x_coords,
            y_coords,
            story_count,
            story_height,
        )
        if first_segment:
            items.append({**first_segment, "role": "first", "direction": direction_key})
    return items


def _preview_highlight_svg(
    item: dict[str, Any],
    candidate: dict[str, Any],
    plan_x: float,
    plan_y: float,
    plan_w: float,
    plan_h: float,
    elev_x: float,
    elev_y: float,
    elev_w: float,
    elev_h: float,
) -> str:
    """Draw a highlighted element in plan and elevation."""
    x_coords = _cumulative_coordinates(candidate["spans_x"])
    y_coords = _cumulative_coordinates(candidate["spans_y"])
    total_x = max(x_coords[-1], 1.0)
    total_y = max(y_coords[-1], 1.0)
    total_z = max(float(candidate["story_count"]) * float(candidate["story_height"]), 1.0)

    def px(value: float) -> float:
        return plan_x + (value / total_x) * plan_w

    def py(value: float) -> float:
        return plan_y + (value / total_y) * plan_h

    def ex(value: float) -> float:
        return elev_x + (value / total_x) * elev_w

    def ey(value: float) -> float:
        return elev_y + elev_h - (value / total_z) * elev_h

    cls = f"{item['direction'].lower()}-{'critical' if item['role'] == 'critical' else 'first'}"
    a = item["a"]
    b = item["b"]
    if item["type"] == "column":
        radius = 15 if item["role"] == "critical" else 11
        plan_offsets = {
            ("X", "critical"): (14, 0),
            ("X", "first"): (-14, 0),
            ("Y", "critical"): (0, 14),
            ("Y", "first"): (0, -14),
        }
        elev_offsets = {
            ("X", "critical"): 8,
            ("X", "first"): -8,
            ("Y", "critical"): 18,
            ("Y", "first"): -18,
        }
        plan_dx, plan_dy = plan_offsets.get((str(item["direction"]), str(item["role"])), (0, 0))
        column_offset = elev_offsets.get((str(item["direction"]), str(item["role"])), 0)
        return (
            f'<circle cx="{px(a[0]) + plan_dx:.1f}" cy="{py(a[1]) + plan_dy:.1f}" r="{radius}" class="marker {cls}"/>'
            f'<line x1="{ex(a[0]) + column_offset:.1f}" y1="{ey(a[2]):.1f}" '
            f'x2="{ex(b[0]) + column_offset:.1f}" y2="{ey(b[2]):.1f}" class="highlight {cls}"/>'
        )
    plan_line = (
        f'<line x1="{px(a[0]):.1f}" y1="{py(a[1]):.1f}" '
        f'x2="{px(b[0]):.1f}" y2="{py(b[1]):.1f}" class="highlight {cls}"/>'
    )
    uses_y = abs(a[0] - b[0]) < 0.001
    elev_a = (a[1] / max(total_y, 1.0)) * total_x if uses_y else a[0]
    elev_b = (b[1] / max(total_y, 1.0)) * total_x if uses_y else b[0]
    elev_offset = 5 if item["direction"] == "Y" and item["role"] == "first" else -5 if item["direction"] == "X" and item["role"] == "first" else 0
    return (
        plan_line
        + f'<line x1="{ex(elev_a):.1f}" y1="{ey(a[2]) + elev_offset:.1f}" '
        + f'x2="{ex(elev_b):.1f}" y2="{ey(b[2]) + elev_offset:.1f}" class="highlight {cls}"/>'
    )


def _preview_element_segment(
    name: str,
    x_coords: list[float],
    y_coords: list[float],
    story_count: int,
    story_height: float,
) -> dict[str, Any] | None:
    """Parse generated column/beam names into element coordinates."""
    text = str(name or "").strip()
    match = re.match(r"^C_(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$", text, re.IGNORECASE)
    if match:
        story = int(match.group(1))
        x = _closest_preview_coord(x_coords, float(match.group(2)))
        y = _closest_preview_coord(y_coords, float(match.group(3)))
        if 1 <= story <= story_count and x is not None and y is not None:
            return {"name": text, "type": "column", "a": [x, y, (story - 1) * story_height], "b": [x, y, story * story_height]}
    match = re.match(r"^BX_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$", text, re.IGNORECASE)
    if match:
        story = int(match.group(1))
        index = int(match.group(2))
        y = _closest_preview_coord(y_coords, float(match.group(3)))
        if 1 <= story <= story_count and 0 <= index < len(x_coords) - 1 and y is not None:
            z = story * story_height
            return {"name": text, "type": "beam", "a": [x_coords[index], y, z], "b": [x_coords[index + 1], y, z]}
    match = re.match(r"^BY_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$", text, re.IGNORECASE)
    if match:
        story = int(match.group(1))
        index = int(match.group(2))
        x = _closest_preview_coord(x_coords, float(match.group(3)))
        if 1 <= story <= story_count and 0 <= index < len(y_coords) - 1 and x is not None:
            z = story * story_height
            return {"name": text, "type": "beam", "a": [x, y_coords[index], z], "b": [x, y_coords[index + 1], z]}
    return None


def _closest_preview_coord(coords: list[float], value: float) -> float | None:
    """Return closest grid coordinate if the parsed element name is on-grid."""
    best = min(coords, key=lambda coord: abs(coord - value))
    return best if abs(best - value) <= 0.15 else None


def _scaled_coords(spans: list[float], origin: float, size: float) -> list[float]:
    """Scale span coordinates into an SVG drawing interval."""
    total = sum(spans)
    coords = [origin]
    running = 0.0
    for span in spans:
        running += span
        coords.append(origin + size * running / total)
    return coords


def _model_preview_title(candidate: dict[str, Any]) -> str:
    """Return a compact title for SVG preview."""
    return f"{candidate['story_count']}Story X{candidate['x_bay_count']} Y{candidate['y_bay_count']} {candidate['concrete_class']}"


def _svg_escape(value: str) -> str:
    """Escape text for SVG output."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_safe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Convert dataclass section values into JSON-safe dictionaries."""
    safe = dict(candidate)
    for key in ("pushover_results", "plastic_hinge_results", "hinge_inventory", "story_drift_results"):
        safe.pop(key, None)
    safe["column_section"] = asdict(candidate["column_section"])
    safe["beam_section"] = asdict(candidate["beam_section"])
    return safe


def _compact_pushover_results(results: object) -> dict[str, Any]:
    """Keep pushover result metadata compact enough for large batch generation."""
    if not isinstance(results, dict):
        return {}
    compact = dict(results)
    curves = results.get("curves", {})
    if isinstance(curves, dict):
        compact_curves: dict[str, Any] = {}
        for direction, curve in curves.items():
            if not isinstance(curve, dict):
                continue
            clean_curve = dict(curve)
            points = curve.get("points", [])
            if isinstance(points, list) and len(points) > MAX_STORED_CURVE_POINTS_PER_DIRECTION:
                clean_curve["points"] = _sample_result_points(points, MAX_STORED_CURVE_POINTS_PER_DIRECTION)
                clean_curve["stored_point_count"] = len(clean_curve["points"])
                clean_curve["original_point_count"] = len(points)
            compact_curves[str(direction)] = clean_curve
        compact["curves"] = compact_curves
    return compact


def _compact_hinge_results(results: object) -> dict[str, Any]:
    """Keep hinge summaries and only a ranked subset of proxy event rows."""
    if not isinstance(results, dict):
        return {}
    compact = dict(results)
    events_by_direction = results.get("proxy_events_by_direction", {})
    if isinstance(events_by_direction, dict):
        compact_events: dict[str, Any] = {}
        for direction, payload in events_by_direction.items():
            if not isinstance(payload, dict):
                continue
            clean_payload = dict(payload)
            events = payload.get("events", [])
            if isinstance(events, list):
                clean_payload["events"] = _ranked_hinge_events(events, MAX_STORED_HINGE_EVENTS_PER_DIRECTION)
                clean_payload["stored_event_count"] = len(clean_payload["events"])
                clean_payload["original_event_count"] = len(events)
            compact_events[str(direction)] = clean_payload
        compact["proxy_events_by_direction"] = compact_events
    return compact


def _write_proxy_hinge_events_csv(output_dir: Path, model_stem: str, results: object) -> None:
    """Persist full proxy hinge events for later SAP-exact calibration."""
    if not isinstance(results, dict):
        return
    events_by_direction = results.get("proxy_events_by_direction", {})
    if not isinstance(events_by_direction, dict):
        return
    rows: list[dict[str, Any]] = []
    for direction, payload in events_by_direction.items():
        if not isinstance(payload, dict):
            continue
        case_name = str(payload.get("case") or f"PUSHOVER_{str(direction).upper()}")
        events = payload.get("events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict):
                rows.append(
                    {
                        "model": model_stem,
                        "direction": str(direction).upper(),
                        "case": case_name,
                        **event,
                    }
                )
    proxy_dir = output_dir / "hinge_validation" / "proxy_events"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    path = proxy_dir / f"{model_stem}_proxy_hinges.csv"
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _hinge_results_csv_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny hinge summary suitable for the flat metadata CSV."""
    summary = results.get("summary_by_direction", {}) if isinstance(results, dict) else {}
    events = results.get("proxy_events_by_direction", {}) if isinstance(results, dict) else {}
    return {
        "available": results.get("available"),
        "proxy_note": results.get("proxy_note", ""),
        "directions": {
            direction: {
                "event_count": value.get("event_count"),
                "stored_event_count": events.get(direction, {}).get("stored_event_count") if isinstance(events.get(direction), dict) else None,
                "state_counts": value.get("state_counts", {}),
                "first_plastic_hinge": _event_identity(value.get("first_plastic_hinge")),
                "first_ls_level": _event_identity(value.get("first_ls_level")),
                "first_cp_level": _event_identity(value.get("first_cp_level")),
                "critical_event": _event_identity((value.get("critical_events") or [None])[0] if isinstance(value.get("critical_events"), list) else None),
            }
            for direction, value in summary.items()
            if isinstance(value, dict)
        },
    }


def _event_identity(event: object) -> dict[str, Any] | None:
    """Return compact event identity fields."""
    if not isinstance(event, dict):
        return None
    keys = ("step_number", "load_step", "element_name", "element_type", "hinge_location", "hinge_state_level", "plastic_rotation_rad", "demand_capacity_ratio")
    return {key: event.get(key) for key in keys if key in event}


def _ranked_hinge_events(events: list[Any], limit: int) -> list[dict[str, Any]]:
    """Return a compact ranked sample of hinge events."""
    rows = [event for event in events if isinstance(event, dict)]
    ranked = sorted(
        rows,
        key=lambda event: (
            _level_rank(str(event.get("hinge_state_level", "A-B"))),
            float(event.get("plastic_rotation_rad") or 0.0),
            float(event.get("demand_capacity_ratio") or 0.0),
            float(event.get("step_number") or 0.0),
        ),
        reverse=True,
    )
    return [_event_identity(event) | {
        "roof_displacement_m": event.get("roof_displacement_m"),
        "base_shear_kn": event.get("base_shear_kn"),
        "moment_kn_m": event.get("moment_kn_m"),
        "axial_force_kn": event.get("axial_force_kn"),
        "shear_v2_kn": event.get("shear_v2_kn"),
        "shear_v3_kn": event.get("shear_v3_kn"),
    } for event in ranked[:limit] if _event_identity(event)]


def _sample_result_points(points: list[Any], limit: int) -> list[Any]:
    """Downsample result points while keeping first and last values."""
    if len(points) <= limit:
        return points
    step = max(1, len(points) // max(1, limit - 1))
    sampled = points[::step][: limit - 1]
    if points[-1] not in sampled:
        sampled.append(points[-1])
    return sampled


def _write_metadata_csv(path: Path, rows: list[ModelMetadata]) -> None:
    """Rewrite metadata CSV after each iteration so completed work is preserved."""
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
