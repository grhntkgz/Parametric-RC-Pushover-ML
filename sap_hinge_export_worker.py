"""Isolated batch worker for exact SAP2000 frame-hinge table exports.

Each .sdb model is opened in its own disposable SAP2000 process. Cached results
are cleared, X/Y pushover cases are rerun, and the step-by-step hinge table is
exported immediately. SAP2000 v22 can terminate while closing the table viewer;
process isolation keeps that desktop instability away from model generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from sap_hinge_ui_export import PUSHOVER_OUTPUT_CASES, export_active_hinge_result_tables, export_model_hinge_result_tables


ProgressCallback = Callable[[str, dict[str, Any]], None]
StopCallback = Callable[[], bool]
SUMMARY_RELATIVE_PATH = Path("hinge_validation") / "export_worker_summary.json"


def export_models_in_isolated_workers(
    output_dir: Path,
    model_paths: Iterable[Path] | None = None,
    timeout_seconds: float = 1800.0,
    max_attempts_per_model: int = 1,
    min_rows_per_direction: int = 100,
    progress_callback: ProgressCallback | None = None,
    stop_callback: StopCallback | None = None,
) -> dict[str, Any]:
    """Export exact hinge CSV files with retryable disposable workers per model."""
    output_dir = output_dir.resolve()
    paths = sorted(model_paths or output_dir.glob("*.sdb"))
    records: list[dict[str, Any]] = []
    started_at = time.time()
    cancelled = False
    for index, model_path in enumerate(paths, start=1):
        if stop_callback is not None and stop_callback():
            cancelled = True
            break
        model_path = model_path.resolve()
        _emit(progress_callback, "exact_hinge_export_started", {"index": index, "total": len(paths), "model_path": str(model_path)})
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts_per_model + 1):
            if stop_callback is not None and stop_callback():
                cancelled = True
                break
            _emit(
                progress_callback,
                "exact_hinge_export_attempt_started",
                {"index": index, "total": len(paths), "attempt": attempt, "max_attempts": max_attempts_per_model, "model_path": str(model_path)},
            )
            record = _run_one_model_subprocess(model_path, output_dir, timeout_seconds, min_rows_per_direction, stop_callback)
            attempts.append(_compact_attempt_record(attempt, record))
            if record["status"] == "cancelled":
                cancelled = True
                break
            if record["status"] == "success":
                break
            if attempt < max_attempts_per_model:
                _emit(
                    progress_callback,
                    "exact_hinge_export_retrying",
                    {
                        "index": index,
                        "total": len(paths),
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "max_attempts": max_attempts_per_model,
                        "model_path": str(model_path),
                        "message": record.get("message", ""),
                    },
                )
        if cancelled:
            break
        record = {**record, "attempt_count": len(attempts), "attempts": attempts}
        records.append(record)
        _emit(
            progress_callback,
            "exact_hinge_export_finished",
            {
                "index": index,
                "total": len(paths),
                "model_path": str(model_path),
                "status": record["status"],
                "message": record.get("message", ""),
            },
        )
    summary = {
        "output_dir": str(output_dir),
        "model_count": len(paths),
        "min_rows_per_direction": max(1, min_rows_per_direction),
        "success_count": sum(record["status"] == "success" for record in records),
        "failed_count": sum(record["status"] != "success" for record in records),
        "cancelled": cancelled,
        "duration_seconds": round(time.time() - started_at, 3),
        "records": records,
    }
    summary_path = output_dir / SUMMARY_RELATIVE_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _compact_attempt_record(attempt: int, record: dict[str, Any]) -> dict[str, Any]:
    """Keep retry history useful without recursively embedding full worker records."""
    return {
        "attempt": attempt,
        "status": record.get("status", "failed"),
        "duration_seconds": record.get("duration_seconds", 0.0),
        "message": record.get("message", ""),
        "exported_files": record.get("exported_files", []),
    }


def export_one_model(model_path: Path, output_dir: Path, wait_seconds: float, min_rows_per_direction: int = 100) -> dict[str, Any]:
    """Export one analyzed model and return a JSON-serializable worker record."""
    started_at = time.time()
    try:
        direction_exports: dict[str, dict[str, Any]] = {}
        exported_files: list[str] = []
        for case_name in PUSHOVER_OUTPUT_CASES:
            result = export_model_hinge_result_tables(
                model_path.resolve(),
                output_dir.resolve(),
                wait_seconds,
                output_cases=(case_name,),
                export_stem=f"{model_path.stem}__{case_name.lower()}",
            )
            direction_exports[case_name] = result
            exported_files.extend(result.get("exported_files", []))
        validation = _validate_exported_hinge_files(exported_files, min_rows_per_direction=max(1, min_rows_per_direction))
        status = "success" if all(result.get("available") for result in direction_exports.values()) and validation["is_valid"] else "failed"
        message = "Exact SAP hinge tables exported separately for X and Y pushover cases."
        if not validation["is_valid"]:
            message = validation["message"]
        return {
            "model_path": str(model_path.resolve()),
            "model_stem": model_path.stem,
            "status": status,
            "duration_seconds": round(time.time() - started_at, 3),
            "message": message,
            "export_validation": validation,
            "available": status == "success",
            "direction_exports": direction_exports,
            "exported_files": exported_files,
            "message": message,
        }
    except Exception as exc:  # noqa: BLE001 - one failed desktop export must not stop the batch.
        return {
            "model_path": str(model_path.resolve()),
            "model_stem": model_path.stem,
            "status": "failed",
            "duration_seconds": round(time.time() - started_at, 3),
            "message": str(exc),
            "exported_files": [],
        }


def export_active_model_session(
    model_path: Path,
    output_dir: Path,
    process_id: int | None,
    wait_seconds: float,
    min_rows_per_direction: int = 100,
    output_cases: tuple[str, ...] = PUSHOVER_OUTPUT_CASES,
) -> dict[str, Any]:
    """Export exact hinge tables from an already solved SAP2000 session."""
    started_at = time.time()
    oapi_record: dict[str, Any] = {
        "status": "skipped",
        "message": "DatabaseTables OAPI active export is disabled by default because SAP2000 v22 can terminate abnormally on this installation.",
    }
    if os.environ.get("SAP2000_ENABLE_DATABASE_TABLES_EXPORT") == "1":
        oapi_record = _export_active_model_session_oapi(
            model_path,
            output_dir,
            min_rows_per_direction,
            output_cases,
            started_at,
        )
        if oapi_record.get("status") == "success":
            return oapi_record
    try:
        direction_exports: dict[str, dict[str, Any]] = {}
        exported_files: list[str] = []
        for case_name in output_cases:
            result = export_active_hinge_result_tables(
                output_dir.resolve(),
                f"{model_path.stem}__{case_name.lower()}",
                wait_seconds,
                process_id,
                output_cases=(case_name,),
            )
            direction_exports[case_name] = result
            exported_files.extend(result.get("exported_files", []))
        validation = _validate_exported_hinge_files(exported_files, min_rows_per_direction=max(1, min_rows_per_direction))
        status = "success" if all(result.get("available") for result in direction_exports.values()) and validation["is_valid"] else "failed"
        message = "Exact SAP hinge tables exported from the first solved SAP session."
        if not validation["is_valid"]:
            message = validation["message"]
        return {
            "model_path": str(model_path.resolve()),
            "model_stem": model_path.stem,
            "status": status,
            "duration_seconds": round(time.time() - started_at, 3),
            "message": message,
            "export_validation": validation,
            "available": status == "success",
            "direction_exports": direction_exports,
            "exported_files": exported_files,
            "attempt_count": 1,
            "inline_first_solve_export": True,
            "active_session_export_subprocess": True,
        }
    except Exception as exc:  # noqa: BLE001 - one failed desktop export must not stop the batch.
        return {
            "model_path": str(model_path.resolve()),
            "model_stem": model_path.stem,
            "status": "failed",
            "duration_seconds": round(time.time() - started_at, 3),
            "message": f"Exact SAP hinge export failed in first solved session: {exc}. OAPI attempt: {oapi_record.get('message', '')}",
            "exported_files": [],
            "attempt_count": 1,
            "inline_first_solve_export": True,
            "active_session_export_subprocess": True,
            "oapi_export_attempt": oapi_record,
        }


def _export_active_model_session_oapi(
    model_path: Path,
    output_dir: Path,
    min_rows_per_direction: int,
    output_cases: tuple[str, ...],
    started_at: float,
) -> dict[str, Any]:
    """Try direct SAP2000 DatabaseTables export without desktop UI automation."""
    exported_files: list[str] = []
    direction_exports: dict[str, dict[str, Any]] = {}
    try:
        import comtypes.client
    except ImportError as exc:
        return _failed_active_oapi_record(model_path, started_at, f"comtypes is not available: {exc}", exported_files, direction_exports)
    try:
        sap_object = comtypes.client.GetActiveObject("CSI.SAP2000.API.SapObject")
        sap_model = sap_object.SapModel
        db = sap_model.DatabaseTables
        raw_dir = output_dir / "hinge_validation" / "raw_exports"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for case_name in output_cases:
            _check_sap_table_ret(db.SetLoadCasesSelectedForDisplay([case_name]), f"SetLoadCasesSelectedForDisplay({case_name})")
            _set_step_by_step_output_options(db)
            table_key = _find_hinge_states_table_key(db)
            target = raw_dir / f"{model_path.stem}__{case_name.lower()}__frame_hinge_states.csv"
            _export_display_table_csv(db, table_key, target)
            exported_files.append(str(target))
            direction_exports[case_name] = {
                "available": target.exists(),
                "message": "Exact SAP hinge table exported through DatabaseTables OAPI.",
                "output_cases": [case_name],
                "selected_tables": [table_key],
                "exported_files": [str(target)],
                "oapi_database_tables": True,
            }
        validation = _validate_exported_hinge_files(exported_files, min_rows_per_direction=max(1, min_rows_per_direction))
        status = "success" if validation["is_valid"] else "failed"
        return {
            "model_path": str(model_path.resolve()),
            "model_stem": model_path.stem,
            "status": status,
            "duration_seconds": round(time.time() - started_at, 3),
            "message": "Exact SAP hinge tables exported from the first solved SAP session through DatabaseTables OAPI." if status == "success" else validation["message"],
            "export_validation": validation,
            "available": status == "success",
            "direction_exports": direction_exports,
            "exported_files": exported_files,
            "attempt_count": 1,
            "inline_first_solve_export": True,
            "active_session_export_subprocess": True,
            "oapi_database_tables": True,
        }
    except Exception as exc:  # noqa: BLE001 - fallback to UI automation may still work.
        return _failed_active_oapi_record(model_path, started_at, str(exc), exported_files, direction_exports)


def _failed_active_oapi_record(
    model_path: Path,
    started_at: float,
    message: str,
    exported_files: list[str],
    direction_exports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return one failed direct-OAPI export attempt record."""
    return {
        "model_path": str(model_path.resolve()),
        "model_stem": model_path.stem,
        "status": "failed",
        "duration_seconds": round(time.time() - started_at, 3),
        "message": message,
        "available": False,
        "direction_exports": direction_exports,
        "exported_files": exported_files,
        "attempt_count": 1,
        "inline_first_solve_export": True,
        "active_session_export_subprocess": True,
        "oapi_database_tables": True,
    }


def _check_sap_table_ret(result: Any, context: str) -> None:
    """Raise when a DatabaseTables call returns a nonzero SAP code."""
    ret = result[-1] if isinstance(result, (list, tuple)) and result else result
    if int(ret) != 0:
        raise RuntimeError(f"SAP DatabaseTables {context} failed: {ret}")


def _set_step_by_step_output_options(db: Any) -> None:
    """Best-effort request for step-by-step nonlinear static table rows."""
    candidates = (
        (False, 0.0, 0.0, 0.0, False, 0, 0, False, 0, 0, 2, 2, 2, 2, 2),
        (False, 0.0, 0.0, 0.0, False, 0, 0, False, 0, 0, 1, 1, 1, 1, 1),
        (False, 0.0, 0.0, 0.0, False, 0, 0, False, 0, 0, 0, 0, 0, 0, 0),
    )
    last_error: Exception | None = None
    for args in candidates:
        try:
            _check_sap_table_ret(db.SetOutputOptionsForDisplay(*args), "SetOutputOptionsForDisplay")
            return
        except Exception as exc:  # noqa: BLE001 - SAP version enum values vary.
            last_error = exc
    if last_error is not None:
        raise last_error


def _find_hinge_states_table_key(db: Any) -> str:
    """Find the installed SAP table key for Frame Hinge States."""
    table_rows = []
    for getter_name in ("GetAvailableTables", "GetAllTables"):
        try:
            result = getattr(db, getter_name)()
            _check_sap_table_ret(result, getter_name)
            count = int(result[0] or 0)
            keys = list(result[1] or ())
            names = list(result[2] or ())
            for index in range(min(count, len(keys), len(names))):
                table_rows.append((str(keys[index]), str(names[index])))
        except Exception:
            continue
    for key, name in table_rows:
        text = f"{key} {name}".lower()
        if "frame" in text and "hinge" in text and "state" in text:
            return key
    for candidate in ("Frame Hinge States", "Frame Hinge State", "Frame Hinge Assignments"):
        if any(candidate.lower() in f"{key} {name}".lower() for key, name in table_rows):
            return candidate
    return "Frame Hinge States"


def _export_display_table_csv(db: Any, table_key: str, target: Path) -> None:
    """Export one DatabaseTables display table to a semicolon CSV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    attempts = (
        lambda: db.GetTableForDisplayCSVFile(table_key, "", str(target), ";"),
        lambda: db.GetTableForDisplayCSVFile(table_key, "", str(target)),
        lambda: db.GetTableForDisplayCSVFile(table_key, [], "", str(target), ";"),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            result = attempt()
            _check_sap_table_ret(result, f"GetTableForDisplayCSVFile({table_key})")
            if target.exists():
                return
        except Exception as exc:  # noqa: BLE001 - comtypes optional/out arg dispatch varies.
            last_error = exc
    raise RuntimeError(f"SAP DatabaseTables could not export {table_key} to {target}: {last_error}")


def _run_one_model_subprocess(
    model_path: Path,
    output_dir: Path,
    timeout_seconds: float,
    min_rows_per_direction: int,
    stop_callback: StopCallback | None = None,
) -> dict[str, Any]:
    """Run one worker subprocess with hard-timeout and user-cancel support."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "one",
        "--model",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--wait-seconds",
        str(min(timeout_seconds, 300.0)),
        "--min-rows-per-direction",
        str(max(1, min_rows_per_direction)),
    ]
    started_at = time.time()
    existing_excel_processes = _windows_process_ids("EXCEL.EXE")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        while True:
            if stop_callback is not None and stop_callback():
                _terminate_process_tree(process.pid)
                record = {
                    "model_path": str(model_path),
                    "model_stem": model_path.stem,
                    "status": "cancelled",
                    "duration_seconds": round(time.time() - started_at, 3),
                    "message": "Exact SAP hinge export cancelled by user.",
                    "exported_files": [],
                }
                break
            remaining = timeout_seconds - (time.time() - started_at)
            if remaining <= 0.0:
                _terminate_process_tree(process.pid)
                record = {
                    "model_path": str(model_path),
                    "model_stem": model_path.stem,
                    "status": "failed",
                    "duration_seconds": round(time.time() - started_at, 3),
                    "message": f"Exact SAP hinge export timed out after {timeout_seconds:.0f} seconds.",
                    "exported_files": [],
                }
                break
            try:
                stdout, stderr = process.communicate(timeout=min(0.5, remaining))
            except subprocess.TimeoutExpired:
                continue
            record = _last_json_line(stdout)
            if record is None:
                record = {
                    "model_path": str(model_path),
                    "model_stem": model_path.stem,
                    "status": "failed",
                    "duration_seconds": round(time.time() - started_at, 3),
                    "message": stderr.strip() or f"Exact hinge worker exited with code {process.returncode}.",
                    "exported_files": [],
                }
            break
    finally:
        for excel_process_id in _windows_process_ids("EXCEL.EXE") - existing_excel_processes:
            subprocess.run(
                ["taskkill", "/PID", str(excel_process_id), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
    return record


def _terminate_process_tree(process_id: int) -> None:
    """Terminate one disposable worker and every SAP/Excel descendant."""
    subprocess.run(
        ["taskkill", "/PID", str(process_id), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )


def _last_json_line(output: str) -> dict[str, Any] | None:
    """Read the final JSON object emitted by a one-model subprocess."""
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _validate_exported_hinge_files(exported_files: Iterable[str], min_rows_per_direction: int = 100) -> dict[str, Any]:
    """Require real SAP Frame Hinge States rows in both pushover directions."""
    min_rows_per_direction = max(1, int(min_rows_per_direction))
    files: list[dict[str, Any]] = []
    for value in exported_files:
        path = Path(value)
        row_count = 0
        table_name = ""
        has_required_columns = False
        if path.exists():
            rows = _read_hinge_csv_rows(path)
            table_name = rows[0][0].strip() if rows and rows[0] else ""
            headers = {header.strip() for header in rows[1]} if len(rows) > 1 else set()
            has_required_columns = {"Frame", "OutputCase", "StepNum", "AssignHinge", "HingeState"}.issubset(headers)
            row_count = sum(any(cell.strip() for cell in row) for row in rows[2:])
        files.append(
            {
                "path": str(path),
                "table_name": table_name,
                "row_count": row_count,
                "min_required_rows": min_rows_per_direction,
                "has_required_columns": has_required_columns,
            }
        )
    is_valid = bool(files) and all(
        file["table_name"].lower() == "table:  frame hinge states"
        and file["has_required_columns"]
        and file["row_count"] >= min_rows_per_direction
        for file in files
    )
    message = f"Exact SAP Frame Hinge States table contains at least {min_rows_per_direction} rows in each direction."
    if not is_valid:
        row_summary = ", ".join(
            f"{_direction_from_export_path(file['path'])}={file['row_count']}" for file in files
        ) or "no files"
        message = (
            "SAP Frame Hinge States export is empty, incomplete, or below the minimum row threshold "
            f"({row_summary}; required >= {min_rows_per_direction} per direction)."
        )
    return {
        "is_valid": is_valid,
        "message": message,
        "min_required_rows_per_direction": min_rows_per_direction,
        "files": files,
    }


def _read_hinge_csv_rows(path: Path) -> list[list[str]]:
    """Read a SAP CSV table using the delimiter that exposes the expected headers."""
    best_rows: list[list[str]] = []
    best_score = -1
    for delimiter in (";", ",", "\t"):
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
        headers = {header.strip() for header in rows[1]} if len(rows) > 1 else set()
        score = len({"Frame", "OutputCase", "StepNum", "AssignHinge", "HingeState"} & headers)
        if score > best_score:
            best_rows = rows
            best_score = score
    return best_rows


def _direction_from_export_path(path: str) -> str:
    """Infer X/Y direction from the worker export file name."""
    lowered = Path(path).name.lower()
    if "__pushover_x__" in lowered:
        return "X"
    if "__pushover_y__" in lowered:
        return "Y"
    return "?"


def _windows_process_ids(image_name: str) -> set[int]:
    """Return Windows process IDs for one executable image name."""
    completed = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    process_ids: set[int] = set()
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) < 2 or row[0].lower() != image_name.lower():
            continue
        try:
            process_ids.add(int(row[1]))
        except ValueError:
            continue
    return process_ids


def _emit(callback: ProgressCallback | None, event: str, payload: dict[str, Any]) -> None:
    """Emit optional progress event."""
    if callback is not None:
        callback(event, payload)


def build_parser() -> argparse.ArgumentParser:
    """Build worker CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("one", help="Export one analyzed .sdb model.")
    one.add_argument("--model", type=Path, required=True)
    one.add_argument("--output-dir", type=Path, required=True)
    one.add_argument("--wait-seconds", type=float, default=120.0)
    one.add_argument("--min-rows-per-direction", type=int, default=100)

    active = subparsers.add_parser("active", help="Export from an already solved active SAP2000 process.")
    active.add_argument("--model", type=Path, required=True)
    active.add_argument("--output-dir", type=Path, required=True)
    active.add_argument("--process-id", type=int, default=0)
    active.add_argument("--wait-seconds", type=float, default=300.0)
    active.add_argument("--min-rows-per-direction", type=int, default=100)
    active.add_argument("--output-cases", nargs="+", default=list(PUSHOVER_OUTPUT_CASES))

    batch = subparsers.add_parser("batch", help="Export all root-level .sdb models in one output directory.")
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--timeout-seconds", type=float, default=1800.0)
    batch.add_argument("--max-attempts-per-model", type=int, default=3)
    batch.add_argument("--min-rows-per-direction", type=int, default=100)
    return parser


def main() -> None:
    """Run worker CLI."""
    args = build_parser().parse_args()
    if args.command == "one":
        result = export_one_model(args.model, args.output_dir, args.wait_seconds, max(1, args.min_rows_per_direction))
    elif args.command == "active":
        process_id = args.process_id if args.process_id > 0 else None
        result = export_active_model_session(
            args.model,
            args.output_dir,
            process_id,
            args.wait_seconds,
            max(1, args.min_rows_per_direction),
            tuple(str(case) for case in args.output_cases),
        )
    else:
        result = export_models_in_isolated_workers(
            args.output_dir,
            timeout_seconds=args.timeout_seconds,
            max_attempts_per_model=max(1, args.max_attempts_per_model),
            min_rows_per_direction=max(1, args.min_rows_per_direction),
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
