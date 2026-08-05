"""Run lightweight repository checks without SAP2000.

The smoke test is intended for reviewers and maintainers who want to verify
that the non-SAP2000 parts of the repository are usable. It checks the reduced
example dataset, compiles key Python modules, and regenerates capacity-curve
derived metrics from the sample metadata in a temporary directory.
"""

from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_METADATA = ROOT / "examples" / "sample_metadata"
SAMPLE_PREVIEWS = ROOT / "examples" / "sample_previews"
SAMPLE_LOGS = ROOT / "examples" / "sample_logs"

MODULES_TO_COMPILE = [
    "config.py",
    "design_rules.py",
    "fema440.py",
    "behavior_ml.py",
    "capacity_curve_analysis.py",
    "ml_model.py",
    "model_generator.py",
    "sap_api.py",
    "dashboard/server.py",
]


def main() -> int:
    """Execute smoke checks and return a process status code."""

    print("Repository smoke test")
    check_example_dataset()
    compile_modules()
    run_capacity_curve_example()
    print("Smoke test completed successfully.")
    return 0


def check_example_dataset() -> None:
    """Verify that the reduced example dataset exists."""

    metadata_count = count_files(SAMPLE_METADATA, "*_metadata.json")
    preview_count = count_files(SAMPLE_PREVIEWS, "*_preview.svg")
    log_count = count_files(SAMPLE_LOGS, "*.log")
    if metadata_count == 0 or preview_count == 0:
        raise RuntimeError("Example metadata and preview files are required for dashboard review.")
    print(f"Example dataset: {metadata_count} metadata, {preview_count} previews, {log_count} logs")


def compile_modules() -> None:
    """Compile key Python modules to catch syntax errors."""

    for relative_path in MODULES_TO_COMPILE:
        py_compile.compile(str(ROOT / relative_path), doraise=True)
    print(f"Compiled {len(MODULES_TO_COMPILE)} Python modules")


def run_capacity_curve_example() -> None:
    """Regenerate capacity-curve metrics from the sample metadata."""

    temp_root = Path(tempfile.mkdtemp(prefix="rc_pushover_smoke_"))
    try:
        output_dir = temp_root / "capacity_curve_analysis"
        command = [
            sys.executable,
            str(ROOT / "capacity_curve_analysis.py"),
            "--metadata-dir",
            str(SAMPLE_METADATA),
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        expected = output_dir / "capacity_curve_report.md"
        if not expected.exists():
            raise RuntimeError("Capacity-curve smoke output was not created.")
        print(f"Capacity-curve example output: {expected}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def count_files(folder: Path, pattern: str) -> int:
    """Count files matching a glob pattern."""

    return len(list(folder.glob(pattern))) if folder.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
