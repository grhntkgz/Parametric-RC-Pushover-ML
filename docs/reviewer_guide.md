# Reviewer Guide

This repository is organized so that the post-processing and machine-learning interpretation workflow can be inspected without rerunning SAP2000.

## What Can Be Reviewed Without SAP2000

- Dashboard source code and static interface files.
- Reduced example metadata under `examples/sample_metadata/`.
- Representative SVG previews under `examples/sample_previews/`.
- Example capacity-curve derived metrics under `examples/sample_capacity_curve_analysis/`.
- Behavior ML source code for Random Forest, XGBoost, and LightGBM based discriminative parameter analysis.
- Capacity-curve metric extraction script.

## What Requires SAP2000

The full automated model-generation and nonlinear pushover workflow requires:

- Windows
- SAP2000 installation
- SAP2000 OAPI/COM access
- Sufficient local disk space for generated `.sdb`, `.s2k`, `.msh`, `.out`, metadata, and preview files

The repository includes the SAP2000 automation code, but the complete production-scale numerical workflow is not expected to be rerun by every reviewer.

## Quick Reproducibility Check

For a lightweight non-SAP2000 check, run:

```powershell
python scripts/smoke_test.py
```

The smoke test checks that the example dataset is present, verifies that the main Python modules compile, and regenerates capacity-curve derived metrics from the reduced example metadata in a temporary output folder. This check does not require SAP2000 and does not train the full XGBoost/LightGBM Behavior ML models.

For full Behavior ML comparisons, install the packages in `requirements.txt`.

## Dashboard Review

Start the dashboard:

```powershell
python dashboard/server.py
```

Then open:

```text
http://127.0.0.1:8765
```

Use the example metadata, logs, and preview folders if SAP2000 is not available.

## Publication-Scale Data

The full dataset can be large because it may include thousands of metadata files, preview SVGs, logs, and SAP2000 model files. Large generated outputs are intentionally excluded from Git and should be archived separately for manuscript review or long-term reproducibility.
