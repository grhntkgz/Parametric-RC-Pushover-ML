# Parametric Reinforced Concrete Pushover Framework

This repository contains a Python-based research framework for automated parametric reinforced concrete building generation, nonlinear static pushover analysis, structural response extraction, and classification-based machine learning interpretation.

The full model-generation workflow uses SAP2000 OAPI/COM on Windows. The dashboard and machine-learning analysis modules can also be reviewed without SAP2000 by using the reduced example metadata and preview files included in `examples/`.

## Main Capabilities

- Controlled randomized generation of reinforced concrete building parameters.
- Preliminary TS 500 / TBDY 2018-oriented filtering rules for engineering plausibility.
- SAP2000 OAPI/COM model creation for 3D reinforced concrete frame and frame-wall systems.
- Raft foundation and simplified Winkler spring soil representation.
- X and Y direction nonlinear static pushover analysis setup.
- Plastic hinge assignment and proxy-based hinge state interpretation.
- Extraction of capacity-curve, drift, base-shear, plastic-rotation, first-hinge, critical-element, and damage-state indicators.
- Dashboard-based review of completed analyses and model representative SVG views.
- Behavior ML module using Random Forest, XGBoost, and LightGBM for classification-based discriminative parameter analysis.

## Repository Layout

```text
.
├── config.py
├── design_rules.py
├── sap_api.py
├── model_generator.py
├── main.py
├── dashboard/
│   ├── server.py
│   └── static/
├── behavior_ml.py
├── ml_model.py
├── examples/
│   ├── sample_metadata/
│   ├── sample_previews/
│   └── sample_logs/
├── figures/
├── docs/
└── tests/
```

## Installation

Use Python 3.10+ on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`comtypes` is required only for the SAP2000 automation workflow. XGBoost and LightGBM are optional but recommended for the Behavior ML comparisons.

## Run the Dashboard

```powershell
python dashboard/server.py
```

Open:

```text
http://127.0.0.1:8765
```

The dashboard can read previously generated metadata, logs, and preview SVG files. If SAP2000 is not available, use the example files in `examples/` to review the analysis and ML workflow.

## Full SAP2000 Workflow

The full generation workflow requires:

- Windows
- SAP2000 installed
- SAP2000 OAPI/COM access
- A compatible SAP2000 executable path configured in the dashboard or `config.py`

Run:

```powershell
python main.py
```

Generated SAP2000 models, logs, and large result files are intentionally excluded from Git by default. They should be archived separately for publication-scale datasets.

## Example Data

The `examples/` directory contains a reduced demonstration set:

- metadata JSON files
- representative model SVG previews
- sample log files

This sample set is intended for reviewers who want to inspect the dashboard and machine-learning analysis workflow without running SAP2000.

## Research Scope

This framework is intended for synthetic parametric analysis and data-supported interpretation of nonlinear behavior trends. It is not a replacement for final reinforced concrete design, detailed code compliance checks, or project-specific engineering verification.

The included design checks and hinge-state evaluations should be interpreted as preliminary filtering and comparative research tools.

## Citation / Data Availability

For manuscript review, the repository can provide the source code and a reduced demonstration dataset. The complete analysis metadata and large SAP2000 result files should be archived separately, for example on Zenodo, OSF, Figshare, or an institutional repository.
