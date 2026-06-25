"""Small dependency-free ML helpers for SAP2000 generated model results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from som_model import DEFAULT_X_COLUMNS, X_COLUMNS, assign_bmu, build_som_rows, numeric_or_ordinal_value, preprocess_som_input, train_som_weights


NUMERIC_FEATURES = (
    "story_count",
    "bay_count",
    "x_bay_count",
    "y_bay_count",
    "avg_span",
    "avg_span_x",
    "avg_span_y",
    "max_span_x",
    "max_span_y",
    "story_height",
    "total_height",
    "concrete_fck",
    "steel_fy",
    "column_width",
    "column_depth",
    "beam_width",
    "beam_depth",
    "rho_col",
    "rho_beam_top",
    "rho_beam_bottom",
    "slab_thickness",
    "slab_rebar_ratio",
    "raft_thickness",
    "raft_rebar_ratio",
    "wall_count",
    "wall_area",
    "wall_thickness",
    "wall_length",
    "wall_rebar_ratio",
    "subgrade_modulus",
    "target_drift",
)

CATEGORICAL_FEATURES = ("direction", "concrete_class", "steel_class", "soil_class", "has_shear_walls")
BASE_NUMERIC_FEATURES = NUMERIC_FEATURES
BASE_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
SOM_FEATURE_PREFIX = "som_"
STAGE_FEATURE_PREFIX = "stage_"
TARGETS = (
    "first_hinge_type",
    "critical_element_type",
    "critical_state",
    "has_lscp",
    "first_column_available",
    "first_ls_available",
    "first_cp_available",
)
MODEL_FILE = "ml_model.json"
TRAINING_CACHE_FILE = "ml_training_cache.jsonl"
ALGORITHMS = ("knn", "random_forest", "xgboost", "lightgbm")
REGRESSION_TARGETS = (
    "peak_base_shear",
    "max_rotation",
    "max_story_drift_ratio",
    "max_displacement",
    "fema_ductility_mu",
    "fema_beta_eff_percent",
    "fema_target_displacement_proxy",
)


def model_file_for(algorithm: str) -> str:
    """Return JSON model file name for an algorithm."""
    return f"ml_model_{normalize_algorithm(algorithm)}.json"


def external_model_file_for(algorithm: str) -> str:
    """Return external model file name for an algorithm."""
    return f"ml_model_{normalize_algorithm(algorithm)}.joblib"


def resolve_ml_x_columns(output_dir: Path, requested: list[str] | None) -> list[str]:
    """Resolve the shared SOM/ML X-column selection and reject unknown fields."""
    if requested is not None:
        unknown = [str(key) for key in requested if str(key) not in X_COLUMNS]
        if unknown:
            raise ValueError(f"Bilinmeyen ML X parametreleri: {', '.join(unknown)}")
        selected = list(dict.fromkeys(str(key) for key in requested))
        if not selected:
            raise ValueError("Makine ogrenmesi icin en az bir X parametresi secmelisin.")
        return selected

    for path in (output_dir / "som_model.json", output_dir / model_file_for(active_algorithm(output_dir))):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected = [str(key) for key in payload.get("selected_x_columns", []) if str(key) in X_COLUMNS]
        if selected:
            return list(dict.fromkeys(selected))
    return list(DEFAULT_X_COLUMNS)


@dataclass(frozen=True)
class Dataset:
    """Training rows and labels extracted from metadata."""

    rows: list[dict[str, Any]]


def train_ml_model(
    output_dir: Path,
    algorithm: str = "knn",
    random_seed: int = 42,
    preserve_existing: bool = False,
    use_som_features: bool = False,
    selected_x_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train dependency-free classifiers from generated metadata."""
    algorithm = normalize_algorithm(algorithm)
    preserved_snapshot: dict[str, Any] = {}
    if preserve_existing and (output_dir / model_file_for(algorithm)).exists():
        preserved_snapshot = save_ml_snapshot(output_dir, algorithm)

    dataset = build_dataset(output_dir)
    if len(dataset.rows) < 20:
        raise ValueError("Makine ogrenmesi icin en az 20 yon-sonucu gerekir.")
    dataset = Dataset([strip_som_features(row) for row in dataset.rows])
    save_training_cache(output_dir, dataset.rows)
    selected_x = resolve_ml_x_columns(output_dir, selected_x_columns)
    som_feature_model: dict[str, Any] = {}
    if use_som_features:
        dataset = Dataset(add_som_derived_features(dataset.rows, random_seed, selected_x))
        som_feature_model = dataset.rows[0].get("_som_feature_model", {}) if dataset.rows else {}
        for row in dataset.rows:
            row.pop("_som_feature_model", None)

    base_feature_spec = build_feature_spec(dataset.rows, selected_x)
    base_encoded = [(encode_features(row, base_feature_spec), row) for row in dataset.rows]
    rng = random.Random(random_seed)
    rng.shuffle(base_encoded)

    split = max(1, int(len(base_encoded) * 0.8))
    base_train_rows = base_encoded[:split]
    base_test_rows = base_encoded[split:] or base_encoded[:]

    regression_models: dict[str, Any] = {}
    regression_metrics: dict[str, Any] = {}
    for target in REGRESSION_TARGETS:
        regression_model = train_numeric_knn_target(base_train_rows, target)
        regression_models[target] = regression_model
        actual = [numeric_target(row, target) for _, row in base_test_rows]
        predicted = [predict_numeric_knn(regression_model, base_train_rows, features, target) for features, _ in base_test_rows]
        paired = [(a, p) for a, p in zip(actual, predicted) if a is not None and p is not None]
        regression_metrics[target] = regression_score_metrics(paired)

    staged_train = add_out_of_fold_stage_features(base_train_rows, regression_models)
    staged_test = add_stage_features(base_test_rows, regression_models, base_train_rows)
    feature_spec = build_feature_spec([row for _, row in staged_train], selected_x)
    train_rows = [(encode_features(row, feature_spec), row) for _, row in staged_train]
    test_rows = [(encode_features(row, feature_spec), row) for _, row in staged_test]
    encoded = train_rows + test_rows

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    external_models: dict[str, Any] = {}
    for target in TARGETS:
        if algorithm == "knn":
            target_model = train_knn_target(train_rows, target)
        elif algorithm == "random_forest":
            target_model = train_random_forest_target(train_rows, target, random_seed + len(models))
        elif algorithm in ("xgboost", "lightgbm"):
            target_model, external_model = train_external_target(train_rows, target, algorithm, random_seed + len(models))
            if external_model is not None:
                external_models[target] = external_model
        else:
            raise_optional_algorithm_error(algorithm)
        predictions = [predict_target(target_model, train_rows, features, target) for features, _ in test_rows]
        actual = [row[target] for _, row in test_rows]
        metrics[target] = classification_metrics(actual, predictions)
        if "k" in target_model:
            metrics[target]["best_k"] = target_model["k"]
        if "validation_accuracy" in target_model:
            metrics[target]["validation_accuracy"] = target_model["validation_accuracy"]
        if "tree_count" in target_model:
            metrics[target]["tree_count"] = target_model["tree_count"]
        if "n_estimators" in target_model:
            metrics[target]["n_estimators"] = target_model["n_estimators"]
        models[target] = target_model

    payload = {
        "version": 1,
        "algorithm": algorithm,
        "use_som_features": use_som_features,
        "selected_x_columns": selected_x,
        "som_feature_model": som_feature_model,
        "base_feature_spec": base_feature_spec,
        "regression_models": regression_models,
        "regression_metrics": regression_metrics,
        "feature_engineering_note": (
            "SOM destekli ozellikler yalnizca X parametreleriyle egitilen SOM konumu, BMU mesafesi, "
            "genel ortalamaya gore z-skorlari ve X parametre kombinasyonlarindan uretilir; Y sonuc dagilimi feature olarak kullanilmaz."
            if use_som_features
            else "Ham X parametreleri kullanildi; SOM destekli ozellikler kapali."
        ),
        "external_model_file": external_model_file_for(algorithm) if external_models else "",
        "trained_at_source": str((output_dir / "models_metadata.csv").resolve()),
        "sample_count": len(dataset.rows),
        "feature_spec": feature_spec,
        "models": models,
        "metrics": metrics,
        "training_rows": [
            {
                "features": features,
                "targets": {target: row[target] for target in TARGETS},
                "events": {
                    "first": row.get("first_event", {}),
                    "critical": row.get("critical_event", {}),
                    "first_column": row.get("first_column_event", {}),
                    "first_ls": row.get("first_ls_event", {}),
                    "first_cp": row.get("first_cp_event", {}),
                },
            }
            for features, row in encoded
        ],
        "regression_training_rows": [
            {
                "features": features,
                "targets": {target: numeric_target(row, target) for target in REGRESSION_TARGETS},
            }
            for features, row in base_train_rows
        ],
        "feature_importance_proxy": feature_importance_proxy([row for _, row in encoded], feature_spec),
    }
    if external_models:
        dump_external_models(output_dir / external_model_file_for(algorithm), external_models)
    model_path = output_dir / model_file_for(algorithm)
    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / MODEL_FILE).write_text(json.dumps({"active_algorithm": algorithm, "active_model_file": model_path.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "preserved_snapshot": preserved_snapshot,
        **{key: value for key, value in payload.items() if key != "training_rows"},
    }


def predict_ml(output_dir: Path, params: dict[str, Any], algorithm: str | None = None) -> dict[str, Any]:
    """Predict first hinge and critical element classes for user parameters."""
    algorithm = normalize_algorithm(algorithm or active_algorithm(output_dir))
    model_path = output_dir / model_file_for(algorithm)
    if not model_path.exists():
        raise ValueError(f"Once {algorithm} modelini egitin.")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    row = normalize_prediction_params(params)
    if model.get("use_som_features"):
        row = add_som_features_to_prediction(row, model.get("som_feature_model", {}))
    regression_predictions: dict[str, float | None] = {}
    regression_rows = [
        (item.get("features", []), item.get("targets", {}))
        for item in model.get("regression_training_rows", [])
        if isinstance(item, dict)
    ]
    base_spec = model.get("base_feature_spec") or model.get("feature_spec", {})
    base_features = encode_features(row, base_spec)
    for target, regression_model in model.get("regression_models", {}).items():
        value = predict_numeric_knn(regression_model, regression_rows, base_features, target)
        regression_predictions[target] = value
        row[f"{STAGE_FEATURE_PREFIX}{target}"] = value if value is not None else 0.0
    features = encode_features(row, model["feature_spec"])
    train_items = [(item["features"], item) for item in model["training_rows"]]
    train_rows = [(features, item["targets"]) for features, item in train_items]
    external_models = load_external_models(output_dir, model)
    predictions: dict[str, Any] = {}
    for target, target_model in model["models"].items():
        if target in external_models:
            target_model = dict(target_model)
            target_model["_external_model"] = external_models[target]
        prediction, probabilities, neighbors = predict_target_with_details(target_model, train_rows, features, target)
        predictions[target] = {
            "prediction": prediction,
            "probabilities": probabilities,
            "nearest_neighbors": neighbors,
        }
    predicted_events = {
        "first": estimate_event_location(
            train_items,
            features,
            "first",
            "first_hinge_type",
            predictions["first_hinge_type"]["prediction"],
            model_location_k(model["models"]["first_hinge_type"]),
            row,
        ),
        "critical": estimate_event_location(
            train_items,
            features,
            "critical",
            "critical_element_type",
            predictions["critical_element_type"]["prediction"],
            model_location_k(model["models"]["critical_element_type"]),
            row,
        ),
        "first_column": estimate_event_location(
            train_items,
            features,
            "first_column",
            "first_column_available",
            predictions["first_column_available"]["prediction"],
            model_location_k(model["models"]["first_column_available"]),
            row,
        ),
        "first_ls": estimate_event_location(
            train_items,
            features,
            "first_ls",
            "first_ls_available",
            predictions["first_ls_available"]["prediction"],
            model_location_k(model["models"]["first_ls_available"]),
            row,
        ),
        "first_cp": estimate_event_location(
            train_items,
            features,
            "first_cp",
            "first_cp_available",
            predictions["first_cp_available"]["prediction"],
            model_location_k(model["models"]["first_cp_available"]),
            row,
        ),
    }
    return {
        "input": row,
        "predictions": predictions,
        "regression_predictions": regression_predictions,
        "predicted_events": predicted_events,
        "model_summary": {
            "sample_count": model["sample_count"],
            "metrics": model["metrics"],
            "regression_metrics": model.get("regression_metrics", {}),
            "selected_x_columns": model.get("selected_x_columns", []),
            "use_som_features": bool(model.get("use_som_features")),
            "feature_engineering_note": model.get("feature_engineering_note", ""),
        },
    }


def ml_status(output_dir: Path) -> dict[str, Any]:
    """Return current ML model status."""
    has_metadata = (output_dir / "models_metadata.csv").exists() or any(output_dir.glob("run_*/models_metadata.csv"))
    dataset_count = len(build_dataset(output_dir).rows) if has_metadata else 0
    models = {}
    for algorithm in ALGORITHMS:
        path = output_dir / model_file_for(algorithm)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            models[algorithm] = {
                "available": True,
                "model_path": str(path),
                "algorithm": payload.get("algorithm"),
                "sample_count": payload.get("sample_count"),
                "use_som_features": bool(payload.get("use_som_features")),
                "metrics": payload.get("metrics", {}),
                "regression_metrics": payload.get("regression_metrics", {}),
                "feature_importance_proxy": payload.get("feature_importance_proxy", []),
                "selected_x_columns": payload.get("selected_x_columns", []),
            }
        else:
            models[algorithm] = {"available": False, "model_path": str(path), "algorithm": algorithm}
    active = active_algorithm(output_dir)
    model_path = output_dir / model_file_for(active)
    if not model_path.exists():
        return {"available": False, "dataset_count": dataset_count, "model_path": str(model_path), "active_algorithm": active, "models": models}
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "dataset_count": dataset_count,
        "model_path": str(model_path),
        "active_algorithm": active,
        "algorithm": payload.get("algorithm"),
        "sample_count": payload.get("sample_count"),
        "use_som_features": bool(payload.get("use_som_features")),
        "feature_engineering_note": payload.get("feature_engineering_note", ""),
        "metrics": payload.get("metrics", {}),
        "regression_metrics": payload.get("regression_metrics", {}),
        "feature_importance_proxy": payload.get("feature_importance_proxy", []),
        "selected_x_columns": payload.get("selected_x_columns", []),
        "models": models,
    }


def save_ml_snapshot(output_dir: Path, algorithm: str | None = None) -> dict[str, Any]:
    """Save the current trained model files as one overwriteable copy per algorithm."""
    algorithm = normalize_algorithm(algorithm or active_algorithm(output_dir))
    model_path = output_dir / model_file_for(algorithm)
    if not model_path.exists():
        raise ValueError("Kaydedilecek egitilmis ML modeli yok.")
    snapshot_dir = output_dir / "ml_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for source in (model_path, output_dir / external_model_file_for(algorithm)):
        if source.exists():
            target = snapshot_dir / source.name
            target.write_bytes(source.read_bytes())
            saved.append(str(target))
    return {"saved": saved, "snapshot_dir": str(snapshot_dir)}


def reset_ml_model(output_dir: Path, algorithm: str | None = None) -> dict[str, Any]:
    """Delete active ML model files without touching SAP2000 artifacts."""
    algorithm = normalize_algorithm(algorithm or active_algorithm(output_dir))
    deleted: list[str] = []
    for source in (output_dir / model_file_for(algorithm), output_dir / external_model_file_for(algorithm), output_dir / MODEL_FILE):
        if source.exists():
            source.unlink()
            deleted.append(str(source))
    return {"deleted": deleted, "available": False}


def active_algorithm(output_dir: Path) -> str:
    """Return active algorithm from pointer file, falling back to xgboost/knn."""
    pointer = output_dir / MODEL_FILE
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            value = normalize_algorithm(str(data.get("active_algorithm") or data.get("algorithm") or "knn"))
            if value in ALGORITHMS:
                return value
        except json.JSONDecodeError:
            pass
    for algorithm in ("xgboost", "lightgbm", "random_forest", "knn"):
        if (output_dir / model_file_for(algorithm)).exists():
            return algorithm
    return "knn"


def build_dataset(output_dir: Path) -> Dataset:
    """Build one training sample per model direction from metadata CSV and ML cache."""
    rows: list[dict[str, Any]] = []
    som_index: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for som_row in build_som_rows(output_dir):
            key = (str(som_row.get("model_name") or ""), str(som_row.get("direction") or ""))
            som_index[key] = som_row
    except (OSError, ValueError, TypeError):
        som_index = {}
    csv_paths = [output_dir / "models_metadata.csv"]
    if output_dir.exists():
        csv_paths.extend(sorted(output_dir.glob("run_*/models_metadata.csv")))
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        metadata_root = csv_path.parent
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for raw in csv.DictReader(file):
                try:
                    result = json.loads(raw.get("mafsal_sonuc_okuma_durumu") or "{}")
                except json.JSONDecodeError:
                    continue
                directions = result.get("directions") if isinstance(result, dict) else None
                if not isinstance(directions, dict):
                    continue
                for direction in ("X", "Y"):
                    direction_result = json_direction_result(metadata_root, raw, direction) or directions.get(direction)
                    if not isinstance(direction_result, dict):
                        continue
                    first = direction_result.get("first_plastic_hinge") or {}
                    critical = direction_result.get("critical_event") or first_critical_event(direction_result.get("critical_events"))
                    first_column = direction_result.get("first_column_hinge") or {}
                    first_ls = direction_result.get("first_ls_level") or {}
                    first_cp = direction_result.get("first_cp_level") or {}
                    counts = direction_result.get("state_counts") or {}
                    if not first or not critical:
                        continue
                    row = metadata_features(raw, direction)
                    spans_x = parse_list(raw.get("aciklik_uzunluklari_x"))
                    spans_y = parse_list(raw.get("aciklik_uzunluklari_y"))
                    story_count = int(parse_float(raw.get("kat_sayisi")) or 1)
                    story_height = parse_float(raw.get("kat_yuksekligi")) or 3.2
                    first_location = parse_event_location(str(first.get("element_name") or ""), spans_x, spans_y, story_count, story_height)
                    critical_location = parse_event_location(str(critical.get("element_name") or ""), spans_x, spans_y, story_count, story_height)
                    first_column_location = parse_event_location(str(first_column.get("element_name") or ""), spans_x, spans_y, story_count, story_height)
                    first_ls_location = parse_event_location(str(first_ls.get("element_name") or ""), spans_x, spans_y, story_count, story_height)
                    first_cp_location = parse_event_location(str(first_cp.get("element_name") or ""), spans_x, spans_y, story_count, story_height)
                    row.update(
                        {
                            "first_hinge_type": str(first.get("element_type") or "unknown"),
                            "critical_element_type": str(critical.get("element_type") or "unknown"),
                            "critical_state": str(critical.get("hinge_state_level") or "unknown"),
                            "has_lscp": "yes" if int(counts.get("LS-CP", 0) or 0) > 0 else "no",
                            "first_column_available": "yes" if first_column else "no",
                            "first_ls_available": "yes" if first_ls else "no",
                            "first_cp_available": "yes" if first_cp else "no",
                            "first_event": {
                                "element_name": str(first.get("element_name") or ""),
                                "element_type": str(first.get("element_type") or "unknown"),
                                "hinge_state_level": str(first.get("hinge_state_level") or ""),
                                "step_number": parse_float(first.get("step_number")),
                                "location": first_location,
                            },
                            "critical_event": {
                                "element_name": str(critical.get("element_name") or ""),
                                "element_type": str(critical.get("element_type") or "unknown"),
                                "hinge_state_level": str(critical.get("hinge_state_level") or ""),
                                "step_number": parse_float(critical.get("step_number")),
                                "location": critical_location,
                            },
                            "first_column_event": make_event_payload(first_column, first_column_location),
                            "first_ls_event": make_event_payload(first_ls, first_ls_location),
                            "first_cp_event": make_event_payload(first_cp, first_cp_location),
                        }
                    )
                    result_row = som_index.get((str(row.get("model_name") or ""), direction), {})
                    for target in REGRESSION_TARGETS:
                        row[target] = numeric_target(result_row, target)
                    row["_source_key"] = training_row_key(row)
                    rows.append(row)
    return Dataset(merge_training_cache(output_dir, rows))


def merge_training_cache(output_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge preserved ML cache with freshly parsed metadata rows."""
    merged: dict[str, dict[str, Any]] = {}
    cache_path = output_dir / TRAINING_CACHE_FILE
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("_source_key") or f"cache::{len(merged)}")
                merged[key] = row
    for index, row in enumerate(rows):
        key = str(row.get("_source_key") or f"metadata::{index}")
        merged[key] = row
    return list(merged.values())


def save_training_cache(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Persist training rows independently from SAP2000 model artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / TRAINING_CACHE_FILE
    with cache_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def training_row_key(row: dict[str, Any]) -> str:
    """Return a stable cache key for one learned direction result."""
    payload = {key: value for key, value in row.items() if key != "_source_key"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def metadata_features(row: dict[str, Any], direction: str) -> dict[str, Any]:
    """Extract ML feature row from one metadata CSV row."""
    spans_x = parse_list(row.get("aciklik_uzunluklari_x"))
    spans_y = parse_list(row.get("aciklik_uzunluklari_y"))
    column_width, column_depth = parse_section(row.get("kolon_kesitleri"))
    beam_width, beam_depth = parse_section(row.get("kiris_kesitleri"))
    story_count = parse_float(row.get("kat_sayisi"))
    story_height = parse_float(row.get("kat_yuksekligi"))
    total_height = parse_float(row.get("toplam_yukseklik")) or story_count * story_height
    column_area = column_width * column_depth
    beam_area = beam_width * beam_depth
    column_beam_stiffness_ratio = safe_divide(column_width * column_depth**3, beam_width * beam_depth**3)
    plan_area = sum(spans_x) * sum(spans_y) if spans_x and spans_y else 0.0
    total_mass_proxy = plan_area * max(story_count, 1.0)
    subgrade_modulus = parse_float(row.get("zemin_yatak_katsayisi_kn_m3"))
    has_shear_walls = str(row.get("perde_var")).strip().lower() in {"true", "1", "yes", "evet"}
    direction_key = str(direction or "").upper()
    direction_spans = spans_y if direction_key == "Y" else spans_x
    direction_bay_count = parse_float(row.get("y_yonu_aciklik_sayisi" if direction_key == "Y" else "x_yonu_aciklik_sayisi"))
    wall_count = parse_float(row.get("perde_sayisi")) if has_shear_walls else 0.0
    wall_thickness = parse_float(row.get("perde_kalinligi_m")) if has_shear_walls else 0.0
    wall_length = parse_float(row.get("perde_uzunlugu_m")) if has_shear_walls else 0.0
    return {
        "model_name": Path(str(row.get("model_kayit_yolu") or "")).stem,
        "direction": direction,
        "story_count": story_count,
        "bay_count": direction_bay_count,
        "x_bay_count": parse_float(row.get("x_yonu_aciklik_sayisi")),
        "y_bay_count": parse_float(row.get("y_yonu_aciklik_sayisi")),
        "avg_span": sum(direction_spans) / len(direction_spans) if direction_spans else 0.0,
        "avg_span_x": sum(spans_x) / len(spans_x) if spans_x else 0.0,
        "avg_span_y": sum(spans_y) / len(spans_y) if spans_y else 0.0,
        "max_span_x": max(spans_x) if spans_x else 0.0,
        "max_span_y": max(spans_y) if spans_y else 0.0,
        "story_height": story_height,
        "total_height": total_height,
        "concrete_class": str(row.get("beton_sinifi") or "C30"),
        "concrete_fck": parse_float(row.get("beton_fck_mpa")) or concrete_fck(str(row.get("beton_sinifi") or "C30")),
        "steel_class": str(row.get("celik_sinifi") or "B420C"),
        "steel_fy": 500.0 if "500" in str(row.get("celik_sinifi")) else 420.0,
        "column_width": column_width,
        "column_depth": column_depth,
        "beam_width": beam_width,
        "beam_depth": beam_depth,
        "column_area": column_area,
        "beam_area": beam_area,
        "column_beam_stiffness_ratio": column_beam_stiffness_ratio,
        "rho_col": parse_float(row.get("kolon_donati_orani")),
        "rho_beam_top": parse_float(row.get("kiris_mesnet_ust_donati_orani")),
        "rho_beam_bottom": parse_float(row.get("kiris_aciklik_alt_donati_orani")),
        "slab_thickness": parse_float(row.get("doseme_kalinligi_m")),
        "slab_rebar_ratio": parse_float(row.get("doseme_donati_orani")),
        "soil_class": str(row.get("zemin_sinifi") or "ZC"),
        "foundation_type": str(row.get("temel_tipi") or "radye"),
        "subgrade_modulus": subgrade_modulus,
        "raft_thickness": parse_float(row.get("radye_kalinligi_m")),
        "raft_rebar_ratio": parse_float(row.get("radye_donati_orani")),
        "target_drift": parse_float(row.get("pushover_hedef_otelemesi_orani")),
        "has_shear_walls": has_shear_walls,
        "wall_count": wall_count,
        "wall_area": wall_count * wall_thickness * wall_length,
        "wall_thickness": wall_thickness,
        "wall_length": wall_length,
        "wall_rebar_ratio": parse_float(row.get("perde_donati_orani")) if has_shear_walls else 0.0,
        "initial_stiffness_proxy": safe_divide(column_area * column_depth**2 * max(story_count, 1.0), max(total_height, 1.0)),
        "design_base_shear_ratio_proxy": safe_divide(total_mass_proxy, max(subgrade_modulus, 1.0)),
        "total_mass_proxy": total_mass_proxy,
    }


def json_direction_result(output_dir: Path, row: dict[str, Any], direction: str) -> dict[str, Any] | None:
    """Read richer hinge summary from per-model JSON metadata when available."""
    model_path = Path(str(row.get("model_kayit_yolu") or ""))
    if not model_path.name:
        return None
    metadata_path = output_dir / f"{model_path.stem}_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    plastic = data.get("plastic_hinges", {}) if isinstance(data.get("plastic_hinges"), dict) else {}
    results = plastic.get("results", {}) if isinstance(plastic.get("results"), dict) else {}
    summaries = results.get("summary_by_direction", {}) if isinstance(results.get("summary_by_direction"), dict) else {}
    value = summaries.get(direction)
    return value if isinstance(value, dict) else None


def add_som_derived_features(rows: list[dict[str, Any]], random_seed: int, selected_x_columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Add SOM-informed X-only features to ML rows.

    The SOM is trained only from input/design parameters. Result labels such as
    critical state, first hinge type, or LS/CP flags are deliberately excluded to
    avoid target leakage.
    """
    if len(rows) < 4:
        return rows
    selected_x = [key for key in (selected_x_columns or DEFAULT_X_COLUMNS) if key in rows[0] and key in X_COLUMNS]
    if len(selected_x) < 3:
        return rows
    som_input = [{key: row.get(key) for key in selected_x} for row in rows]
    x_scaled, feature_spec = preprocess_som_input(som_input, selected_x)
    side = max(2, min(8, round(math.sqrt(max(len(rows) / 2.0, 4.0)))))
    weights = train_som_weights(x_scaled, side, side, 800, random_seed)
    assignments = assign_bmu(weights, x_scaled, side)
    numeric_stats = som_numeric_stats(rows, selected_x)
    zscore_features = top_som_zscore_features(rows, numeric_stats, limit=8)
    interaction_pairs = [(a, b) for index, a in enumerate(zscore_features[:5]) for b in zscore_features[index + 1 : 5]]
    enriched: list[dict[str, Any]] = []
    for row, vector, assignment in zip(rows, x_scaled, assignments):
        item = dict(row)
        bmu_x = int(assignment.get("cell_x", assignment.get("x", 0)))
        bmu_y = int(assignment.get("cell_y", assignment.get("y", 0)))
        item["som_bmu_x_norm"] = bmu_x / max(side - 1, 1)
        item["som_bmu_y_norm"] = bmu_y / max(side - 1, 1)
        item["som_bmu_distance"] = float(assignment.get("distance") or 0.0)
        item["som_cell_id"] = f"{bmu_x}_{bmu_y}"
        z_values: dict[str, float] = {}
        for key in zscore_features:
            mean, std = numeric_stats[key]
            z = (numeric_or_ordinal_value(key, row.get(key)) - mean) / std if std > 1e-9 else 0.0
            clipped = max(-4.0, min(4.0, z))
            z_values[key] = clipped
            item[f"som_z_{key}"] = clipped
            item[f"som_state_{key}"] = "high" if clipped >= 0.5 else "low" if clipped <= -0.5 else "near_avg"
        for left, right in interaction_pairs:
            item[f"som_combo_{left}__{right}"] = z_values.get(left, 0.0) * z_values.get(right, 0.0)
        enriched.append(item)
    if enriched:
        enriched[0]["_som_feature_model"] = {
            "enabled": True,
            "selected_x_columns": selected_x,
            "width": side,
            "height": side,
            "weights": weights,
            "feature_spec": feature_spec,
            "numeric_stats": {key: {"mean": mean, "std": std} for key, (mean, std) in numeric_stats.items()},
            "zscore_features": zscore_features,
            "interaction_pairs": interaction_pairs,
            "note": "SOM model is trained only with X/design parameters; cell result labels are not used as ML inputs.",
        }
    return enriched


def strip_som_features(row: dict[str, Any]) -> dict[str, Any]:
    """Return a row without transient SOM-derived feature columns."""
    return {key: value for key, value in row.items() if not key.startswith(SOM_FEATURE_PREFIX)}


def add_som_features_to_prediction(row: dict[str, Any], som_feature_model: dict[str, Any]) -> dict[str, Any]:
    """Add SOM-derived features to one prediction row using stored SOM weights."""
    if not som_feature_model.get("enabled"):
        return row
    selected_x = [key for key in som_feature_model.get("selected_x_columns", []) if key in X_COLUMNS]
    weights = som_feature_model.get("weights", [])
    feature_spec = som_feature_model.get("feature_spec", {})
    if not selected_x or not isinstance(weights, list) or not weights:
        return row
    som_input = [{key: row.get(key) for key in selected_x}]
    x_scaled = transform_som_input_with_spec(som_input, selected_x, feature_spec)
    if not x_scaled:
        return row
    side = int(som_feature_model.get("width") or 1)
    assignment = assign_bmu(weights, x_scaled, side)[0]
    item = dict(row)
    bmu_x = int(assignment.get("cell_x", assignment.get("x", 0)))
    bmu_y = int(assignment.get("cell_y", assignment.get("y", 0)))
    item["som_bmu_x_norm"] = bmu_x / max(side - 1, 1)
    item["som_bmu_y_norm"] = bmu_y / max(int(som_feature_model.get("height") or side) - 1, 1)
    item["som_bmu_distance"] = float(assignment.get("distance") or 0.0)
    item["som_cell_id"] = f"{bmu_x}_{bmu_y}"
    numeric_stats = som_feature_model.get("numeric_stats", {})
    z_values: dict[str, float] = {}
    for key in som_feature_model.get("zscore_features", []):
        stats = numeric_stats.get(key, {}) if isinstance(numeric_stats, dict) else {}
        mean = float(stats.get("mean") or 0.0)
        std = float(stats.get("std") or 0.0)
        z = (numeric_or_ordinal_value(key, row.get(key)) - mean) / std if std > 1e-9 else 0.0
        clipped = max(-4.0, min(4.0, z))
        z_values[key] = clipped
        item[f"som_z_{key}"] = clipped
        item[f"som_state_{key}"] = "high" if clipped >= 0.5 else "low" if clipped <= -0.5 else "near_avg"
    for left, right in som_feature_model.get("interaction_pairs", []):
        item[f"som_combo_{left}__{right}"] = z_values.get(left, 0.0) * z_values.get(right, 0.0)
    return item


def transform_som_input_with_spec(rows: list[dict[str, Any]], selected_x: list[str], feature_spec: dict[str, Any]) -> list[list[float]]:
    """Encode SOM input rows using a stored SOM preprocessing spec."""
    encoded_features = list(feature_spec.get("encoded_features", []))
    category_levels = feature_spec.get("category_levels", {}) if isinstance(feature_spec.get("category_levels"), dict) else {}
    means = [float(value) for value in feature_spec.get("means", [])]
    stds = [float(value) or 1.0 for value in feature_spec.get("stds", [])]
    encoded: list[list[float]] = []
    for row in rows:
        raw: list[float] = []
        for feature in encoded_features:
            source = str(feature.get("source") or "")
            kind = str(feature.get("kind") or "")
            if kind == "one_hot":
                encoded_name = str(feature.get("encoded") or "")
                level = encoded_name.split("=", 1)[1] if "=" in encoded_name else ""
                value = str(row.get(source) if row.get(source) not in (None, "") else "-")
                raw.append(1.0 if value == level else 0.0)
            else:
                raw.append(numeric_or_ordinal_value(source, row.get(source)))
        scaled = []
        for index, value in enumerate(raw):
            mean = means[index] if index < len(means) else 0.0
            std = stds[index] if index < len(stds) else 1.0
            scaled.append((value - mean) / std)
        encoded.append(scaled)
    return encoded


def som_numeric_stats(rows: list[dict[str, Any]], selected_x: list[str]) -> dict[str, tuple[float, float]]:
    """Return mean/std for numeric SOM X columns."""
    stats: dict[str, tuple[float, float]] = {}
    for key in selected_x:
        if X_COLUMNS[key].get("kind") not in {"numeric", "ordinal"}:
            continue
        values = [numeric_or_ordinal_value(key, row.get(key)) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        stats[key] = (mean, math.sqrt(variance))
    return stats


def top_som_zscore_features(rows: list[dict[str, Any]], stats: dict[str, tuple[float, float]], limit: int) -> list[str]:
    """Pick numeric X features with the largest spread for SOM interactions."""
    ranked = sorted(stats.items(), key=lambda item: item[1][1], reverse=True)
    return [key for key, (_mean, std) in ranked if std > 1e-9][:limit]


def first_critical_event(events: Any) -> dict[str, Any]:
    """Return the first stored critical event from a list."""
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                return event
    return {}


def normalize_prediction_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize dashboard prediction payload into ML feature row."""
    row: dict[str, Any] = {}
    for key in BASE_NUMERIC_FEATURES:
        row[key] = parse_float(params.get(key))
    for key in BASE_CATEGORICAL_FEATURES:
        row[key] = params.get(key)
    row["has_shear_walls"] = bool(row.get("has_shear_walls"))
    if not row.get("concrete_fck"):
        row["concrete_fck"] = concrete_fck(str(row.get("concrete_class") or "C30"))
    if not row.get("steel_fy"):
        row["steel_fy"] = 500.0 if "500" in str(row.get("steel_class")) else 420.0
    if not row.get("total_height"):
        row["total_height"] = row["story_count"] * row["story_height"]
    row["column_area"] = row["column_width"] * row["column_depth"]
    row["beam_area"] = row["beam_width"] * row["beam_depth"]
    row["column_beam_stiffness_ratio"] = safe_divide(
        row["column_width"] * row["column_depth"] ** 3,
        row["beam_width"] * row["beam_depth"] ** 3,
    )
    row["foundation_type"] = str(params.get("foundation_type") or "radye")
    row["wall_count"] = parse_float(params.get("wall_count")) if row["has_shear_walls"] else 0.0
    row["wall_thickness"] = parse_float(params.get("wall_thickness")) if row["has_shear_walls"] else 0.0
    row["wall_length"] = parse_float(params.get("wall_length")) if row["has_shear_walls"] else 0.0
    row["wall_rebar_ratio"] = parse_float(params.get("wall_rebar_ratio")) if row["has_shear_walls"] else 0.0
    row["wall_area"] = row["wall_count"] * row["wall_thickness"] * row["wall_length"]
    direction_key = str(row.get("direction") or "").upper()
    if direction_key == "Y":
        row["bay_count"] = row["y_bay_count"]
        row["avg_span"] = row["avg_span_y"]
    else:
        row["bay_count"] = row["x_bay_count"]
        row["avg_span"] = row["avg_span_x"]
    plan_area = row["avg_span_x"] * row["x_bay_count"] * row["avg_span_y"] * row["y_bay_count"]
    row["total_mass_proxy"] = plan_area * max(row["story_count"], 1.0)
    row["initial_stiffness_proxy"] = safe_divide(
        row["column_area"] * row["column_depth"] ** 2 * max(row["story_count"], 1.0),
        max(row["total_height"], 1.0),
    )
    row["design_base_shear_ratio_proxy"] = safe_divide(row["total_mass_proxy"], max(row["subgrade_modulus"], 1.0))
    return row


def available_numeric_features(rows: list[dict[str, Any]]) -> list[str]:
    """Return base plus SOM-derived numeric features present in rows."""
    features = list(BASE_NUMERIC_FEATURES)
    dynamic = sorted(
        key
        for row in rows
        for key, value in row.items()
        if key.startswith((SOM_FEATURE_PREFIX, STAGE_FEATURE_PREFIX)) and isinstance(value, (int, float)) and key not in features
    )
    for key in dynamic:
        if key not in features:
            features.append(key)
    return features


def available_categorical_features(rows: list[dict[str, Any]]) -> list[str]:
    """Return base plus SOM-derived categorical features present in rows."""
    features = list(BASE_CATEGORICAL_FEATURES)
    dynamic = sorted(
        key
        for row in rows
        for key, value in row.items()
        if key.startswith((SOM_FEATURE_PREFIX, STAGE_FEATURE_PREFIX)) and isinstance(value, str) and key not in features
    )
    for key in dynamic:
        if key not in features:
            features.append(key)
    return features


def build_feature_spec(rows: list[dict[str, Any]], selected_x_columns: list[str] | None = None) -> dict[str, Any]:
    """Build numeric ranges and categorical value lists."""
    numeric: dict[str, dict[str, float]] = {}
    if selected_x_columns is None:
        numeric_features = available_numeric_features(rows)
        categorical_features = available_categorical_features(rows)
    else:
        selected = [key for key in selected_x_columns if key in X_COLUMNS]
        numeric_features = [key for key in selected if X_COLUMNS[key].get("kind") == "numeric"]
        categorical_features = [key for key in selected if X_COLUMNS[key].get("kind") != "numeric"]
        numeric_features.extend(
            key
            for key in available_numeric_features(rows)
            if key.startswith((SOM_FEATURE_PREFIX, STAGE_FEATURE_PREFIX)) and key not in numeric_features
        )
        categorical_features.extend(
            key
            for key in available_categorical_features(rows)
            if key.startswith((SOM_FEATURE_PREFIX, STAGE_FEATURE_PREFIX)) and key not in categorical_features
        )
    for key in numeric_features:
        values = [float(row.get(key) or 0.0) for row in rows]
        numeric[key] = {"min": min(values), "max": max(values)}
    categorical = {key: sorted({str(row.get(key)) for row in rows}) for key in categorical_features}
    return {"numeric": numeric, "categorical": categorical}


def encode_features(row: dict[str, Any], spec: dict[str, Any]) -> list[float]:
    """Encode numeric and categorical features for distance-based learning."""
    encoded: list[float] = []
    for key in spec["numeric"]:
        bounds = spec["numeric"][key]
        value = float(row.get(key) or 0.0)
        span = float(bounds["max"]) - float(bounds["min"])
        encoded.append(0.0 if span == 0.0 else (value - float(bounds["min"])) / span)
    for key in spec["categorical"]:
        value = str(row.get(key))
        for option in spec["categorical"][key]:
            encoded.append(1.0 if value == option else 0.0)
    return encoded


def numeric_target(row: dict[str, Any], target: str) -> float | None:
    """Return one finite numeric target or None when the result is unavailable."""
    value = row.get(target) if isinstance(row, dict) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def train_numeric_knn_target(train_rows: list[tuple[list[float], dict[str, Any]]], target: str) -> dict[str, Any]:
    """Select k for a leakage-safe first-stage numeric KNN regressor."""
    valid = [(features, row) for features, row in train_rows if numeric_target(row, target) is not None]
    if not valid:
        return {"algorithm": "knn_regression", "k": 1, "available": False, "mean": 0.0, "sample_count": 0}
    mean = sum(numeric_target(row, target) or 0.0 for _, row in valid) / len(valid)
    if len(valid) < 3:
        return {"algorithm": "knn_regression", "k": 1, "available": True, "mean": mean, "sample_count": len(valid)}
    candidates = [value for value in (1, 3, 5, 7, 9, 11) if value < len(valid)] or [1]
    best_k = candidates[0]
    best_mae = float("inf")
    for k in candidates:
        errors: list[float] = []
        for index, (features, row) in enumerate(valid):
            pool = valid[:index] + valid[index + 1 :]
            predicted = predict_numeric_knn({"k": k, "available": True, "mean": mean}, pool, features, target)
            actual = numeric_target(row, target)
            if predicted is not None and actual is not None:
                errors.append(abs(actual - predicted))
        mae = sum(errors) / len(errors) if errors else float("inf")
        if mae < best_mae:
            best_k, best_mae = k, mae
    return {
        "algorithm": "knn_regression",
        "k": best_k,
        "available": True,
        "mean": mean,
        "sample_count": len(valid),
        "validation_mae": round(best_mae, 6),
    }


def predict_numeric_knn(
    model: dict[str, Any],
    train_rows: list[tuple[list[float], dict[str, Any]]],
    features: list[float],
    target: str,
) -> float | None:
    """Predict one numeric result from nearest valid structural-parameter rows."""
    if not model.get("available"):
        return None
    distances = []
    for item_features, row in train_rows:
        value = numeric_target(row, target)
        if value is not None:
            distances.append((euclidean(features, item_features), value))
    if not distances:
        return float(model.get("mean") or 0.0)
    nearest = sorted(distances, key=lambda item: item[0])[: max(1, min(int(model.get("k", 5)), len(distances)))]
    exact = [value for distance, value in nearest if distance <= 1e-12]
    if exact:
        return sum(exact) / len(exact)
    weights = [(1.0 / max(distance, 1e-9), value) for distance, value in nearest]
    return sum(weight * value for weight, value in weights) / sum(weight for weight, _ in weights)


def add_out_of_fold_stage_features(
    base_rows: list[tuple[list[float], dict[str, Any]]], regression_models: dict[str, Any]
) -> list[tuple[list[float], dict[str, Any]]]:
    """Create first-stage training predictions without exposing each row's own result."""
    staged: list[tuple[list[float], dict[str, Any]]] = []
    for index, (features, row) in enumerate(base_rows):
        pool = base_rows[:index] + base_rows[index + 1 :]
        item = dict(row)
        for target, model in regression_models.items():
            value = predict_numeric_knn(model, pool, features, target)
            item[f"{STAGE_FEATURE_PREFIX}{target}"] = value if value is not None else 0.0
        staged.append((features, item))
    return staged


def add_stage_features(
    base_rows: list[tuple[list[float], dict[str, Any]]],
    regression_models: dict[str, Any],
    regression_training_rows: list[tuple[list[float], dict[str, Any]]],
) -> list[tuple[list[float], dict[str, Any]]]:
    """Add first-stage numeric predictions to held-out or future rows."""
    staged: list[tuple[list[float], dict[str, Any]]] = []
    for features, row in base_rows:
        item = dict(row)
        for target, model in regression_models.items():
            value = predict_numeric_knn(model, regression_training_rows, features, target)
            item[f"{STAGE_FEATURE_PREFIX}{target}"] = value if value is not None else 0.0
        staged.append((features, item))
    return staged


def regression_score_metrics(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Return MAE, RMSE and R2 for one held-out numeric target."""
    if not pairs:
        return {"test_count": 0, "mae": None, "rmse": None, "r2": None}
    actual = [item[0] for item in pairs]
    predicted = [item[1] for item in pairs]
    errors = [a - p for a, p in pairs]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(actual) / len(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    residual = sum(error * error for error in errors)
    r2 = 1.0 - residual / total if total > 1e-12 else None
    return {
        "test_count": len(pairs),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6) if r2 is not None else None,
    }


def optimize_k(train_rows: list[tuple[list[float], dict[str, Any]]], target: str) -> tuple[int, float]:
    """Select k by leave-one-out validation on a small candidate set."""
    candidates = [1, 3, 5, 7, 9, 11]
    best_k = 1
    best_score = -1.0
    for k in candidates:
        correct = 0
        total = 0
        for index, (features, row) in enumerate(train_rows):
            pool = train_rows[:index] + train_rows[index + 1 :]
            if not pool:
                continue
            prediction = predict_knn(pool, features, target, min(k, len(pool)))
            correct += int(prediction == row[target])
            total += 1
        score = correct / total if total else 0.0
        if score > best_score:
            best_k = k
            best_score = score
    return best_k, best_score


def normalize_algorithm(algorithm: str) -> str:
    """Normalize algorithm name from dashboard."""
    value = str(algorithm or "knn").strip().lower().replace("-", "_")
    aliases = {"rf": "random_forest", "randomforest": "random_forest", "xgboost": "xgboost", "lightgbm": "lightgbm"}
    return aliases.get(value, value)


def raise_optional_algorithm_error(algorithm: str) -> None:
    """Raise a useful message for optional algorithms not bundled here."""
    if algorithm == "xgboost":
        raise ValueError("XGBoost secildi ancak xgboost paketi bu Python ortaminda kurulu degil.")
    if algorithm == "lightgbm":
        raise ValueError("LightGBM secildi ancak lightgbm paketi bu Python ortaminda kurulu degil.")
    raise ValueError(f"Desteklenmeyen algoritma: {algorithm}")


def train_knn_target(train_rows: list[tuple[list[float], dict[str, Any]]], target: str) -> dict[str, Any]:
    """Train one KNN target by selecting k."""
    best_k, best_score = optimize_k(train_rows, target)
    return {"algorithm": "knn", "k": best_k, "validation_accuracy": round(best_score, 4)}


def train_random_forest_target(train_rows: list[tuple[list[float], dict[str, Any]]], target: str, random_seed: int) -> dict[str, Any]:
    """Train a compact dependency-free random forest classifier."""
    rng = random.Random(random_seed)
    tree_count = 60
    max_depth = 8
    min_samples = 4
    feature_count = len(train_rows[0][0])
    feature_sample_count = max(1, int(math.sqrt(feature_count)))
    trees = []
    for _ in range(tree_count):
        sample = [rng.choice(train_rows) for _ in range(len(train_rows))]
        trees.append(build_tree(sample, target, rng, max_depth, min_samples, feature_sample_count, feature_count))
    return {
        "algorithm": "random_forest",
        "trees": trees,
        "tree_count": tree_count,
        "max_depth": max_depth,
        "min_samples": min_samples,
        "location_k": 7,
    }


def train_external_target(
    train_rows: list[tuple[list[float], dict[str, Any]]], target: str, algorithm: str, random_seed: int
) -> tuple[dict[str, Any], Any | None]:
    """Train an optional package-backed classifier."""
    labels = [str(row[target]) for _, row in train_rows]
    unique_labels = sorted(set(labels))
    if len(unique_labels) == 1:
        return {"algorithm": algorithm, "constant": unique_labels[0], "labels": unique_labels, "location_k": 7}, None

    try:
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:  # noqa: BLE001
        raise ValueError("scikit-learn kurulu degil; XGBoost/LightGBM icin gerekli.") from exc

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x = [features for features, _ in train_rows]

    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001
            raise ValueError("xgboost paketi bu Python ortaminda kurulu degil.") from exc
        estimator = XGBClassifier(
            n_estimators=90,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_seed,
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # noqa: BLE001
            raise ValueError("lightgbm paketi bu Python ortaminda kurulu degil.") from exc
        estimator = LGBMClassifier(
            n_estimators=90,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_seed,
            verbose=-1,
        )
    else:
        raise_optional_algorithm_error(algorithm)

    estimator.fit(x, y)
    external_model = {"estimator": estimator, "encoder": encoder}
    return {
        "algorithm": algorithm,
        "external": True,
        "labels": [str(label) for label in encoder.classes_],
        "n_estimators": 90,
        "location_k": 7,
    }, external_model


def dump_external_models(path: Path, external_models: dict[str, Any]) -> None:
    """Persist optional package-backed models with joblib."""
    try:
        import joblib
    except Exception as exc:  # noqa: BLE001
        raise ValueError("joblib kurulu degil; harici ML modeli kaydedilemedi.") from exc
    joblib.dump(external_models, path)


def load_external_models(output_dir: Path, model: dict[str, Any]) -> dict[str, Any]:
    """Load optional package-backed models when available."""
    filename = model.get("external_model_file")
    if not filename:
        return {}
    path = output_dir / str(filename)
    if not path.exists():
        return {}
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return {}


def build_tree(
    rows: list[tuple[list[float], dict[str, Any]]],
    target: str,
    rng: random.Random,
    max_depth: int,
    min_samples: int,
    feature_sample_count: int,
    feature_count: int,
    depth: int = 0,
) -> dict[str, Any]:
    """Build one decision tree using Gini splits."""
    labels = [str(row[target]) for _, row in rows]
    majority = Counter(labels).most_common(1)[0][0]
    if depth >= max_depth or len(rows) <= min_samples or len(set(labels)) == 1:
        return {"leaf": True, "label": majority, "counts": dict(Counter(labels))}
    split = best_split(rows, target, rng.sample(range(feature_count), min(feature_sample_count, feature_count)))
    if split is None:
        return {"leaf": True, "label": majority, "counts": dict(Counter(labels))}
    feature_index, threshold, left, right = split
    if not left or not right:
        return {"leaf": True, "label": majority, "counts": dict(Counter(labels))}
    return {
        "leaf": False,
        "feature": feature_index,
        "threshold": threshold,
        "default": majority,
        "left": build_tree(left, target, rng, max_depth, min_samples, feature_sample_count, feature_count, depth + 1),
        "right": build_tree(right, target, rng, max_depth, min_samples, feature_sample_count, feature_count, depth + 1),
    }


def best_split(
    rows: list[tuple[list[float], dict[str, Any]]], target: str, feature_indices: list[int]
) -> tuple[int, float, list[tuple[list[float], dict[str, Any]]], list[tuple[list[float], dict[str, Any]]]] | None:
    """Find the best Gini split among sampled features."""
    best: tuple[float, int, float, list[tuple[list[float], dict[str, Any]]], list[tuple[list[float], dict[str, Any]]]] | None = None
    for feature_index in feature_indices:
        values = sorted({features[feature_index] for features, _ in rows})
        if len(values) <= 1:
            continue
        if len(values) > 12:
            step = max(1, len(values) // 12)
            thresholds = [(values[index] + values[min(index + step, len(values) - 1)]) / 2 for index in range(0, len(values) - 1, step)]
        else:
            thresholds = [(values[index] + values[index + 1]) / 2 for index in range(len(values) - 1)]
        for threshold in thresholds:
            left = [(features, row) for features, row in rows if features[feature_index] <= threshold]
            right = [(features, row) for features, row in rows if features[feature_index] > threshold]
            if not left or not right:
                continue
            score = weighted_gini(left, right, target)
            if best is None or score < best[0]:
                best = (score, feature_index, threshold, left, right)
    if best is None:
        return None
    return best[1], round(best[2], 6), best[3], best[4]


def weighted_gini(left: list[tuple[list[float], dict[str, Any]]], right: list[tuple[list[float], dict[str, Any]]], target: str) -> float:
    """Return weighted Gini impurity."""
    total = len(left) + len(right)
    return (len(left) / total) * gini(left, target) + (len(right) / total) * gini(right, target)


def gini(rows: list[tuple[list[float], dict[str, Any]]], target: str) -> float:
    """Return Gini impurity."""
    counts = Counter(str(row[target]) for _, row in rows)
    total = len(rows)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def predict_knn(train_rows: list[tuple[list[float], dict[str, Any]]], features: list[float], target: str, k: int) -> str:
    """Predict one target with k nearest neighbors."""
    prediction, _, _ = predict_knn_with_details(train_rows, features, target, k)
    return prediction


def predict_knn_with_details(
    train_rows: list[tuple[list[float], dict[str, Any]]], features: list[float], target: str, k: int
) -> tuple[str, dict[str, float], list[dict[str, Any]]]:
    """Predict with probabilities and neighbor summary."""
    distances = sorted(((euclidean(features, item_features), targets) for item_features, targets in train_rows), key=lambda item: item[0])
    nearest = distances[: max(1, min(k, len(distances)))]
    counts = Counter(str(targets[target]) for _, targets in nearest)
    prediction = counts.most_common(1)[0][0]
    probabilities = {label: round(count / len(nearest), 3) for label, count in counts.items()}
    neighbors = [{"distance": round(distance, 4), "label": targets[target]} for distance, targets in nearest[:5]]
    return prediction, probabilities, neighbors


def predict_target(model: dict[str, Any], train_rows: list[tuple[list[float], dict[str, Any]]], features: list[float], target: str) -> str:
    """Predict one target with the selected model."""
    prediction, _, _ = predict_target_with_details(model, train_rows, features, target)
    return prediction


def predict_target_with_details(
    model: dict[str, Any], train_rows: list[tuple[list[float], dict[str, Any]]], features: list[float], target: str
) -> tuple[str, dict[str, float], list[dict[str, Any]]]:
    """Predict with selected model and return compact details."""
    algorithm = model.get("algorithm")
    if algorithm == "knn":
        return predict_knn_with_details(train_rows, features, target, int(model["k"]))
    if algorithm == "random_forest":
        votes = Counter(predict_tree(tree, features) for tree in model.get("trees", []))
        if not votes:
            return predict_knn_with_details(train_rows, features, target, 5)
        total = sum(votes.values())
        prediction = votes.most_common(1)[0][0]
        probabilities = {label: round(count / total, 3) for label, count in votes.items()}
        details = [{"label": label, "votes": count} for label, count in votes.most_common(5)]
        return prediction, probabilities, details
    if algorithm in ("xgboost", "lightgbm"):
        if "constant" in model:
            label = str(model["constant"])
            return label, {label: 1.0}, [{"label": label, "constant": True}]
        external = model.get("_external_model")
        if not external:
            return predict_knn_with_details(train_rows, features, target, 5)
        estimator = external["estimator"]
        encoder = external["encoder"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction_index = estimator.predict([features])[0]
            prob_values = estimator.predict_proba([features])[0] if hasattr(estimator, "predict_proba") else None
        prediction = str(encoder.inverse_transform([prediction_index])[0])
        probabilities: dict[str, float] = {}
        if prob_values is not None:
            for label, probability in zip(encoder.classes_, prob_values):
                probabilities[str(label)] = round(float(probability), 3)
        else:
            probabilities[prediction] = 1.0
        details = [{"label": label, "probability": probability} for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]]
        return prediction, probabilities, details
    return predict_knn_with_details(train_rows, features, target, 5)


def predict_tree(tree: dict[str, Any], features: list[float]) -> str:
    """Predict one decision tree."""
    node = tree
    while not node.get("leaf"):
        feature_index = int(node["feature"])
        threshold = float(node["threshold"])
        node = node["left"] if features[feature_index] <= threshold else node["right"]
    return str(node.get("label") or node.get("default") or "unknown")


def model_location_k(model: dict[str, Any]) -> int:
    """Return nearest-neighbor count used for event location projection."""
    if model.get("algorithm") == "knn":
        return int(model.get("k", 5))
    return int(model.get("location_k", 7))


def estimate_event_location(
    train_items: list[tuple[list[float], dict[str, Any]]],
    features: list[float],
    event_key: str,
    class_target: str,
    predicted_class: str,
    k: int,
    input_row: dict[str, Any],
) -> dict[str, Any]:
    """Project the closest learned event location onto the requested geometry."""
    if predicted_class == "no":
        return {"available": False, "element_name": "", "element_type": "", "method": "classified_unavailable"}
    distances = sorted(((euclidean(features, item_features), item) for item_features, item in train_items), key=lambda item: item[0])
    filtered = [
        (distance, item)
        for distance, item in distances
        if item.get("targets", {}).get(class_target) == predicted_class
        and isinstance(item.get("events", {}).get(event_key, {}).get("location"), dict)
    ]
    candidates = filtered[: max(1, k)] or distances[: max(1, k)]
    distance, item = candidates[0]
    source_event = item.get("events", {}).get(event_key, {})
    source_location = source_event.get("location", {}) if isinstance(source_event, dict) else {}
    projected = project_location_to_input(source_location, input_row)
    return {
        "source_element_name": source_event.get("element_name", ""),
        "source_element_type": source_event.get("element_type", predicted_class),
        "source_state": source_event.get("hinge_state_level", ""),
        "source_step": source_event.get("step_number"),
        "distance": round(distance, 4),
        "element_name": projected.get("element_name", ""),
        "element_type": projected.get("element_type", predicted_class),
        "story": projected.get("story"),
        "location": projected,
        "method": "nearest_neighbor_location_projection",
    }


def make_event_payload(event: dict[str, Any], location: dict[str, Any]) -> dict[str, Any]:
    """Return compact event payload for ML location projection."""
    if not event:
        return {}
    return {
        "element_name": str(event.get("element_name") or ""),
        "element_type": str(event.get("element_type") or "unknown"),
        "hinge_state_level": str(event.get("hinge_state_level") or ""),
        "step_number": parse_float(event.get("step_number")),
        "location": location,
    }


def parse_event_location(element_name: str, spans_x: list[float], spans_y: list[float], story_count: int, story_height: float) -> dict[str, Any]:
    """Parse SAP frame name into normalized plan and elevation location."""
    x_coords = cumulative(spans_x)
    y_coords = cumulative(spans_y)
    total_x = max(x_coords[-1], 1.0)
    total_y = max(y_coords[-1], 1.0)
    total_z = max(story_count * story_height, 1.0)
    match = re.match(r"^C_(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$", element_name, re.I)
    if match:
        story = clamp_int(int(match.group(1)), 1, story_count)
        x = parse_float(match.group(2))
        y = parse_float(match.group(3))
        return {
            "element_type": "column",
            "orientation": "Z",
            "story": story,
            "nx": clamp01(x / total_x),
            "ny": clamp01(y / total_y),
            "nz": clamp01(((story - 0.5) * story_height) / total_z),
        }
    match = re.match(r"^BX_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$", element_name, re.I)
    if match:
        story = clamp_int(int(match.group(1)), 1, story_count)
        index = clamp_int(int(match.group(2)), 0, max(len(x_coords) - 2, 0))
        y = parse_float(match.group(3))
        x_mid = (x_coords[index] + x_coords[min(index + 1, len(x_coords) - 1)]) / 2
        return {
            "element_type": "beam",
            "orientation": "X",
            "story": story,
            "nx": clamp01(x_mid / total_x),
            "ny": clamp01(y / total_y),
            "nz": clamp01((story * story_height) / total_z),
        }
    match = re.match(r"^BY_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$", element_name, re.I)
    if match:
        story = clamp_int(int(match.group(1)), 1, story_count)
        index = clamp_int(int(match.group(2)), 0, max(len(y_coords) - 2, 0))
        x = parse_float(match.group(3))
        y_mid = (y_coords[index] + y_coords[min(index + 1, len(y_coords) - 1)]) / 2
        return {
            "element_type": "beam",
            "orientation": "Y",
            "story": story,
            "nx": clamp01(x / total_x),
            "ny": clamp01(y_mid / total_y),
            "nz": clamp01((story * story_height) / total_z),
        }
    return {}


def project_location_to_input(location: dict[str, Any], input_row: dict[str, Any]) -> dict[str, Any]:
    """Project normalized learned location onto the user's bay grid."""
    x_bays = max(1, int(round(float(input_row.get("x_bay_count") or 1))))
    y_bays = max(1, int(round(float(input_row.get("y_bay_count") or 1))))
    story_count = max(1, int(round(float(input_row.get("story_count") or 1))))
    story_height = float(input_row.get("story_height") or 3.2)
    avg_x = max(float(input_row.get("avg_span_x") or 5.0), 0.1)
    avg_y = max(float(input_row.get("avg_span_y") or 5.0), 0.1)
    x_coords = [round(index * avg_x, 2) for index in range(x_bays + 1)]
    y_coords = [round(index * avg_y, 2) for index in range(y_bays + 1)]
    story = clamp_int(int(round(float(location.get("story", 1) or 1))), 1, story_count)
    nx = clamp01(float(location.get("nx", 0.0) or 0.0))
    ny = clamp01(float(location.get("ny", 0.0) or 0.0))
    element_type = str(location.get("element_type") or "column")
    orientation = str(location.get("orientation") or ("Z" if element_type == "column" else "X"))
    if element_type == "column":
        xi = clamp_int(round(nx * x_bays), 0, x_bays)
        yi = clamp_int(round(ny * y_bays), 0, y_bays)
        element_name = f"C_{story}_{x_coords[xi]:.2f}_{y_coords[yi]:.2f}"
        return {"element_name": element_name, "element_type": "column", "orientation": "Z", "story": story, "x": x_coords[xi], "y": y_coords[yi]}
    if orientation == "Y":
        xi = clamp_int(round(nx * x_bays), 0, x_bays)
        yi = clamp_int(math.floor(ny * y_bays), 0, y_bays - 1)
        element_name = f"BY_{story}_{yi}_{x_coords[xi]:.2f}"
        return {"element_name": element_name, "element_type": "beam", "orientation": "Y", "story": story, "x": x_coords[xi], "y": y_coords[yi]}
    xi = clamp_int(math.floor(nx * x_bays), 0, x_bays - 1)
    yi = clamp_int(round(ny * y_bays), 0, y_bays)
    element_name = f"BX_{story}_{xi}_{y_coords[yi]:.2f}"
    return {"element_name": element_name, "element_type": "beam", "orientation": "X", "story": story, "x": x_coords[xi], "y": y_coords[yi]}


def classification_metrics(actual: list[str], predictions: list[str]) -> dict[str, Any]:
    """Return compact classification metrics."""
    total = len(actual)
    correct = sum(1 for a, p in zip(actual, predictions) if a == p)
    labels = sorted(set(actual) | set(predictions))
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for a, p in zip(actual, predictions):
        confusion[a][p] += 1
    return {"accuracy": round(correct / total, 4) if total else 0.0, "test_count": total, "labels": labels, "confusion": confusion}


def feature_importance_proxy(rows: list[dict[str, Any]], feature_spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rank simple single-feature separation for first hinge type."""
    ranked: list[dict[str, Any]] = []
    numeric_features = list((feature_spec or {}).get("numeric", {}).keys()) or list(BASE_NUMERIC_FEATURES)
    for key in numeric_features:
        groups: dict[str, list[float]] = {}
        for row in rows:
            groups.setdefault(str(row["first_hinge_type"]), []).append(float(row.get(key) or 0.0))
        if len(groups) < 2:
            continue
        means = [sum(values) / len(values) for values in groups.values() if values]
        if len(means) >= 2:
            ranked.append({"feature": key, "score": round(max(means) - min(means), 4)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:10]


def euclidean(left: list[float], right: list[float]) -> float:
    """Return Euclidean distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator/denominator with zero protection."""
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def parse_float(value: object) -> float:
    """Parse numeric metadata robustly."""
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return 0.0


def parse_list(value: object) -> list[float]:
    """Parse JSON list stored in CSV."""
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [parse_float(item) for item in parsed if parse_float(item) > 0.0]


def parse_section(value: object) -> tuple[float, float]:
    """Parse '40x40 cm' section labels into meters."""
    match = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", str(value or ""))
    if not match:
        return 0.0, 0.0
    return parse_float(match.group(1)) / 100.0, parse_float(match.group(2)) / 100.0


def cumulative(spans: list[float]) -> list[float]:
    """Return cumulative grid coordinates."""
    coords = [0.0]
    for span in spans:
        coords.append(round(coords[-1] + span, 3))
    if len(coords) == 1:
        coords.append(1.0)
    return coords


def clamp01(value: float) -> float:
    """Clamp a float to [0, 1]."""
    return max(0.0, min(1.0, value))


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    """Clamp an int to inclusive bounds."""
    return max(min_value, min(max_value, value))


def concrete_fck(concrete_class: str) -> float:
    """Return fck MPa from class label."""
    match = re.search(r"(\d+)", concrete_class)
    return float(match.group(1)) if match else 30.0
