"""Self Organizing Map utilities for generated SAP2000 model results.

Do not include pushover result variables in SOM training. These variables are
post-analysis labels and must only be used for coloring and interpretation.
Including them in X would cause target leakage.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fema440 import calculate_fema440_for_curve


SOM_FILE = "som_model.json"
TARGET_LEAKAGE_MESSAGE = (
    "Bu değişken pushover sonucu elde edilen bir performans çıktısıdır. "
    "SOM eğitimine dahil edilirse hasar bilgisi modele sızar ve sonuç yapay olarak iyileşir. "
    "Bu değişken yalnızca renklendirme/yorumlama için kullanılmalıdır."
)

X_COLUMNS: dict[str, dict[str, Any]] = {
    "direction": {"label": "Analiz yönü", "kind": "nominal"},
    "story_count": {"label": "Kat sayısı", "kind": "numeric"},
    "bay_count": {"label": "Açıklık sayısı", "kind": "numeric"},
    "avg_span": {"label": "Açıklık mesafesi", "kind": "numeric"},
    "story_height": {"label": "Kat yüksekliği", "kind": "numeric"},
    "total_height": {"label": "Yapı yüksekliği", "kind": "numeric"},
    "column_width": {"label": "Kolon genişliği", "kind": "numeric"},
    "column_depth": {"label": "Kolon yüksekliği", "kind": "numeric"},
    "column_area": {"label": "Kolon alanı", "kind": "numeric"},
    "beam_width": {"label": "Kiriş genişliği", "kind": "numeric"},
    "beam_depth": {"label": "Kiriş yüksekliği", "kind": "numeric"},
    "beam_area": {"label": "Kiriş alanı", "kind": "numeric"},
    "column_beam_stiffness_ratio": {"label": "Kolon/kiriş rijitlik oranı", "kind": "numeric"},
    "concrete_class": {"label": "Beton sınıfı", "kind": "ordinal", "mapping": "concrete"},
    "steel_class": {"label": "Çelik sınıfı", "kind": "ordinal", "mapping": "steel"},
    "rho_col": {"label": "Kolon donatı oranı", "kind": "numeric"},
    "rho_beam_top": {"label": "Kiriş üst donatı oranı", "kind": "numeric"},
    "rho_beam_bottom": {"label": "Kiriş alt donatı oranı", "kind": "numeric"},
    "slab_thickness": {"label": "Döşeme kalınlığı", "kind": "numeric"},
    "slab_rebar_ratio": {"label": "Döşeme donatı oranı", "kind": "numeric"},
    "soil_class": {"label": "Zemin sınıfı", "kind": "ordinal", "mapping": "soil"},
    "foundation_type": {"label": "Temel tipi", "kind": "nominal"},
    "subgrade_modulus": {"label": "Zemin yatak katsayısı", "kind": "numeric"},
    "raft_thickness": {"label": "Radye kalınlığı", "kind": "numeric"},
    "raft_rebar_ratio": {"label": "Radye donatı oranı", "kind": "numeric"},
    "target_drift": {"label": "Hedef drift", "kind": "numeric"},
    "has_shear_walls": {"label": "Perde var/yok", "kind": "nominal"},
    "wall_count": {"label": "Perde sayısı", "kind": "numeric"},
    "wall_area": {"label": "Perde alanı", "kind": "numeric"},
    "wall_thickness": {"label": "Perde kalınlığı", "kind": "numeric"},
    "wall_length": {"label": "Perde uzunluğu", "kind": "numeric"},
    "wall_rebar_ratio": {"label": "Perde donatı oranı", "kind": "numeric"},
}

DEFAULT_X_COLUMNS = (
    "story_count",
    "bay_count",
    "avg_span",
    "total_height",
    "column_area",
    "beam_depth",
    "column_beam_stiffness_ratio",
    "concrete_class",
    "steel_class",
    "rho_col",
    "rho_beam_top",
    "rho_beam_bottom",
    "slab_thickness",
    "slab_rebar_ratio",
    "soil_class",
    "foundation_type",
    "subgrade_modulus",
    "raft_thickness",
    "raft_rebar_ratio",
    "target_drift",
    "has_shear_walls",
    "wall_area",
)

RESULT_METRICS: dict[str, dict[str, Any]] = {
    "critical_state": {"label": "Kritik hasar seviyesi", "type": "category"},
    "max_hinge_state": {"label": "Maksimum mafsal durumu", "type": "category"},
    "performance_level_at_target": {"label": "Hedef deplasmanda performans seviyesi", "type": "category"},
    "peak_base_shear": {"label": "Taban kesme kapasitesi", "type": "number"},
    "max_displacement": {"label": "Maksimum deplasman", "type": "number"},
    "target_displacement": {"label": "Hedef deplasman", "type": "number"},
    "ductility_ratio": {"label": "Süneklik oranı", "type": "number"},
    "collapse_mechanism": {"label": "Göçme mekanizması", "type": "category"},
    "lscp_count": {"label": "LS-CP eleman sayısı", "type": "number"},
    "cpc_count": {"label": "CP eleman sayısı", "type": "number"},
    "lscp_ratio": {"label": "LS-CP oranı", "type": "number", "format": "percent"},
    "cp_ratio": {"label": "CP oranı", "type": "number", "format": "percent"},
    "max_story_drift_ratio": {"label": "Maksimum göreli kat ötelenmesi", "type": "number", "format": "percent"},
    "max_rotation": {"label": "Maksimum plastik rotasyon", "type": "number"},
    "first_hinge_type": {"label": "İlk mafsal tipi", "type": "category"},
    "critical_element_type": {"label": "Kritik eleman tipi", "type": "category"},
    "has_lscp": {"label": "LS-CP oluştu mu", "type": "category"},
    "has_cp": {"label": "CP oluştu mu", "type": "category"},
    "damage_class": {"label": "Hasar sınıfı", "type": "category"},
    "fema_ductility_mu": {"label": "FEMA 440 süneklik μ", "type": "number"},
    "fema_beta_eff_percent": {"label": "FEMA 440 etkin sönüm βeff", "type": "number", "format": "percent_value"},
    "fema_teff_t0_ratio": {"label": "FEMA 440 Teff/T0", "type": "number"},
    "fema_r_proxy": {"label": "FEMA 440 R proxy", "type": "number"},
    "fema_c1": {"label": "FEMA 440 C1", "type": "number"},
    "fema_target_displacement_proxy": {"label": "FEMA 440 hedef deplasman proxy", "type": "number"},
    "fema_target_capacity_ratio": {"label": "FEMA 440 hedef/kapasite oranı", "type": "number"},
    "fema_capacity_status": {"label": "FEMA 440 kapasite durumu", "type": "category"},
    "first_hinge_story": {"label": "Ilk plastik mafsal kacinci katta", "type": "number"},
    "critical_element_story": {"label": "Kritik eleman kacinci katta", "type": "number"},
    "first_hinge_plan_zone": {"label": "Ilk plastik mafsal kenarda/ortada", "type": "category"},
    "critical_element_plan_zone": {"label": "Kritik eleman kenarda/ortada", "type": "category"},
}

Y_ALIASES = {
    "kritik_hasar_seviyesi": "critical_state",
    "maksimum_plastik_mafsal_seviyesi": "max_hinge_state",
    "maksimum_mafsal_durumu": "max_hinge_state",
    "hasar_sinifi": "damage_class",
    "ls_cp_eleman_sayisi": "lscp_count",
    "cp_eleman_sayisi": "cpc_count",
    "ls_cp_orani": "lscp_ratio",
    "cp_orani": "cp_ratio",
    "fema_suneklik": "fema_ductility_mu",
    "fema_etkin_sonum": "fema_beta_eff_percent",
    "fema_hedef_kapasite_orani": "fema_target_capacity_ratio",
    "fema_kapasite_durumu": "fema_capacity_status",
}

X_ALIASES = {
    "x_bay_count": "bay_count",
    "y_bay_count": "bay_count",
    "avg_span_x": "avg_span",
    "avg_span_y": "avg_span",
    "max_span": "avg_span",
    "max_span_x": "avg_span",
    "max_span_y": "avg_span",
}

REMOVED_X_COLUMNS = {
    "initial_stiffness_proxy",
    "design_base_shear_ratio_proxy",
    "total_mass_proxy",
}


def train_som(
    output_dir: Path,
    width: int = 8,
    height: int = 8,
    iterations: int = 800,
    result_metric: str = "critical_state",
    random_seed: int = 42,
    selected_x_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Train a dependency-free SOM using only selected pre-analysis X columns."""
    width = max(2, min(int(width), 20))
    height = max(2, min(int(height), 20))
    iterations = max(50, min(int(iterations), 20_000))
    selected_y_column = normalize_y_column(result_metric)
    selected_x = validate_selected_columns(selected_x_columns or list(DEFAULT_X_COLUMNS), selected_y_column)

    rows = build_som_rows(output_dir)
    if len(rows) < 4:
        raise ValueError("SOM için en az 4 yön-sonucu gerekir.")

    som_input = [{key: row.get(key) for key in selected_x} for row in rows]
    y_values = [row.get(selected_y_column) for row in rows]
    x_scaled, feature_spec = preprocess_som_input(som_input, selected_x)
    weights = train_som_weights(x_scaled, width, height, iterations, random_seed)
    assignments = assign_bmu(weights, x_scaled, width)
    cells = summarize_cells(rows, assignments, y_values, selected_y_column, selected_x, width, height)
    attach_distinctive_features(cells, rows, selected_x)

    quantization_error = calculate_quantization_error(weights, x_scaled)
    topographic_error = calculate_topographic_error(weights, x_scaled, width)
    purity = calculate_purity(cells, selected_y_column)
    u_matrix = compute_u_matrix(weights, width, height)
    interpretation = som_interpretation(purity)

    payload = {
        "available": True,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "width": width,
        "height": height,
        "iterations": iterations,
        "sample_count": len(rows),
        "feature_count": len(x_scaled[0]) if x_scaled else 0,
        "selected_x_columns": selected_x,
        "selected_x_labels": [{"key": key, "label": X_COLUMNS[key]["label"]} for key in selected_x],
        "selected_y_column": selected_y_column,
        "result_metric": selected_y_column,
        "result_metric_meta": RESULT_METRICS[selected_y_column],
        "excluded_y_columns": list(RESULT_METRICS),
        "target_leakage_warning": TARGET_LEAKAGE_MESSAGE,
        "training_note": "SOM modeli yalnızca seçilen X parametreleriyle eğitilir. Seçilen Y sonucu eğitimde kullanılmaz; yalnızca harita hücrelerinin mühendislik yorumunu yapmak için sonradan renklendirilir. Böylece modelin hasar sonucunu ezberlemesi engellenir.",
        "source_note": "SOM yalnızca X parametreleriyle eğitilir; seçilen Y sonucu harita hücrelerini yorumlamak için sonradan renklendirilir.",
        "engineering_question": "Benzer yapısal/tasarım parametrelerine sahip modeller, pushover sonucunda benzer kritik hasar seviyelerine mi ulaşıyor?",
        "quantization_error": round(quantization_error, 5),
        "topographic_error": round(topographic_error, 5),
        "purity": None if purity is None else round(purity, 5),
        "interpretation": interpretation,
        "feature_spec": feature_spec,
        "x_columns": X_COLUMNS,
        "result_metrics": RESULT_METRICS,
        "cells": cells,
        "u_matrix": u_matrix,
        "assignments": assignments[:1000],
        "analysis_rows": build_analysis_rows(rows, bmu_assignments=assignments, selected_x_columns=selected_x),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SOM_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_analysis_rows(
    rows: list[dict[str, Any]],
    bmu_assignments: list[dict[str, Any]],
    selected_x_columns: list[str],
) -> list[dict[str, Any]]:
    """Persist compact per-record SOM rows for downstream class analysis."""
    exported: list[dict[str, Any]] = []
    for row, assignment in zip(rows, bmu_assignments):
        record = {
            "model_name": row.get("model_name"),
            "direction": row.get("direction"),
            "node": int(assignment.get("node") or 0),
            "cell_x": int(assignment.get("cell_x") or 0),
            "cell_y": int(assignment.get("cell_y") or 0),
            "distance": round(float(assignment.get("distance") or 0.0), 6),
        }
        for key in selected_x_columns:
            record[key] = row.get(key)
        for key in RESULT_METRICS:
            record[key] = row.get(key)
        exported.append(record)
    return exported


def optimize_som_grid(
    output_dir: Path,
    start_size: int = 3,
    max_size: int = 8,
    iterations: int = 800,
    result_metric: str = "critical_state",
    random_seed: int = 42,
    selected_x_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Try square SOM grids from 3x3 to max_size and keep the best one.

    The score balances quantization error, topographic error, and purity when
    the selected Y result is categorical. Lower score is better.
    """
    start_size = max(3, min(int(start_size), 20))
    max_size = max(3, min(int(max_size), 20))
    if start_size > max_size:
        raise ValueError("Min grid, maks grid değerinden büyük olamaz.")
    iterations = max(50, min(int(iterations), 20_000))
    candidates: list[dict[str, Any]] = []
    results_by_size: dict[tuple[int, int], dict[str, Any]] = {}
    for size in range(start_size, max_size + 1):
        result = train_som(
            output_dir,
            size,
            size,
            iterations,
            result_metric,
            random_seed,
            selected_x_columns,
        )
        results_by_size[(size, size)] = result
        candidates.append(
            {
                "width": size,
                "height": size,
                "quantization_error": result.get("quantization_error"),
                "topographic_error": result.get("topographic_error"),
                "purity": result.get("purity"),
                "filled_cells": sum(1 for cell in result.get("cells", []) if int(cell.get("hit_count") or 0) > 0),
                "sample_count": result.get("sample_count"),
            }
        )
    scored = score_som_grid_candidates(candidates)
    best = min(scored, key=lambda item: (float(item["optimization_score"]), int(item["width"])))
    best_result = results_by_size[(int(best["width"]), int(best["height"]))]
    best_result["optimization"] = {
        "available": True,
        "start_size": start_size,
        "max_size": max_size,
        "selected_width": best["width"],
        "selected_height": best["height"],
        "selected_score": best["optimization_score"],
        "score_note": (
            "Lower is better. Score uses normalized quantization error, "
            "topographic error, and purity loss for categorical Y metrics."
        ),
        "candidates": scored,
    }
    (output_dir / SOM_FILE).write_text(json.dumps(best_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_result


def score_som_grid_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return SOM grid candidates with normalized optimization scores."""
    q_values = [float(item["quantization_error"]) for item in candidates if is_number(item.get("quantization_error"))]
    t_values = [float(item["topographic_error"]) for item in candidates if is_number(item.get("topographic_error"))]
    q_min, q_max = (min(q_values), max(q_values)) if q_values else (0.0, 1.0)
    t_min, t_max = (min(t_values), max(t_values)) if t_values else (0.0, 1.0)

    def normalize(value: Any, low: float, high: float) -> float:
        number = to_float(value)
        if number is None:
            return 1.0
        if abs(high - low) < 1.0e-12:
            return 0.0
        return max(0.0, min(1.0, (number - low) / (high - low)))

    scored: list[dict[str, Any]] = []
    for item in candidates:
        q_norm = normalize(item.get("quantization_error"), q_min, q_max)
        t_norm = normalize(item.get("topographic_error"), t_min, t_max)
        purity = to_float(item.get("purity"))
        if purity is None:
            score = 0.60 * q_norm + 0.40 * t_norm
        else:
            score = 0.45 * q_norm + 0.35 * t_norm + 0.20 * (1.0 - max(0.0, min(1.0, purity)))
        scored.append({**item, "optimization_score": round(score, 6)})
    return sorted(scored, key=lambda item: (float(item["optimization_score"]), int(item["width"])))


def som_status(output_dir: Path) -> dict[str, Any]:
    """Return latest SOM payload and available column metadata."""
    path = output_dir / SOM_FILE
    base = {
        "x_columns": X_COLUMNS,
        "default_x_columns": list(DEFAULT_X_COLUMNS),
        "result_metrics": RESULT_METRICS,
        "target_leakage_warning": TARGET_LEAKAGE_MESSAGE,
    }
    if not path.exists():
        return {"available": False, "message": "Henüz SOM eğitilmedi.", **base}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "message": str(exc), **base}
    payload.update({key: value for key, value in base.items() if key not in payload})
    payload["x_columns"] = X_COLUMNS
    payload["default_x_columns"] = list(DEFAULT_X_COLUMNS)
    if isinstance(payload.get("selected_x_columns"), list):
        selected_x = validate_selected_columns(payload["selected_x_columns"], str(payload.get("result_metric") or payload.get("selected_y_column") or "critical_state"))
        payload["selected_x_columns"] = selected_x
        payload["selected_x_labels"] = [{"key": key, "label": X_COLUMNS[key]["label"]} for key in selected_x]
    payload["result_metrics"] = {**(payload.get("result_metrics") if isinstance(payload.get("result_metrics"), dict) else {}), **RESULT_METRICS}
    payload["target_leakage_warning"] = TARGET_LEAKAGE_MESSAGE
    return payload


def classify_columns(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Classify available columns as allowed X or post-analysis Y."""
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    return {
        "x_columns": [key for key in X_COLUMNS if key in available],
        "y_columns": [key for key in RESULT_METRICS if key in available],
    }


def get_allowed_x_columns(rows: list[dict[str, Any]] | None = None) -> list[str]:
    """Return columns allowed in SOM training."""
    if rows is None:
        return list(X_COLUMNS)
    return classify_columns(rows)["x_columns"]


def get_result_y_columns(rows: list[dict[str, Any]] | None = None) -> list[str]:
    """Return post-analysis result columns that may only color the map."""
    if rows is None:
        return list(RESULT_METRICS)
    return classify_columns(rows)["y_columns"]


def validate_selected_columns(selected_x_columns: list[str] | tuple[str, ...], selected_y_column: str) -> list[str]:
    """Validate user-selected X and Y columns and reject target leakage."""
    normalized_x = []
    for key in selected_x_columns:
        raw_key = Y_ALIASES.get(str(key), str(key))
        if raw_key in REMOVED_X_COLUMNS:
            continue
        normalized_x.append(X_ALIASES.get(raw_key, raw_key))
    invalid_y_in_x = [key for key in normalized_x if key in RESULT_METRICS]
    if invalid_y_in_x:
        raise ValueError(f"{TARGET_LEAKAGE_MESSAGE} Yasak değişken(ler): {', '.join(invalid_y_in_x)}")
    unknown = [key for key in normalized_x if key not in X_COLUMNS]
    if unknown:
        raise ValueError(f"SOM eğitiminde kullanılabilecek X değişkeni değil: {', '.join(unknown)}")
    if selected_y_column not in RESULT_METRICS:
        raise ValueError(f"Bilinmeyen Y sonuç değişkeni: {selected_y_column}")
    unique = list(dict.fromkeys(normalized_x))
    if not unique:
        raise ValueError("SOM eğitimi için en az bir X parametresi seçilmelidir.")
    return unique


def preprocess_som_input(som_input: list[dict[str, Any]], selected_x_columns: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
    """Encode X columns, fill missing values, and StandardScaler-scale all features."""
    raw_vectors: list[list[float]] = []
    encoded_features: list[dict[str, Any]] = []
    category_levels: dict[str, list[str]] = {}
    numeric_columns: list[str] = []

    for column in selected_x_columns:
        meta = X_COLUMNS[column]
        kind = meta["kind"]
        if kind in {"numeric", "ordinal"}:
            numeric_columns.append(column)
        elif kind == "nominal":
            levels = sorted({str(row.get(column) if row.get(column) not in (None, "") else "-") for row in som_input})
            category_levels[column] = levels
            encoded_features.extend({"source": column, "encoded": f"{column}={level}", "kind": "one_hot"} for level in levels)
    encoded_features = [{"source": column, "encoded": column, "kind": X_COLUMNS[column]["kind"]} for column in numeric_columns] + encoded_features

    for row in som_input:
        vector: list[float] = []
        for column in numeric_columns:
            vector.append(numeric_or_ordinal_value(column, row.get(column)))
        for column, levels in category_levels.items():
            value = str(row.get(column) if row.get(column) not in (None, "") else "-")
            vector.extend(1.0 if value == level else 0.0 for level in levels)
        raw_vectors.append(vector)

    means: list[float] = []
    stds: list[float] = []
    for index in range(len(raw_vectors[0])):
        values = [vector[index] for vector in raw_vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance) or 1.0
        means.append(mean)
        stds.append(std)

    scaled = [[(value - means[index]) / stds[index] for index, value in enumerate(vector)] for vector in raw_vectors]
    return scaled, {
        "selected_x_columns": selected_x_columns,
        "encoded_features": encoded_features,
        "category_levels": category_levels,
        "scaler": "StandardScaler",
        "means": [round(value, 6) for value in means],
        "stds": [round(value, 6) for value in stds],
    }


def train_som_weights(x_scaled: list[list[float]], width: int, height: int, iterations: int, random_seed: int) -> list[list[float]]:
    """Train SOM node weights from scaled X only."""
    best_weights: list[list[float]] | None = None
    best_score: tuple[float, float] | None = None
    restart_count = 5 if len(x_scaled) <= width * height * 3 else 3
    for restart in range(restart_count):
        weights = train_som_weights_once(x_scaled, width, height, iterations, random_seed + restart * 7919)
        score = (calculate_topographic_error(weights, x_scaled, width), calculate_quantization_error(weights, x_scaled))
        if best_score is None or score < best_score:
            best_score = score
            best_weights = weights
    return best_weights or train_som_weights_once(x_scaled, width, height, iterations, random_seed)


def train_som_weights_once(x_scaled: list[list[float]], width: int, height: int, iterations: int, random_seed: int) -> list[list[float]]:
    """Train one deterministic SOM candidate from scaled X only."""
    rng = random.Random(random_seed)
    weights = initialize_weights(x_scaled, width, height, rng)
    radius0 = max(width, height) / 2.0
    learning0 = 0.45
    for step in range(iterations):
        vector = x_scaled[rng.randrange(len(x_scaled))]
        bmu = best_matching_unit(vector, weights)
        progress = step / max(iterations - 1, 1)
        learning = learning0 * math.exp(-3.0 * progress)
        radius = max(0.65, radius0 * math.exp(-2.5 * progress))
        bmu_x = bmu % width
        bmu_y = bmu // width
        for node, weight in enumerate(weights):
            node_x = node % width
            node_y = node // width
            grid_dist_sq = float((node_x - bmu_x) ** 2 + (node_y - bmu_y) ** 2)
            influence = math.exp(-grid_dist_sq / (2.0 * radius * radius))
            if influence < 0.001:
                continue
            for dim, value in enumerate(vector):
                weight[dim] += learning * influence * (value - weight[dim])
    return weights


def assign_bmu(weights: list[list[float]], x_scaled: list[list[float]], width: int) -> list[dict[str, Any]]:
    """Assign each record to its best matching unit."""
    assignments = []
    exact_signature_nodes: dict[tuple[float, ...], int] = {}
    for vector in x_scaled:
        signature = tuple(round(value, 10) for value in vector)
        node = exact_signature_nodes.get(signature)
        if node is None:
            node = best_matching_unit(vector, weights)
            exact_signature_nodes[signature] = node
        assignments.append({"node": node, "cell_x": node % width, "cell_y": node // width, "distance": euclidean(vector, weights[node])})
    return assignments


def summarize_cells(
    rows: list[dict[str, Any]],
    bmu_assignments: list[dict[str, Any]],
    y_values: list[Any],
    selected_y_column: str,
    selected_x_columns: list[str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Summarize hit count, selected Y, and all Y distributions for each SOM cell."""
    cells: list[dict[str, Any]] = [
        {
            "x": node % width,
            "y": node // width,
            "hit_count": 0,
            "models": [],
            "result_summaries": {key: {"values": [], "counts": {}} for key in RESULT_METRICS},
            "x_summaries": {key: {"values": [], "counts": {}, "kind": X_COLUMNS[key]["kind"]} for key in selected_x_columns},
            "feature_means": defaultdict(float),
        }
        for node in range(width * height)
    ]
    for row, assignment, y_value in zip(rows, bmu_assignments, y_values):
        node = int(assignment["node"])
        cell = cells[node]
        cell["hit_count"] += 1
        cell["models"].append({"name": row["model_name"], "direction": row["direction"], "distance": round(float(assignment["distance"]), 4)})
        for metric_key, metric_meta in RESULT_METRICS.items():
            summary = cell["result_summaries"][metric_key]
            value = y_value if metric_key == selected_y_column else row.get(metric_key)
            if metric_meta["type"] == "number" and is_number(value):
                summary["values"].append(float(value))
            elif metric_meta["type"] == "category" and value not in (None, "", "-"):
                summary["counts"][str(value)] = int(summary["counts"].get(str(value), 0)) + 1
        for feature in selected_x_columns:
            meta = X_COLUMNS[feature]
            x_summary = cell["x_summaries"][feature]
            raw_value = row.get(feature)
            if meta["kind"] in {"numeric", "ordinal"}:
                numeric_value = numeric_or_ordinal_value(feature, raw_value)
                x_summary["values"].append(numeric_value)
                cell["feature_means"][feature] += numeric_value
            else:
                text = str(raw_value if raw_value not in (None, "") else "-")
                x_summary["counts"][text] = int(x_summary["counts"].get(text, 0)) + 1

    for cell in cells:
        hit_count = int(cell["hit_count"])
        cell["feature_means"] = {key: round(value / hit_count, 6) for key, value in cell["feature_means"].items()} if hit_count else {}
        for feature in selected_x_columns:
            finalize_x_summary(cell["x_summaries"][feature])
        for metric_key, metric_meta in RESULT_METRICS.items():
            finalize_result_summary(cell["result_summaries"][metric_key], metric_meta)
        selected = cell["result_summaries"][selected_y_column]
        cell["result_avg"] = selected.get("avg")
        cell["result_min"] = selected.get("min")
        cell["result_max"] = selected.get("max")
        cell["dominant_category"] = selected.get("dominant")
        cell["dominant_count"] = selected.get("dominant_count", 0)
        cell["dominant_ratio"] = round(float(cell["dominant_count"]) / hit_count, 5) if hit_count and selected.get("dominant_count") is not None else None
        cell["models"] = cell["models"][:12]
    return cells


def finalize_x_summary(summary: dict[str, Any]) -> None:
    """Convert raw selected-X cell values into numeric or categorical summaries."""
    if summary.get("kind") in {"numeric", "ordinal"}:
        values = [float(value) for value in summary.pop("values", []) if is_number(value)]
        summary.pop("counts", None)
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary.update({"avg": round(mean, 6), "min": round(min(values), 6), "max": round(max(values), 6), "std": round(math.sqrt(variance), 6), "count": len(values)})
        else:
            summary.update({"avg": None, "min": None, "max": None, "std": None, "count": 0})
        return

    counts = Counter(summary.pop("counts", {}))
    summary.pop("values", None)
    if counts:
        dominant, dominant_count = counts.most_common(1)[0]
        total = sum(counts.values())
        summary.update({"dominant": dominant, "dominant_count": dominant_count, "dominant_ratio": round(dominant_count / total, 5), "counts": dict(counts), "count": total})
    else:
        summary.update({"dominant": None, "dominant_count": 0, "dominant_ratio": None, "counts": {}, "count": 0})


def finalize_result_summary(summary: dict[str, Any], metric_meta: dict[str, Any]) -> None:
    """Convert raw per-cell Y lists into numerical or categorical summaries."""
    if metric_meta["type"] == "number":
        values = [float(value) for value in summary.pop("values", []) if is_number(value)]
        summary.pop("counts", None)
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary.update(
                {
                    "avg": round(mean, 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                    "std": round(math.sqrt(variance), 6),
                    "count": len(values),
                }
            )
        else:
            summary.update({"avg": None, "min": None, "max": None, "std": None, "count": 0})
        return

    counts = Counter(summary.pop("counts", {}))
    summary.pop("values", None)
    if counts:
        dominant, dominant_count = counts.most_common(1)[0]
        total = sum(counts.values())
        summary.update(
            {
                "dominant": dominant,
                "dominant_count": dominant_count,
                "dominant_ratio": round(dominant_count / total, 5),
                "counts": dict(counts),
                "count": total,
            }
        )
    else:
        summary.update({"dominant": None, "dominant_count": 0, "dominant_ratio": None, "counts": {}, "count": 0})


def calculate_quantization_error(weights: list[list[float]], x_scaled: list[list[float]]) -> float:
    """Average distance between X vectors and their BMU."""
    return sum(euclidean(vector, weights[best_matching_unit(vector, weights)]) for vector in x_scaled) / len(x_scaled)


def calculate_topographic_error(weights: list[list[float]], x_scaled: list[list[float]], width: int) -> float:
    """Fraction of records whose first and second BMUs are not adjacent."""
    if len(weights) < 2:
        return 0.0
    errors = 0
    for vector in x_scaled:
        ordered = sorted(range(len(weights)), key=lambda index: euclidean(vector, weights[index]))
        if not nodes_adjacent(ordered[0], ordered[1], width):
            errors += 1
    return errors / len(x_scaled)


def calculate_purity(cell_summaries: list[dict[str, Any]], selected_y_column: str) -> float | None:
    """Calculate category purity for the selected Y column."""
    if RESULT_METRICS[selected_y_column]["type"] != "category":
        return None
    dominant_sum = 0
    total = 0
    for cell in cell_summaries:
        summary = cell["result_summaries"][selected_y_column]
        dominant_sum += int(summary.get("dominant_count") or 0)
        total += int(summary.get("count") or 0)
    return dominant_sum / total if total else None


def attach_distinctive_features(cells: list[dict[str, Any]], rows: list[dict[str, Any]], selected_x_columns: list[str]) -> None:
    """Attach features whose cell mean differs most from global X means."""
    global_stats = compute_global_feature_stats(rows, selected_x_columns)
    for cell in cells:
        cell["distinctive_features"] = distinctive_features(cell.get("feature_means", {}), global_stats)


def compute_global_feature_stats(rows: list[dict[str, Any]], selected_x_columns: list[str]) -> dict[str, dict[str, float]]:
    """Return global mean and standard deviation for numeric selected X features."""
    stats: dict[str, dict[str, float]] = {}
    for key in selected_x_columns:
        if X_COLUMNS[key]["kind"] not in {"numeric", "ordinal"}:
            continue
        values = [numeric_or_ordinal_value(key, row.get(key)) for row in rows]
        mean = sum(values) / max(len(values), 1)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        stats[key] = {"mean": round(mean, 6), "std": round(math.sqrt(variance), 6), "min": round(min(values), 6), "max": round(max(values), 6)}
    return stats


def distinctive_features(feature_means: dict[str, Any], global_stats: dict[str, dict[str, float]], limit: int = 6) -> list[dict[str, Any]]:
    """Rank X features whose cell mean differs most from the global mean."""
    ranked: list[dict[str, Any]] = []
    for key, stats in global_stats.items():
        value = to_float(feature_means.get(key))
        mean = to_float(stats.get("mean"))
        std = to_float(stats.get("std")) or 0.0
        if value is None or mean is None:
            continue
        diff = value - mean
        if abs(diff) < 1.0e-12:
            continue
        score = abs(diff) / std if std > 1.0e-12 else abs(diff)
        ranked.append({"feature": key, "cell_mean": round(value, 6), "global_mean": round(mean, 6), "difference": round(diff, 6), "direction": "yüksek" if diff > 0.0 else "düşük", "score": round(score, 4)})
    return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)[:limit]


def build_som_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Build one SOM row per model and pushover direction from metadata JSON files."""
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*_metadata.json")):
        if any(part in {"ml_snapshots", "hinge_validation"} for part in path.parts):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("status", "")).lower() not in {"success", "successful", ""}:
            continue
        candidate = data.get("candidate", {}) if isinstance(data.get("candidate"), dict) else {}
        pushover = data.get("pushover", {}) if isinstance(data.get("pushover"), dict) else {}
        plastic = data.get("plastic_hinges", {}) if isinstance(data.get("plastic_hinges"), dict) else {}
        results = pushover.get("results", {}) if isinstance(pushover.get("results"), dict) else {}
        curves = results.get("curves", {}) if isinstance(results.get("curves"), dict) else {}
        story_drifts = results.get("story_drifts", {}) if isinstance(results.get("story_drifts"), dict) else {}
        hinge_results = plastic.get("results", {}) if isinstance(plastic.get("results"), dict) else {}
        summaries = hinge_results.get("summary_by_direction", {}) if isinstance(hinge_results.get("summary_by_direction"), dict) else {}
        directions = pushover.get("directions") if isinstance(pushover.get("directions"), list) else ["X", "Y"]
        for direction in directions:
            direction = str(direction).upper()
            row = base_feature_row(data, candidate, direction)
            curve = curves.get(direction, {}) if isinstance(curves.get(direction), dict) else {}
            summary = summaries.get(direction, {}) if isinstance(summaries.get(direction), dict) else {}
            drift = story_drift_for_direction(story_drifts, direction)
            critical_events = summary.get("critical_events", []) if isinstance(summary.get("critical_events"), list) else []
            first = summary.get("first_plastic_hinge", {}) if isinstance(summary.get("first_plastic_hinge"), dict) else {}
            critical = critical_events[0] if critical_events and isinstance(critical_events[0], dict) else {}
            first_location = classify_event_location(first, candidate)
            critical_location = classify_event_location(critical, candidate)
            counts = summary.get("state_counts", {}) if isinstance(summary.get("state_counts"), dict) else {}
            total_events = sum(int(value or 0) for value in counts.values())
            lscp_count = int(counts.get("LS-CP", 0) or 0)
            cpc_count = int(counts.get("CP-C", 0) or 0)
            fema = calculate_fema440_for_curve(
                curve,
                story_count=to_float(candidate.get("story_count")),
                soil_class=str(candidate.get("soil_class") or ""),
            )
            fema_values = fema_result_values(fema, curve)
            row.update(
                {
                    "peak_base_shear": to_float(curve.get("peak_base_shear_kn")),
                    "max_displacement": to_float(curve.get("final_control_displacement_m")),
                    "target_displacement": to_float(pushover.get("target_displacement_m")),
                    "ductility_ratio": safe_ratio(curve.get("final_control_displacement_m"), first.get("roof_displacement_m")),
                    "max_story_drift_ratio": to_float((drift.get("max_drift") or {}).get("drift_ratio") if isinstance(drift.get("max_drift"), dict) else None),
                    "max_rotation": max((to_float(event.get("plastic_rotation_rad")) or 0.0 for event in critical_events if isinstance(event, dict)), default=0.0),
                    "lscp_count": lscp_count,
                    "cpc_count": cpc_count,
                    "lscp_ratio": lscp_count / total_events if total_events else 0.0,
                    "cp_ratio": cpc_count / total_events if total_events else 0.0,
                    "first_hinge_type": str(first.get("element_type") or "-"),
                    "first_hinge_story": first_location.get("story"),
                    "first_hinge_plan_zone": first_location.get("plan_zone"),
                    "critical_element_type": str(critical.get("element_type") or "-"),
                    "critical_element_story": critical_location.get("story"),
                    "critical_element_plan_zone": critical_location.get("plan_zone"),
                    "critical_state": str(critical.get("hinge_state_level") or "-"),
                    "max_hinge_state": str(critical.get("hinge_state_level") or "-"),
                    "performance_level_at_target": str(critical.get("hinge_state_level") or "-"),
                    "collapse_mechanism": collapse_mechanism(first, critical),
                    "has_lscp": "yes" if lscp_count > 0 else "no",
                    "has_cp": "yes" if cpc_count > 0 else "no",
                    "damage_class": damage_class(str(critical.get("hinge_state_level") or "-")),
                    **fema_values,
                }
            )
            if any(row.get(key) not in (None, "", "-") for key in RESULT_METRICS):
                rows.append(row)
    return rows


def base_feature_row(data: dict[str, Any], candidate: dict[str, Any], direction: str) -> dict[str, Any]:
    """Extract pre-analysis X parameters for one model direction."""
    spans_x = [float(value) for value in candidate.get("spans_x", []) if is_number(value)]
    spans_y = [float(value) for value in candidate.get("spans_y", []) if is_number(value)]
    column = candidate.get("column_section", {}) if isinstance(candidate.get("column_section"), dict) else {}
    beam = candidate.get("beam_section", {}) if isinstance(candidate.get("beam_section"), dict) else {}
    story_count = to_float(candidate.get("story_count")) or 0.0
    story_height = to_float(candidate.get("story_height")) or 0.0
    column_width = to_float(column.get("width")) or 0.0
    column_depth = to_float(column.get("depth")) or 0.0
    beam_width = to_float(beam.get("width")) or 0.0
    beam_depth = to_float(beam.get("depth")) or 0.0
    concrete_class = str(candidate.get("concrete_class") or "C30")
    steel_class = str(candidate.get("steel_class") or "B420C")
    total_height = story_count * story_height
    column_i = column_width * column_depth**3 / 12.0 if column_width and column_depth else 0.0
    beam_i = beam_width * beam_depth**3 / 12.0 if beam_width and beam_depth else 0.0
    plan_area = (sum(spans_x) if spans_x else 0.0) * (sum(spans_y) if spans_y else 0.0)
    total_mass_proxy = plan_area * max(story_count, 1.0)
    direction_key = str(direction or "").upper()
    direction_spans = spans_y if direction_key == "Y" else spans_x
    direction_bay_count = to_float(candidate.get("y_bay_count" if direction_key == "Y" else "x_bay_count")) or 0.0
    wall_count = to_float(candidate.get("wall_count")) or 0.0
    wall_thickness = to_float(candidate.get("wall_thickness_m")) or 0.0
    wall_length = to_float(candidate.get("wall_length_m")) or 0.0
    return {
        "model_name": Path(str(data.get("model_path") or "")).stem or "model",
        "direction": direction,
        "story_count": story_count,
        "bay_count": direction_bay_count,
        "x_bay_count": to_float(candidate.get("x_bay_count")),
        "y_bay_count": to_float(candidate.get("y_bay_count")),
        "avg_span": sum(direction_spans) / len(direction_spans) if direction_spans else 0.0,
        "avg_span_x": sum(spans_x) / len(spans_x) if spans_x else 0.0,
        "avg_span_y": sum(spans_y) / len(spans_y) if spans_y else 0.0,
        "max_span_x": max(spans_x) if spans_x else 0.0,
        "max_span_y": max(spans_y) if spans_y else 0.0,
        "story_height": story_height,
        "total_height": total_height,
        "column_width": column_width,
        "column_depth": column_depth,
        "column_area": column_width * column_depth,
        "beam_width": beam_width,
        "beam_depth": beam_depth,
        "beam_area": beam_width * beam_depth,
        "column_beam_stiffness_ratio": safe_ratio(column_i, beam_i),
        "concrete_class": concrete_class,
        "steel_class": steel_class,
        "rho_col": to_float(candidate.get("rho_col")),
        "rho_beam_top": to_float(candidate.get("beam_top_ratio_support")),
        "rho_beam_bottom": to_float(candidate.get("beam_bottom_ratio_span")),
        "slab_thickness": to_float(candidate.get("slab_thickness_m")),
        "slab_rebar_ratio": to_float(candidate.get("slab_rebar_ratio")),
        "soil_class": str(candidate.get("soil_class") or "ZC"),
        "foundation_type": "raft",
        "subgrade_modulus": to_float(candidate.get("subgrade_modulus_kn_m3")),
        "raft_thickness": to_float(candidate.get("raft_thickness_m")),
        "raft_rebar_ratio": to_float(candidate.get("raft_rebar_ratio")),
        "target_drift": to_float(candidate.get("pushover_target_drift_ratio")),
        "has_shear_walls": "yes" if bool(candidate.get("has_shear_walls")) else "no",
        "wall_count": wall_count,
        "wall_area": wall_count * wall_thickness * wall_length,
        "wall_thickness": wall_thickness,
        "wall_length": wall_length,
        "wall_rebar_ratio": to_float(candidate.get("wall_rebar_ratio")) or 0.0,
        "initial_stiffness_proxy": safe_ratio(column_i * max(story_count, 1.0), max(total_height, 1.0)),
        "design_base_shear_ratio_proxy": safe_ratio(total_mass_proxy, max(to_float(candidate.get("subgrade_modulus_kn_m3")) or 1.0, 1.0)),
        "total_mass_proxy": total_mass_proxy,
    }


def fema_result_values(fema: dict[str, Any], curve: dict[str, Any]) -> dict[str, Any]:
    """Return FEMA 440 result fields for SOM coloring, never for SOM training."""
    if not isinstance(fema, dict) or not fema.get("available"):
        return {
            "fema_ductility_mu": None,
            "fema_beta_eff_percent": None,
            "fema_teff_t0_ratio": None,
            "fema_r_proxy": None,
            "fema_c1": None,
            "fema_target_displacement_proxy": None,
            "fema_target_capacity_ratio": None,
            "fema_capacity_status": "not_available",
        }
    equivalent = fema.get("equivalent_linearization", {}) if isinstance(fema.get("equivalent_linearization"), dict) else {}
    displacement = fema.get("displacement_modification", {}) if isinstance(fema.get("displacement_modification"), dict) else {}
    target = to_float(displacement.get("target_displacement_proxy_m"))
    capacity = to_float(curve.get("final_control_displacement_m"))
    ratio = safe_ratio(target, capacity) if target is not None and capacity is not None else None
    return {
        "fema_ductility_mu": to_float(equivalent.get("ductility_mu")),
        "fema_beta_eff_percent": to_float(equivalent.get("effective_damping_beta_percent")),
        "fema_teff_t0_ratio": to_float(equivalent.get("effective_period_ratio_teff_t0")),
        "fema_r_proxy": to_float(displacement.get("r_capacity_proxy")),
        "fema_c1": to_float(displacement.get("c1")),
        "fema_target_displacement_proxy": target,
        "fema_target_capacity_ratio": ratio,
        "fema_capacity_status": fema_capacity_status(ratio),
    }


def render_som_map(cell_summaries: list[dict[str, Any]], selected_y_column: str) -> dict[str, Any]:
    """Return a compact renderer payload for external callers."""
    return {"selected_y_column": selected_y_column, "cells": cell_summaries, "result_meta": RESULT_METRICS[selected_y_column]}


def som_interpretation(purity: float | None) -> str:
    """Return automatic engineering interpretation for SOM purity."""
    if purity is None:
        return "Seçilen Y sonucu sayısal olduğu için purity skoru hesaplanmadı; hücrelerde ortalama/min/maks/std değerleri yorumlanmalıdır."
    if purity >= 0.70:
        return "Yalnızca X parametreleriyle eğitilen SOM haritasında, seçilen Y sonucu hücreler içinde tutarlı şekilde kümelenmiştir. Bu durum, yapısal/tasarım parametreleri ile pushover performans seviyesi arasında anlamlı bir ilişki olduğunu gösterir."
    return "Yalnızca X parametreleriyle eğitilen SOM haritasında, seçilen Y sonucu hücreler içinde karışık dağılmıştır. Bu durum, mevcut X parametrelerinin seçilen performans sonucunu açıklamada yetersiz kalabileceğini veya daha fazla veri/özellik gerektiğini gösterir."


def numeric_or_ordinal_value(column: str, value: Any) -> float:
    """Convert numeric and ordinal X values to numbers before scaling."""
    mapping = X_COLUMNS[column].get("mapping")
    if mapping == "concrete":
        return concrete_fck(str(value or "C30"))
    if mapping == "steel":
        text = str(value or "B420C")
        return 500.0 if "500" in text else 420.0
    if mapping == "soil":
        return {"ZA": 1.0, "ZB": 2.0, "ZC": 3.0, "ZD": 4.0, "ZE": 5.0}.get(str(value or "ZC").upper(), 3.0)
    return to_float(value) or 0.0


def normalize_y_column(value: str) -> str:
    """Normalize UI/API Y column aliases."""
    key = Y_ALIASES.get(str(value), str(value))
    return key if key in RESULT_METRICS else "critical_state"


def nodes_adjacent(first: int, second: int, width: int) -> bool:
    """Return true when two SOM nodes are 4-neighbor adjacent or identical."""
    x1, y1 = first % width, first // width
    x2, y2 = second % width, second // width
    return abs(x1 - x2) + abs(y1 - y2) <= 1


def initialize_weights(vectors: list[list[float]], width: int, height: int, rng: random.Random) -> list[list[float]]:
    """Initialize SOM node weights from sampled rows with slight jitter."""
    weights: list[list[float]] = []
    for _ in range(width * height):
        source = list(rng.choice(vectors))
        weights.append([value + rng.uniform(-0.03, 0.03) for value in source])
    return weights


def best_matching_unit(vector: list[float], weights: list[list[float]]) -> int:
    """Return index of closest SOM node."""
    return min(range(len(weights)), key=lambda index: euclidean(vector, weights[index]))


def compute_u_matrix(weights: list[list[float]], width: int, height: int) -> list[dict[str, Any]]:
    """Compute average neighbor distance for each node."""
    rows: list[dict[str, Any]] = []
    for node, weight in enumerate(weights):
        x = node % width
        y = node // width
        distances = []
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                distances.append(euclidean(weight, weights[ny * width + nx]))
        rows.append({"x": x, "y": y, "value": round(sum(distances) / len(distances), 5) if distances else 0.0})
    return rows


def story_drift_for_direction(story_drifts: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return story drift summary for one direction."""
    if not isinstance(story_drifts, dict):
        return {}
    by_direction = story_drifts.get("by_direction", {}) if isinstance(story_drifts.get("by_direction"), dict) else {}
    value = by_direction.get(direction) or by_direction.get(direction.upper()) or by_direction.get(direction.lower())
    return value if isinstance(value, dict) else {}


def classify_event_location(event: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Derive story and rough plan zone for one hinge event from its element name."""
    spans_x = [float(value) for value in candidate.get("spans_x", []) if is_number(value)]
    spans_y = [float(value) for value in candidate.get("spans_y", []) if is_number(value)]
    x_coords = cumulative_coords(spans_x)
    y_coords = cumulative_coords(spans_y)
    parsed = parse_frame_element_name(str(event.get("element_name") or ""))
    story = parsed.get("story")
    x = resolve_plan_x(parsed, x_coords)
    y = resolve_plan_y(parsed, y_coords)
    if x is None or y is None:
        plan_zone = "-"
    else:
        x_boundary = is_boundary_coord(x, x_coords)
        y_boundary = is_boundary_coord(y, y_coords)
        if x_boundary or y_boundary:
            plan_zone = "edge"
        else:
            plan_zone = "middle"
    return {"story": story, "plan_zone": plan_zone}


def cumulative_coords(spans: list[float]) -> list[float]:
    """Convert span lengths to cumulative grid coordinates."""
    coords = [0.0]
    for span in spans:
        coords.append(round(coords[-1] + span, 6))
    return coords


def is_boundary_coord(value: Any, coords: list[float], tol: float = 0.15) -> bool:
    """Return true when coordinate is on the first or last grid line."""
    numeric = to_float(value)
    if numeric is None or len(coords) < 2:
        return False
    return abs(numeric - coords[0]) <= tol or abs(numeric - coords[-1]) <= tol


def parse_frame_element_name(name: str) -> dict[str, float | int | None]:
    """Parse SAP frame element names used by this generator."""
    text = str(name or "").strip()
    if not text:
        return {"story": None, "x": None, "y": None, "axis": None, "bay_index": None}
    parts = text.split("_")
    prefix = parts[0].upper() if parts else ""
    try:
        if prefix == "C" and len(parts) >= 4:
            return {
                "story": int(float(parts[1])),
                "x": to_float(parts[2]),
                "y": to_float(parts[3]),
                "axis": None,
                "bay_index": None,
            }
        if prefix == "BX" and len(parts) >= 4:
            story = int(float(parts[1]))
            bay_index = int(float(parts[2]))
            y_value = to_float(parts[3])
            return {"story": story, "x": None, "y": y_value, "axis": "x", "bay_index": bay_index}
        if prefix == "BY" and len(parts) >= 4:
            story = int(float(parts[1]))
            bay_index = int(float(parts[2]))
            x_value = to_float(parts[3])
            return {"story": story, "x": x_value, "y": None, "axis": "y", "bay_index": bay_index}
    except (TypeError, ValueError):
        return {"story": None, "x": None, "y": None, "axis": None, "bay_index": None}
    return {"story": None, "x": None, "y": None, "axis": None, "bay_index": None}


def resolve_plan_x(parsed: dict[str, Any], x_coords: list[float]) -> float | None:
    """Resolve event x coordinate from parsed element name."""
    if parsed.get("axis") == "x":
        return mid_bay_coord(x_coords, parsed.get("bay_index"))
    return to_float(parsed.get("x"))


def resolve_plan_y(parsed: dict[str, Any], y_coords: list[float]) -> float | None:
    """Resolve event y coordinate from parsed element name."""
    if parsed.get("axis") == "y":
        return mid_bay_coord(y_coords, parsed.get("bay_index"))
    return to_float(parsed.get("y"))


def mid_bay_coord(coords: list[float], bay_index: Any) -> float | None:
    """Return midpoint coordinate of one bay index."""
    if len(coords) < 2:
        return None
    try:
        index = int(bay_index)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= len(coords) - 1:
        return None
    return 0.5 * (coords[index] + coords[index + 1])


def collapse_mechanism(first: dict[str, Any], critical: dict[str, Any]) -> str:
    """Return a compact mechanism label from first and critical events."""
    first_type = str(first.get("element_type") or "unknown")
    critical_type = str(critical.get("element_type") or "unknown")
    if critical_type == "column":
        return "column_controlled"
    if critical_type == "beam":
        return "beam_controlled"
    return first_type if first_type != "unknown" else "unknown"


def damage_class(level: str) -> str:
    """Map hinge state level to a broad damage class."""
    if level in {"LS-CP", "CP-C", "C-D", "D-E", "beyond E"}:
        return "advanced_damage"
    if level in {"IO-LS"}:
        return "moderate_damage"
    if level in {"B-IO"}:
        return "limited_damage"
    return "no_or_low_damage"


def fema_capacity_status(ratio: float | None) -> str:
    """Classify FEMA target displacement proxy against available capacity."""
    if ratio is None or not math.isfinite(ratio):
        return "not_available"
    if ratio <= 0.90:
        return "capacity_ok"
    if ratio <= 1.10:
        return "near_capacity"
    return "capacity_exceeded"


def concrete_fck(concrete_class: str) -> float:
    """Return fck MPa from Cxx label."""
    try:
        return float(str(concrete_class).upper().replace("C", ""))
    except ValueError:
        return 30.0


def safe_ratio(numerator: Any, denominator: Any) -> float:
    """Return finite numerator/denominator or zero."""
    num = to_float(numerator) or 0.0
    den = to_float(denominator) or 0.0
    return num / den if abs(den) > 1.0e-12 else 0.0


def to_float(value: Any) -> float | None:
    """Parse a value as float."""
    try:
        if value in ("", None):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def is_number(value: Any) -> bool:
    """Return true when value can be interpreted as a finite number."""
    return to_float(value) is not None


def euclidean(left: list[float], right: list[float]) -> float:
    """Euclidean distance between two encoded vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
