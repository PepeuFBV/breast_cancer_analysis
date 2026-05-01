from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from pipeline.data.constants import VALID_LABELS
from pipeline.utils.naming import parse_legacy_param_combo


@dataclass(frozen=True)
class EvaluationConfig:
    history_dir: Path
    predictions_dir: Path
    output_path: Path
    details_dir: Path | None = None
    top_k: int = 3

    @property
    def resolved_details_dir(self) -> Path:
        return self.details_dir or self.output_path.parent / "evaluation_details"


def load_history_results(history_dir: Path) -> pd.DataFrame:
    all_results: list[pd.DataFrame] = []
    if not history_dir.exists():
        raise FileNotFoundError(f"History directory does not exist: {history_dir}")

    for preproc_path in sorted(history_dir.iterdir()):
        if not preproc_path.is_dir():
            continue
        for model_path in sorted(preproc_path.iterdir()):
            if not model_path.is_dir():
                continue
            for csv_file in sorted(model_path.glob("*.csv")):
                dataframe = pd.read_csv(csv_file)
                dataframe["preproc_id"] = dataframe.get("preproc_id", preproc_path.name)
                dataframe["model_name"] = dataframe.get("model_name", model_path.name)
                if "param_id" not in dataframe.columns:
                    dataframe["param_id"] = csv_file.stem.replace("history_", "")
                all_results.append(dataframe)

    if not all_results:
        raise ValueError(f"No history CSV files found under {history_dir}")

    return pd.concat(all_results, ignore_index=True)


def _params_from_row(row: pd.Series) -> dict[str, Any]:
    if "param_json" in row and pd.notna(row["param_json"]):
        return json.loads(row["param_json"])
    if "param_combo" in row and pd.notna(row["param_combo"]):
        return parse_legacy_param_combo(str(row["param_combo"]))
    return {}


def enrich_parameter_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    param_records = results_df.apply(_params_from_row, axis=1)
    param_df = pd.json_normalize(param_records)
    if param_df.empty:
        return results_df
    return pd.concat([results_df.reset_index(drop=True), param_df.reset_index(drop=True)], axis=1)


def _resolve_probability_columns(predictions_df: pd.DataFrame) -> list[str]:
    columns = [column for column in predictions_df.columns if column.startswith("prob_class_")]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[-1]))


def _label_name(label_index: int) -> str:
    if 0 <= label_index < len(VALID_LABELS):
        return VALID_LABELS[label_index]
    return str(label_index)


def _compute_top_k_summary(probabilities: np.ndarray, y_true: np.ndarray, top_k: int) -> dict[str, Any]:
    if probabilities.size == 0:
        return {
            "top_k_accuracy": np.nan,
            "class_diversity": 0,
            "class_freq_json": json.dumps({}, sort_keys=True),
        }

    effective_top_k = max(1, min(top_k, probabilities.shape[1]))
    top_k_indices = np.argsort(probabilities, axis=1)[:, -effective_top_k:]
    unique, counts = np.unique(top_k_indices, return_counts=True)
    class_freq = dict(zip(unique.tolist(), counts.tolist()))
    in_top_k = [truth in predicted for truth, predicted in zip(y_true, top_k_indices)]
    return {
        "top_k_accuracy": float(np.mean(in_top_k)),
        "class_diversity": len(class_freq),
        "class_freq_json": json.dumps(class_freq, sort_keys=True),
    }


def _compute_multiclass_auc(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    present_labels = sorted(np.unique(y_true).tolist())
    if len(present_labels) < 2:
        return np.nan, np.nan

    if len(present_labels) == 2:
        positive_label = present_labels[1]
        binary_target = (y_true == positive_label).astype(int)
        try:
            auc_value = float(roc_auc_score(binary_target, probabilities[:, positive_label]))
            return auc_value, auc_value
        except ValueError:
            return np.nan, np.nan

    binarized = label_binarize(y_true, classes=present_labels)
    relevant_probabilities = probabilities[:, present_labels]
    try:
        macro_auc = float(
            roc_auc_score(
                binarized,
                relevant_probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
        weighted_auc = float(
            roc_auc_score(
                binarized,
                relevant_probabilities,
                average="weighted",
                multi_class="ovr",
            )
        )
        return macro_auc, weighted_auc
    except ValueError:
        return np.nan, np.nan


def _build_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    labels = list(range(probabilities.shape[1]))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    rows: list[dict[str, Any]] = []
    for label in labels:
        metrics = report.get(
            str(label),
            {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0.0},
        )
        binary_target = (y_true == label).astype(int)
        per_class_auc = np.nan
        if binary_target.min() != binary_target.max():
            try:
                per_class_auc = float(roc_auc_score(binary_target, probabilities[:, label]))
            except ValueError:
                per_class_auc = np.nan

        rows.append(
            {
                "class_index": label,
                "class_label": _label_name(label),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1-score"]),
                "support": float(metrics["support"]),
                "roc_auc_ovr": per_class_auc,
            }
        )

    return pd.DataFrame(rows)


def _build_confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, *, normalize: bool) -> pd.DataFrame:
    labels = list(range(len(VALID_LABELS)))
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=("true" if normalize else None),
    )
    named_labels = [_label_name(label) for label in labels]
    return pd.DataFrame(matrix, index=named_labels, columns=named_labels)


def _write_evaluation_details(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    classification_report_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    normalized_confusion_df: pd.DataFrame,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "metrics_summary.json"
    classification_path = run_dir / "classification_report.csv"
    confusion_path = run_dir / "confusion_matrix.csv"
    normalized_confusion_path = run_dir / "confusion_matrix_normalized.csv"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    classification_report_df.to_csv(classification_path, index=False)
    confusion_df.to_csv(confusion_path, index=True)
    normalized_confusion_df.to_csv(normalized_confusion_path, index=True)

    return {
        "metrics_summary_path": str(summary_path),
        "classification_report_path": str(classification_path),
        "confusion_matrix_path": str(confusion_path),
        "confusion_matrix_normalized_path": str(normalized_confusion_path),
    }


def compute_run_evaluation_stats(
    results_df: pd.DataFrame,
    predictions_dir: Path,
    details_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for _, row in results_df.iterrows():
        predictions_path = predictions_dir / row["preproc_id"] / row["model_name"] / f"{row['param_id']}.csv"
        if not predictions_path.exists():
            continue

        predictions_df = pd.read_csv(predictions_path)
        probability_columns = _resolve_probability_columns(predictions_df)
        if not probability_columns:
            continue

        probabilities = predictions_df[probability_columns].values
        y_true = predictions_df["y_true"].astype(int).to_numpy()
        y_pred = predictions_df["y_pred"].astype(int).to_numpy()

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        macro_auc, weighted_auc = _compute_multiclass_auc(y_true, probabilities)
        top_k_summary = _compute_top_k_summary(probabilities, y_true, top_k)
        classification_report_df = _build_classification_report_df(y_true, y_pred, probabilities)
        confusion_df = _build_confusion_matrix_df(y_true, y_pred, normalize=False)
        normalized_confusion_df = _build_confusion_matrix_df(y_true, y_pred, normalize=True)

        summary = {
            "preproc_id": row["preproc_id"],
            "model_name": row["model_name"],
            "param_id": row["param_id"],
            "support_total": int(len(predictions_df)),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "roc_auc_ovr_macro": macro_auc,
            "roc_auc_ovr_weighted": weighted_auc,
            **top_k_summary,
        }
        detail_paths = _write_evaluation_details(
            details_dir / row["preproc_id"] / row["model_name"] / row["param_id"],
            summary=summary,
            classification_report_df=classification_report_df,
            confusion_df=confusion_df,
            normalized_confusion_df=normalized_confusion_df,
        )
        stats.append({**summary, **detail_paths})

    return stats


def build_final_results(results_df: pd.DataFrame, evaluation_stats: list[dict[str, Any]]) -> pd.DataFrame:
    final_results = enrich_parameter_columns(results_df.copy())
    stats_df = pd.DataFrame(evaluation_stats)

    if not stats_df.empty:
        final_results = final_results.merge(
            stats_df,
            on=["preproc_id", "model_name", "param_id"],
            how="left",
        )

    if "f1_macro" not in final_results.columns:
        final_results["f1_macro"] = np.nan
    if "test_accuracy" not in final_results.columns:
        final_results["test_accuracy"] = np.nan

    ranking_metric = final_results["f1_macro"].fillna(final_results["best_val_acc"])
    final_results["accuracy_improvement"] = final_results["test_accuracy"] - final_results.groupby("model_name")["test_accuracy"].transform("mean")
    final_results["f1_macro_improvement"] = ranking_metric - ranking_metric.groupby(final_results["model_name"]).transform("mean")
    final_results["convergence_speed"] = final_results["best_epoch"]
    final_results["preprocessing_type"] = final_results["preproc_id"].apply(lambda value: "combined" if "__" in value else "single")
    final_results["overall_rank"] = ranking_metric.rank(ascending=False, method="dense")
    final_results["validation_rank"] = final_results["best_val_acc"].rank(ascending=False, method="dense")
    final_results["model_rank"] = ranking_metric.groupby(final_results["model_name"]).rank(ascending=False, method="dense")
    final_results["preproc_rank"] = ranking_metric.groupby(final_results["preproc_id"]).rank(ascending=False, method="dense")

    preferred_order = [
        "overall_rank",
        "validation_rank",
        "preproc_id",
        "model_name",
        "param_id",
        "param_combo",
        "test_accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "roc_auc_ovr_weighted",
        "best_val_acc",
        "top_k_accuracy",
        "best_epoch",
        "convergence_speed",
        "fold",
        "selection_strategy",
        "preprocessing_type",
        "model_rank",
        "preproc_rank",
        "accuracy_improvement",
        "f1_macro_improvement",
        "class_diversity",
        "class_freq_json",
        "support_total",
        "classification_report_path",
        "confusion_matrix_path",
        "confusion_matrix_normalized_path",
        "metrics_summary_path",
        "param_json",
    ]
    remaining_columns = [column for column in final_results.columns if column not in preferred_order]
    ordered_columns = [column for column in preferred_order if column in final_results.columns] + remaining_columns
    return final_results[ordered_columns]


def generate_final_report(config: EvaluationConfig) -> tuple[pd.DataFrame, Path]:
    results_df = load_history_results(config.history_dir)
    details_dir = config.resolved_details_dir
    evaluation_stats = compute_run_evaluation_stats(
        results_df,
        config.predictions_dir,
        details_dir,
        config.top_k,
    )
    final_results = build_final_results(results_df, evaluation_stats)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(config.output_path, index=False)
    return final_results, config.output_path
