from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.utils.naming import parse_legacy_param_combo


@dataclass(frozen=True)
class EvaluationConfig:
    history_dir: Path
    predictions_dir: Path
    output_path: Path
    top_k: int = 3


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


def compute_top_k_stats(results_df: pd.DataFrame, predictions_dir: Path, top_k: int) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for _, row in results_df.iterrows():
        predictions_path = predictions_dir / row["preproc_id"] / row["model_name"] / f"{row['param_id']}.csv"
        if not predictions_path.exists():
            continue

        predictions_df = pd.read_csv(predictions_path)
        probability_columns = [col for col in predictions_df.columns if col.startswith("prob_class_")]
        if not probability_columns:
            continue

        probabilities = predictions_df[probability_columns].values
        top_k_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
        unique, counts = np.unique(top_k_indices, return_counts=True)
        class_freq = dict(zip(unique.tolist(), counts.tolist()))
        in_top_k = [truth in predicted for truth, predicted in zip(predictions_df["y_true"].values, top_k_indices)]
        stats.append(
            {
                "preproc_id": row["preproc_id"],
                "model_name": row["model_name"],
                "param_id": row["param_id"],
                "top_k_accuracy": float(np.mean(in_top_k)),
                "class_diversity": len(class_freq),
                "class_freq_json": json.dumps(class_freq, sort_keys=True),
            }
        )
    return stats


def build_final_results(results_df: pd.DataFrame, top_k_stats: list[dict[str, Any]]) -> pd.DataFrame:
    final_results = enrich_parameter_columns(results_df.copy())
    stats_df = pd.DataFrame(top_k_stats)

    if not stats_df.empty:
        final_results = final_results.merge(
            stats_df,
            on=["preproc_id", "model_name", "param_id"],
            how="left",
        )

    final_results["accuracy_improvement"] = final_results["best_val_acc"] - final_results.groupby("model_name")["best_val_acc"].transform("mean")
    final_results["convergence_speed"] = final_results["best_epoch"]
    final_results["preprocessing_type"] = final_results["preproc_id"].apply(lambda value: "combined" if "__" in value else "single")
    final_results["overall_rank"] = final_results["best_val_acc"].rank(ascending=False, method="dense")
    final_results["model_rank"] = final_results.groupby("model_name")["best_val_acc"].rank(ascending=False, method="dense")
    final_results["preproc_rank"] = final_results.groupby("preproc_id")["best_val_acc"].rank(ascending=False, method="dense")

    preferred_order = [
        "overall_rank",
        "preproc_id",
        "model_name",
        "param_id",
        "param_combo",
        "best_val_acc",
        "top_k_accuracy",
        "best_epoch",
        "convergence_speed",
        "fold",
        "preprocessing_type",
        "model_rank",
        "preproc_rank",
        "accuracy_improvement",
        "class_diversity",
        "class_freq_json",
        "param_json",
    ]
    remaining_columns = [column for column in final_results.columns if column not in preferred_order]
    ordered_columns = [column for column in preferred_order if column in final_results.columns] + remaining_columns
    return final_results[ordered_columns]


def generate_final_report(config: EvaluationConfig) -> tuple[pd.DataFrame, Path]:
    results_df = load_history_results(config.history_dir)
    top_k_stats = compute_top_k_stats(results_df, config.predictions_dir, config.top_k)
    final_results = build_final_results(results_df, top_k_stats)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(config.output_path, index=False)
    return final_results, config.output_path
