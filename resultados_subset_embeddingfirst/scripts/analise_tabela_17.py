from __future__ import annotations

from pathlib import Path

import pandas as pd

from analise_utils import (
    MODEL_ORDER,
    filter_graphrag_with_ontology,
    filter_graphrag_without_ontology,
    filter_rag_without_ontology,
    load_summary,
    run_table,
)


CLASSIC_METRIC_COLUMNS = {
    "ROUGE-L F1 ↑": "rougeL_fmeasure_mean",
    "Exact Match ↑": "exact_match_mean",
    "Token F1 ↑": "token_f1_mean",
}


def mean_classic_metrics(dataframe: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for display_name, column_name in CLASSIC_METRIC_COLUMNS.items():
        if column_name not in dataframe.columns:
            raise ValueError(f"Coluna ausente no resumo: {column_name}")
        metrics[display_name] = pd.to_numeric(dataframe[column_name], errors="coerce").mean()
    return metrics


def add_row(rows: list[dict[str, object]], model: str, method: str, dataframe: pd.DataFrame):
    if dataframe.empty:
        values = {metric: float("nan") for metric in CLASSIC_METRIC_COLUMNS}
    else:
        values = mean_classic_metrics(dataframe)

    rows.append(
        {
            "Modelo": model,
            "Método": method,
            **values,
        }
    )


def build_table(avaliacao_dir: Path):
    summary = load_summary(avaliacao_dir)

    rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        model_df = summary[summary["modelo"] == model]

        add_row(
            rows,
            model,
            "RAG",
            filter_rag_without_ontology(model_df),
        )
        add_row(
            rows,
            model,
            "GraphRAG",
            filter_graphrag_without_ontology(model_df),
        )
        add_row(
            rows,
            model,
            "GraphRAG + Ontologia",
            filter_graphrag_with_ontology(model_df),
        )

    return pd.DataFrame(
        rows,
        columns=["Modelo", "Método", *CLASSIC_METRIC_COLUMNS.keys()],
    )


if __name__ == "__main__":
    run_table(17, build_table)