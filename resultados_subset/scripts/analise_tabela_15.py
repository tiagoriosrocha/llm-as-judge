from __future__ import annotations

from pathlib import Path

import pandas as pd

from analise_utils import (
    filter_graphrag_without_ontology,
    filter_rag_without_ontology,
    load_summary,
    run_table,
    safe_delta_percent,
)


CLASSIC_METRIC_COLUMNS = {
    "ROUGE-1 F1 ↑": "rouge1_fmeasure_mean",
    "ROUGE-2 F1 ↑": "rouge2_fmeasure_mean",
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


def build_table(avaliacao_dir: Path):
    summary = load_summary(avaliacao_dir)

    rag_metrics = mean_classic_metrics(filter_rag_without_ontology(summary))
    graphrag_metrics = mean_classic_metrics(filter_graphrag_without_ontology(summary))

    delta = {"Método": "Δ (%)"}
    for metric in CLASSIC_METRIC_COLUMNS:
        delta[metric] = safe_delta_percent(
            graphrag_metrics[metric],
            rag_metrics[metric],
        )

    rows = [
        {"Método": "RAG", **rag_metrics},
        {"Método": "GraphRAG", **graphrag_metrics},
        delta,
    ]

    return pd.DataFrame(rows, columns=["Método", *CLASSIC_METRIC_COLUMNS.keys()])


if __name__ == "__main__":
    run_table(15, build_table)