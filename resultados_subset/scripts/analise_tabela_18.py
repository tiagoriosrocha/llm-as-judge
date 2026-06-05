from __future__ import annotations

from pathlib import Path

import pandas as pd

from analise_utils import (
    filter_graphrag_with_ontology,
    filter_graphrag_without_ontology,
    filter_rag_without_ontology,
    load_summary,
    run_table,
    safe_delta_percent,
)


ROUGE_L_COLUMNS = {
    "ROUGE-L Precision ↑": "rougeL_precision_mean",
    "ROUGE-L Recall ↑": "rougeL_recall_mean",
    "ROUGE-L F1 ↑": "rougeL_fmeasure_mean",
}


def mean_rouge_l(dataframe: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for display_name, column_name in ROUGE_L_COLUMNS.items():
        if column_name not in dataframe.columns:
            raise ValueError(f"Coluna ausente no resumo: {column_name}")
        metrics[display_name] = pd.to_numeric(dataframe[column_name], errors="coerce").mean()
    return metrics


def build_table(avaliacao_dir: Path):
    summary = load_summary(avaliacao_dir)

    rag_metrics = mean_rouge_l(filter_rag_without_ontology(summary))
    graphrag_metrics = mean_rouge_l(filter_graphrag_without_ontology(summary))
    ontology_metrics = mean_rouge_l(filter_graphrag_with_ontology(summary))

    delta_graphrag = {"Configuração": "Δ GraphRAG vs RAG (%)"}
    delta_ontology = {"Configuração": "Δ Ontologia vs GraphRAG (%)"}

    for metric in ROUGE_L_COLUMNS:
        delta_graphrag[metric] = safe_delta_percent(
            graphrag_metrics[metric],
            rag_metrics[metric],
        )
        delta_ontology[metric] = safe_delta_percent(
            ontology_metrics[metric],
            graphrag_metrics[metric],
        )

    rows = [
        {"Configuração": "RAG", **rag_metrics},
        {"Configuração": "GraphRAG", **graphrag_metrics},
        {"Configuração": "GraphRAG + Ontologia", **ontology_metrics},
        delta_graphrag,
        delta_ontology,
    ]

    return pd.DataFrame(rows, columns=["Configuração", *ROUGE_L_COLUMNS.keys()])


if __name__ == "__main__":
    run_table(18, build_table)