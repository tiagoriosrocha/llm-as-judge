from __future__ import annotations

from pathlib import Path

import pandas as pd

from analise_utils import (
    MODEL_ORDER,
    filter_graphrag_without_ontology,
    filter_rag_without_ontology,
    load_summary,
    mean_metrics,
    run_table,
)


def build_table(avaliacao_dir: Path):
    display_columns = [
        "Correctness ↑",
        "Faithfulness ↑",
        "Completeness ↑",
    ]

    summary = load_summary(avaliacao_dir)
    rows = []

    for model in MODEL_ORDER:
        model_df = summary[summary["modelo"] == model]

        rag_metrics = mean_metrics(
            filter_rag_without_ontology(model_df),
            display_columns,
        )

        graphrag_metrics = mean_metrics(
            filter_graphrag_without_ontology(model_df),
            display_columns,
        )

        rows.append({"Modelo": model, "Método": "RAG", **rag_metrics})
        rows.append({"Modelo": model, "Método": "GraphRAG", **graphrag_metrics})

    return pd.DataFrame(
        rows,
        columns=["Modelo", "Método", *display_columns],
    )


if __name__ == "__main__":
    run_table(2, build_table)