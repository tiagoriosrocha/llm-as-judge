from __future__ import annotations

from pathlib import Path

from analise_utils import build_ontology_delta_by_model, run_table


def build_table(avaliacao_dir: Path):
    return build_ontology_delta_by_model(
        avaliacao_dir,
        [
            "Contextual Recall ↑",
            "Contextual Relevancy ↑",
            "Completeness ↑",
        ],
        include_correctness_values=False,
    )


if __name__ == "__main__":
    run_table(12, build_table)
