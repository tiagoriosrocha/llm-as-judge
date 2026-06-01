from __future__ import annotations

from pathlib import Path

from analise_utils import build_ontology_comparison, run_table


def build_table(avaliacao_dir: Path):
    return build_ontology_comparison(
        avaliacao_dir,
        [
            "Contextual Recall ↑",
            "Contextual Relevancy ↑",
            "Completeness ↑",
        ],
        include_delta=False,
    )


if __name__ == "__main__":
    run_table(7, build_table)
