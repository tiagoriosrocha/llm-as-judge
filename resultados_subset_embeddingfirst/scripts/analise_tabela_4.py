from __future__ import annotations

from pathlib import Path

from analise_utils import build_ontology_comparison, run_table


def build_table(avaliacao_dir: Path):
    return build_ontology_comparison(
        avaliacao_dir,
        [
            "Correctness ↑",
            "Faithfulness ↑",
            "Geological Adequacy ↑",
            "Ontology Alignment ↑",
        ],
        include_delta=True,
    )


if __name__ == "__main__":
    run_table(4, build_table)