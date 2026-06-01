from __future__ import annotations

from pathlib import Path

from analise_utils import build_ontology_comparison, run_table


def build_table(avaliacao_dir: Path):
    return build_ontology_comparison(
        avaliacao_dir,
        [
            "Ontology Alignment ↑",
            "Geological Adequacy ↑",
            "Faithfulness ↑",
        ],
        include_delta=True,
    )


if __name__ == "__main__":
    run_table(14, build_table)
