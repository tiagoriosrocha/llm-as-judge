from __future__ import annotations

from pathlib import Path

from analise_utils import build_ontology_gain_rows, run_table


def build_table(avaliacao_dir: Path):
    return build_ontology_gain_rows(
        avaliacao_dir,
        [
            "Correctness ↑",
            "Faithfulness ↑",
            "Geological Adequacy ↑",
            "Ontology Alignment ↑",
        ],
    )


if __name__ == "__main__":
    run_table(8, build_table)
