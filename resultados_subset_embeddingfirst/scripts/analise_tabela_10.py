from __future__ import annotations

from pathlib import Path

from analise_utils import build_ontology_delta_by_model, run_table


def build_table(avaliacao_dir: Path):
    return build_ontology_delta_by_model(
        avaliacao_dir,
        [
            "Correctness ↑",
            "Faithfulness ↑",
            "Geological Adequacy ↑",
            "Ontology Alignment ↑",
        ],
        include_correctness_values=True,
    )


if __name__ == "__main__":
    run_table(10, build_table)
