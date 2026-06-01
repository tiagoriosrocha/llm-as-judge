from __future__ import annotations

from pathlib import Path

from analise_utils import build_analysis_group_metrics, run_table


def build_table(avaliacao_dir: Path):
    return build_analysis_group_metrics(
        avaliacao_dir,
        [
            "Correctness ↑",
            "Faithfulness ↑",
            "Completeness ↑",
        ],
        by_model=True,
    )


if __name__ == "__main__":
    run_table(2, build_table)
