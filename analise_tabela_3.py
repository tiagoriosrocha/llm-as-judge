from __future__ import annotations

from pathlib import Path

from analise_utils import build_analysis_group_metrics, run_table


def build_table(avaliacao_dir: Path):
    return build_analysis_group_metrics(
        avaliacao_dir,
        [
            "Contextual Recall ↑",
            "Contextual Relevancy ↑",
            "Answer Relevancy ↑",
        ],
        by_model=False,
        include_delta_rows=False,
    )


if __name__ == "__main__":
    run_table(3, build_table)
