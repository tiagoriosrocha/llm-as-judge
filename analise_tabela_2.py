from __future__ import annotations

from pathlib import Path

from analise_utils import build_rag_vs_graphrag_by_model, run_table


def build_table(avaliacao_dir: Path):
    return build_rag_vs_graphrag_by_model(avaliacao_dir)


if __name__ == "__main__":
    run_table(2, build_table)
