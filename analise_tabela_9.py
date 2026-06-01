from __future__ import annotations

from pathlib import Path

from analise_utils import build_matching_table, run_table


def build_table(avaliacao_dir: Path):
    return build_matching_table(avaliacao_dir)


if __name__ == "__main__":
    run_table(9, build_table)
