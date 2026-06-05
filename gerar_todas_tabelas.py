from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa analise_tabela_0.py ate analise_tabela_14.py."
    )
    parser.add_argument(
        "--avaliacao-dir",
        type=Path,
        default=PROJECT_ROOT / "avaliacao",
        help="Pasta com os CSVs de avaliacao. Padrao: avaliacao/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for table_number in range(0, 19):
        script_path = PROJECT_ROOT / f"analise_tabela_{table_number}.py"
        print(f"Gerando tabela {table_number}...", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--avaliacao-dir",
                str(args.avaliacao_dir),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            print(f"Falha ao gerar tabela {table_number}.")
            return completed.returncode
    print("Todas as tabelas foram geradas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
