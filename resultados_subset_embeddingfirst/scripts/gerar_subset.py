from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import gerar_resumo_avaliadas


SCRIPT_DIR = Path(__file__).resolve().parent
SUBSET_ROOT = SCRIPT_DIR.parent
SOURCE_PROJECT_ROOT = SUBSET_ROOT.parent
DEFAULT_OUTPUT_DIR = SUBSET_ROOT / "resultados_analises"
QUESTION_IDS_FILE = SCRIPT_DIR / "question_ids_subset.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera os resultados de analise para o subset de question_id."
    )
    parser.add_argument(
        "--source-avaliacao-dir",
        type=Path,
        default=SOURCE_PROJECT_ROOT / "avaliacao",
        help="Pasta de avaliacao original usada como fonte.",
    )
    parser.add_argument(
        "--source-resultados-dir",
        type=Path,
        default=SOURCE_PROJECT_ROOT / "resultados",
        help="Pasta de resultados original usada como fonte para matching/auditoria.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta de saida do subset. Padrao: ../resultados_analises/.",
    )
    parser.add_argument(
        "--question-ids-file",
        type=Path,
        default=QUESTION_IDS_FILE,
        help="Arquivo texto com um question_id por linha.",
    )
    return parser.parse_args()


def ensure_inside_subset(path: Path) -> Path:
    resolved = path.resolve()
    subset_root = SUBSET_ROOT.resolve()
    if resolved != subset_root and subset_root not in resolved.parents:
        raise ValueError(f"Caminho de saida fora de resultados_subset: {resolved}")
    return resolved


def load_question_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.append(normalize_question_id(value))

    duplicated = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
    if duplicated:
        raise ValueError(f"question_id duplicado no subset: {', '.join(duplicated)}")
    if not ids:
        raise ValueError(f"Nenhum question_id encontrado em {path}.")
    return ids


def normalize_question_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_question_id_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def drop_repeated_header_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or not {"arquivo_fonte", "tipo_resposta"}.issubset(dataframe.columns):
        return dataframe

    mask = (
        dataframe["arquivo_fonte"].astype(str).str.strip().eq("arquivo_fonte")
        & dataframe["tipo_resposta"].astype(str).str.strip().eq("tipo_resposta")
    )
    return dataframe.loc[~mask].copy() if mask.any() else dataframe


def filter_by_question_ids(
    dataframe: pd.DataFrame,
    question_ids: list[str],
    source_name: str,
    require_all_ids: bool,
) -> pd.DataFrame:
    if "question_id" not in dataframe.columns:
        raise ValueError(f"{source_name} nao possui a coluna question_id.")

    question_id_set = set(question_ids)
    normalized = normalize_question_id_series(dataframe["question_id"])
    filtered = dataframe.loc[normalized.isin(question_id_set)].copy()

    found_ids = set(normalize_question_id_series(filtered["question_id"]))
    missing_ids = sorted(question_id_set - found_ids, key=int)
    if missing_ids and require_all_ids:
        raise ValueError(
            f"{source_name} nao possui estes question_id do subset: "
            + ", ".join(missing_ids)
        )
    if missing_ids:
        print(
            f"Aviso: {source_name} nao contem {len(missing_ids)} question_id do subset.",
            flush=True,
        )

    return filtered


def write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def generate_subset_details(
    source_avaliacao_dir: Path,
    output_dir: Path,
    question_ids: list[str],
) -> pd.DataFrame:
    source_path = source_avaliacao_dir / "todas_avaliadas.csv"
    dataframe = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
    dataframe = drop_repeated_header_rows(dataframe)
    filtered = filter_by_question_ids(
        dataframe,
        question_ids,
        source_path.as_posix(),
        require_all_ids=True,
    )
    write_csv(filtered, output_dir / "todas_avaliadas.csv")
    return filtered


def generate_summary(output_dir: Path, details: pd.DataFrame) -> pd.DataFrame:
    summary = gerar_resumo_avaliadas.build_summary(details)
    gerar_resumo_avaliadas.write_summary(summary, output_dir / "resumo_avaliadas.csv")
    return summary


def reset_filtered_results_dir(output_dir: Path) -> Path:
    filtered_results_dir = ensure_inside_subset(output_dir / "resultados_filtrados")
    if filtered_results_dir.exists():
        shutil.rmtree(filtered_results_dir)
    filtered_results_dir.mkdir(parents=True, exist_ok=True)
    return filtered_results_dir


def copy_json_metadata(source_dir: Path, target_dir: Path) -> None:
    for json_path in source_dir.glob("*.json"):
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, target_dir / json_path.name)


def generate_filtered_result_files(
    source_resultados_dir: Path,
    output_dir: Path,
    question_ids: list[str],
) -> tuple[Path, int]:
    filtered_results_dir = reset_filtered_results_dir(output_dir)
    copied_files = 0

    for source_csv in sorted(source_resultados_dir.rglob("*.csv")):
        dataframe = pd.read_csv(source_csv, encoding="utf-8-sig", low_memory=False)
        dataframe = drop_repeated_header_rows(dataframe)
        if "question_id" not in dataframe.columns:
            continue

        filtered = filter_by_question_ids(
            dataframe,
            question_ids,
            source_csv.as_posix(),
            require_all_ids=False,
        )
        if filtered.empty:
            continue

        relative_path = source_csv.relative_to(source_resultados_dir)
        target_csv = filtered_results_dir / relative_path
        write_csv(filtered, target_csv)
        copy_json_metadata(source_csv.parent, target_csv.parent)
        copied_files += 1

    return filtered_results_dir, copied_files


def run_script(script_name: str, *args: object) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *[str(arg) for arg in args]]
    print("Executando:", " ".join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def run_generators(output_dir: Path, filtered_results_dir: Path) -> None:
    run_script("gerar_todas_tabelas.py", "--avaliacao-dir", output_dir)
    run_script(
        "gerar_tabela_9_audit_csv.py",
        "--results-dir",
        filtered_results_dir,
        "--output",
        output_dir / "tabelas" / "tabela_9_audit.csv",
    )
    run_script(
        "gerar_tabela_9_csv.py",
        "--results-dir",
        filtered_results_dir,
        "--output",
        output_dir / "tabelas" / "tabela_9.csv",
    )
    run_script(
        "gerar_comparacao_5_2_ontologia.py",
        "--antes-dir",
        filtered_results_dir / "petrobras-5-2-com-ontologia-antes-adaptacao",
        "--adaptado-dir",
        filtered_results_dir / "petrobras-5-2-com-ontologia",
        "--output",
        output_dir / "tabelas" / "comparacao_5_2_ontologia_antes_vs_adaptado.csv",
    )
    run_script(
        "auditar_resultados_avaliacao.py",
        "--resultados-dir",
        filtered_results_dir,
        "--avaliacao-dir",
        output_dir,
    )


def write_subset_manifest(
    output_dir: Path,
    question_ids: list[str],
    detail_rows: int,
    summary_rows: int,
    filtered_result_files: int,
) -> None:
    manifest_path = output_dir / "manifesto_subset.txt"
    lines = [
        "Subset de avaliacao",
        f"question_ids: {len(question_ids)}",
        f"linhas_todas_avaliadas: {detail_rows}",
        f"linhas_resumo_avaliadas: {summary_rows}",
        f"arquivos_resultados_filtrados: {filtered_result_files}",
        "",
        "question_ids_usados:",
        *question_ids,
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = ensure_inside_subset(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_ids = load_question_ids(args.question_ids_file)
    print(f"question_id no subset: {len(question_ids)}", flush=True)

    details = generate_subset_details(args.source_avaliacao_dir, output_dir, question_ids)
    summary = generate_summary(output_dir, details)
    filtered_results_dir, filtered_result_files = generate_filtered_result_files(
        args.source_resultados_dir,
        output_dir,
        question_ids,
    )

    run_generators(output_dir, filtered_results_dir)
    write_subset_manifest(
        output_dir,
        question_ids,
        detail_rows=len(details),
        summary_rows=len(summary),
        filtered_result_files=filtered_result_files,
    )

    print(f"Subset gerado em: {output_dir}")
    print(f"Linhas em todas_avaliadas.csv: {len(details)}")
    print(f"Linhas em resumo_avaliadas.csv: {len(summary)}")
    print(f"Arquivos de resultados filtrados: {filtered_result_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
