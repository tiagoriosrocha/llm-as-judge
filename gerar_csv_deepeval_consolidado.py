#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


DEFAULT_OUTPUT_FIELDS = [
    "arquivo_fonte",
    "tipo_resposta",
    "question_id",
    "question",
    "expected_question",
    "context",
    "llm_answer",
]

DATASET_REQUIRED_COLUMNS = ("id", "question", "context")
DATASET_EXPECTED_ANSWER_COLUMNS = (
    "answer_text",
    "expected_answer",
    "expected_question",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolida resultados CSV de execucoes RAG/GraphRAG em um unico arquivo "
            "com uma linha por resposta de modelo."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Pasta raiz do projeto. Padrao: pasta onde o script esta salvo.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("resultados"),
        help="Pasta onde estao as subpastas com os resultados. Padrao: resultados/.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/geopetrollm.csv"),
        help="CSV do dataset usado para buscar o campo de contexto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/todas_execucoes_deepeval.csv"),
        help="Arquivo CSV consolidado de saida.",
    )
    parser.add_argument(
        "--folders-config",
        type=Path,
        default=Path("pastas_contexto.csv"),
        help=(
            "CSV com os nomes das pastas de resultados e se cada uma recebe "
            "contexto ou nao."
        ),
    )
    return parser.parse_args()


def sniff_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    dialect = sniff_dialect(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        return list(reader)


def sanitize_text(value: str | None) -> str:
    if value is None:
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split())


def normalize_path(path: Path) -> str:
    return path.as_posix()


def resolve_under_root(root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else root / candidate


def parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    truthy = {"1", "true", "t", "sim", "s", "yes", "y"}
    falsy = {"0", "false", "f", "nao", "não", "n", "no"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError(
        "Valor invalido para incluir_contexto: "
        f"{raw_value!r}. Use true/false, sim/nao ou 1/0."
    )


def load_folder_context_config(config_path: Path) -> dict[str, bool]:
    folder_context: dict[str, bool] = {}
    for row in read_csv_rows(config_path):
        folder_name = (
            row.get("pasta_resultado")
            or row.get("pasta")
            or row.get("folder_name")
            or ""
        ).strip()
        if not folder_name or folder_name.startswith("#"):
            continue

        include_context_raw = (
            row.get("incluir_contexto")
            or row.get("usa_contexto")
            or row.get("com_contexto")
            or ""
        ).strip()
        if not include_context_raw:
            raise ValueError(
                f"A pasta {folder_name!r} nao possui valor para incluir_contexto."
            )
        if folder_name in folder_context:
            raise ValueError(f"A pasta {folder_name!r} esta duplicada na configuracao.")

        folder_context[folder_name] = parse_bool(include_context_raw)

    if not folder_context:
        raise ValueError(f"Nenhuma pasta valida foi encontrada em {config_path}.")

    return folder_context


def discover_result_csvs(
    root: Path,
    results_dir: Path,
    folder_context: dict[str, bool],
    folder_config_path: Path,
) -> list[tuple[Path, bool]]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Pasta de resultados nao encontrada: {results_dir}")

    existing_folders = {path.name for path in results_dir.iterdir() if path.is_dir()}
    configured_folders = set(folder_context)
    ignored_folders = sorted(existing_folders - configured_folders)
    if ignored_folders:
        sys.stderr.write(
            "Aviso: estas pastas existem em resultados/, mas nao estao listadas em "
            f"{normalize_path(folder_config_path.relative_to(root))} e serao ignoradas: "
            + ", ".join(ignored_folders)
            + "\n"
        )

    result_paths: list[tuple[Path, bool]] = []
    for folder_name, include_context in folder_context.items():
        folder_path = results_dir / folder_name
        if not folder_path.exists():
            sys.stderr.write(
                f"Aviso: a pasta configurada {folder_name!r} nao existe em "
                f"{normalize_path(results_dir.relative_to(root))} e sera ignorada.\n"
            )
            continue

        csv_paths = sorted(path for path in folder_path.rglob("*.csv") if path.is_file())
        if not csv_paths:
            sys.stderr.write(
                f"Aviso: nenhuma fonte CSV encontrada em {normalize_path(folder_path.relative_to(root))}.\n"
            )
            continue

        for path in csv_paths:
            result_paths.append((path, include_context))

    return result_paths


def load_dataset_index(dataset_path: Path) -> dict[str, dict[str, str]]:
    dataset_index: dict[str, dict[str, str]] = {}
    rows = read_csv_rows(dataset_path)
    if not rows:
        raise ValueError(f"Dataset vazio: {dataset_path}")

    fieldnames = set(rows[0])
    missing = [column for column in DATASET_REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            f"{dataset_path} nao possui as colunas obrigatorias: {', '.join(missing)}"
        )

    expected_answer_column = next(
        (
            column
            for column in DATASET_EXPECTED_ANSWER_COLUMNS
            if column in fieldnames
        ),
        None,
    )
    if expected_answer_column is None:
        raise ValueError(
            f"{dataset_path} precisa possuir uma destas colunas de resposta esperada: "
            + ", ".join(DATASET_EXPECTED_ANSWER_COLUMNS)
        )

    for row in rows:
        question_id = (row.get("id") or "").strip()
        if not question_id:
            continue
        if question_id in dataset_index:
            raise ValueError(f"ID duplicado no dataset: {question_id}")
        dataset_index[question_id] = {
            "question": sanitize_text(row.get("question")),
            "context": sanitize_text(row.get("context")),
            "answer_text": sanitize_text(row.get(expected_answer_column)),
        }
    return dataset_index


def build_output_rows(
    source_path: Path,
    root: Path,
    dataset_index: dict[str, dict[str, str]],
    attach_context: bool,
) -> list[dict[str, str]]:
    relative_source_path = source_path.relative_to(root)
    output_rows: list[dict[str, str]] = []

    for row in read_csv_rows(source_path):
        question_id = (row.get("question_id") or "").strip()
        dataset_row = dataset_index.get(question_id, {})
        context = dataset_row.get("context", "") if attach_context else ""
        expected_question = sanitize_text(
            row.get("expected_question")
            or row.get("expected_answer")
            or dataset_row.get("answer_text", "")
        )
        question = sanitize_text(row.get("question") or dataset_row.get("question", ""))

        base_row = {
            "arquivo_fonte": normalize_path(relative_source_path),
            "question_id": question_id,
            "question": question,
            "expected_question": expected_question,
            "context": context,
        }

        if "answers.rag_answer" in row or "answers.graphrag_answer" in row:
            output_rows.append(
                {
                    **base_row,
                    "tipo_resposta": "rag",
                    "llm_answer": sanitize_text(row.get("answers.rag_answer")),
                }
            )
            output_rows.append(
                {
                    **base_row,
                    "tipo_resposta": "graphrag",
                    "llm_answer": sanitize_text(row.get("answers.graphrag_answer")),
                }
            )
            continue

        if "answer" in row:
            output_rows.append(
                {
                    **base_row,
                    "tipo_resposta": "answer",
                    "llm_answer": sanitize_text(row.get("answer")),
                }
            )
            continue

        sys.stderr.write(
            f"Aviso: formato de CSV nao reconhecido em {normalize_path(relative_source_path)}\n"
        )

    return output_rows


def print_folder_context_summary(
    root: Path,
    results_dir: Path,
    folder_context: dict[str, bool],
) -> None:
    print("Configuracao de contexto por pasta:")
    for folder_name in folder_context:
        folder_path = results_dir / folder_name
        status = "com contexto" if folder_context[folder_name] else "sem contexto"
        suffix = "" if folder_path.exists() else " (pasta nao encontrada)"
        print(f"- {normalize_path(folder_path.relative_to(root))}: {status}{suffix}")


def main() -> int:
    args = parse_args()
    root = args.input_root.resolve()
    results_dir = resolve_under_root(root, args.results_dir).resolve()
    dataset_path = resolve_under_root(root, args.dataset).resolve()
    output_path = resolve_under_root(root, args.output).resolve()
    folder_config_path = resolve_under_root(root, args.folders_config).resolve()

    if not dataset_path.exists():
        sys.stderr.write(f"Erro: dataset nao encontrado em {dataset_path}\n")
        return 1
    if not folder_config_path.exists():
        sys.stderr.write(f"Erro: configuracao de pastas nao encontrada em {folder_config_path}\n")
        return 1

    try:
        dataset_index = load_dataset_index(dataset_path)
        folder_context = load_folder_context_config(folder_config_path)
        source_entries = discover_result_csvs(
            root,
            results_dir,
            folder_context,
            folder_config_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Erro: {exc}\n")
        return 1

    output_rows: list[dict[str, str]] = []

    for source_path, attach_context in source_entries:
        output_rows.extend(
            build_output_rows(
                source_path=source_path,
                root=root,
                dataset_index=dataset_index,
                attach_context=attach_context,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    attached_context_rows = sum(1 for row in output_rows if (row["context"] or "").strip())
    empty_context_rows = len(output_rows) - attached_context_rows
    expected_answer_rows = sum(
        1 for row in output_rows if (row["expected_question"] or "").strip()
    )
    empty_expected_answer_rows = len(output_rows) - expected_answer_rows
    print_folder_context_summary(root, results_dir, folder_context)
    print(f"Arquivo gerado: {output_path}")
    print(f"Configuracao usada: {folder_config_path}")
    print(f"CSVs processados: {len(source_entries)}")
    print(f"Linhas geradas: {len(output_rows)}")
    print(f"Linhas com contexto: {attached_context_rows}")
    print(f"Linhas sem contexto: {empty_context_rows}")
    print(f"Linhas com resposta esperada: {expected_answer_rows}")
    print(f"Linhas sem resposta esperada: {empty_expected_answer_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
