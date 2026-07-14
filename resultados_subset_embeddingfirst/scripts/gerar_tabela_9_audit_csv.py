from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SUBSET_ROOT = PROJECT_ROOT.parent
DEFAULT_ANALISES_DIR = SUBSET_ROOT / "resultados_analises"
MODEL_ORDER = ["GPT-4o", "GPT-4.1", "GPT-5-mini", "GPT-5.2", "GPT-5.4", "GPT-o3"]
REQUIRED_COLUMNS = (
    "total_graph_data_nodes",
    "total_edges",
    "total_ontology_valid_true",
)
ADAPTED_MATCHING_FOLDERS = {
    "petrobras-5-2-com-ontologia-embeddingfirst",
}


@dataclass
class BatchMetadata:
    raw_model: str = ""
    processing_type: str = ""
    ontology_id: str = ""
    batch_task_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um CSV de auditoria da Tabela 9 com nodes, edges e matching "
            "por arquivo/modelo."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_ANALISES_DIR / "resultados_filtrados",
        help="Pasta com os resultados filtrados do subset. Padrao: ../resultados_analises/resultados_filtrados/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANALISES_DIR / "tabelas" / "tabela_9_audit.csv",
        help="Caminho do CSV final de auditoria.",
    )
    return parser.parse_args()


def detect_model(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "petrobras-4-1" in lowered:
        return "GPT-4.1"
    if "petrobras-5-2" in lowered:
        return "GPT-5.2"
    if "petrobras-5-4" in lowered:
        return "GPT-5.4"
    if "petrobras-4-0" in lowered:
        return "GPT-4o"
    if "petrobras-0-3" in lowered:
        return "GPT-o3"
    if "petrobras-5-mini" in lowered:
        return "GPT-5-mini"
    return "Desconhecido"


def detect_graph_configuration(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "com-ontologia" in lowered:
        return "Com ontologia"
    if "sem-ontologia" in lowered:
        return "Sem ontologia"
    return "Desconhecida"


def classify_matching_version(path: Path) -> str:
    lowered = path.as_posix().lower()

    original_markers = (
        "antes-adaptacao",
        "antes_adaptacao",
        "matching-original",
        "matching_original",
        "before",
        "original",
    )
    if any(marker in lowered for marker in original_markers):
        return "Matching original"

    path_parts = {part.lower() for part in path.parts}
    if ADAPTED_MATCHING_FOLDERS & path_parts:
        return "Matching adaptado"

    adapted_markers = (
        "adaptatado",
        "adaptado",
        "adaptacao",
        "adaptation",
        "adapt",
    )
    if any(marker in lowered for marker in adapted_markers):
        return "Matching adaptado"

    return "Nao classificado"


def load_batch_metadata(csv_path: Path) -> BatchMetadata:
    json_candidates = sorted(csv_path.parent.glob("*.json"))
    if not json_candidates:
        return BatchMetadata()

    try:
        data = json.loads(json_candidates[0].read_text(encoding="utf-8"))
    except Exception:
        return BatchMetadata()

    metadata = data.get("batch_metadata", {})
    return BatchMetadata(
        raw_model=str(metadata.get("model") or ""),
        processing_type=str(metadata.get("processing_type") or ""),
        ontology_id=str(metadata.get("ontology_id") or ""),
        batch_task_id=str(metadata.get("batch_task_id") or ""),
    )


def require_columns(fieldnames: list[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV vazio ou sem cabecalho: {path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            f"{path} nao possui as colunas obrigatorias: {', '.join(missing)}"
        )


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator * 100


def format_number(value: object) -> object:
    if isinstance(value, int):
        return value
    if not isinstance(value, float):
        return value
    if math.isnan(value):
        return ""
    rounded = round(value, 4)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def build_audit_row(csv_path: Path) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, csv_path)

        total_rows = 0
        total_nodes = 0
        total_edges = 0
        total_matches = 0

        for row in reader:
            total_rows += 1
            total_nodes += int(float(row["total_graph_data_nodes"]))
            total_edges += int(float(row["total_edges"]))
            total_matches += int(float(row["total_ontology_valid_true"]))

    metadata = load_batch_metadata(csv_path)
    model = detect_model(csv_path)
    graph_configuration = detect_graph_configuration(csv_path)
    matching_version = classify_matching_version(csv_path)

    return {
        "Modelo": model,
        "Configuração": graph_configuration,
        "Versão do Matching": matching_version,
        "Pasta de resultados": csv_path.parent.name,
        "Arquivo CSV": csv_path.as_posix(),
        "Perguntas": total_rows,
        "Nodes totais": total_nodes,
        "Edges totais": total_edges,
        "Matches válidos totais": total_matches,
        "Nodes médios por pergunta": total_nodes / total_rows if total_rows else math.nan,
        "Edges médios por pergunta": total_edges / total_rows if total_rows else math.nan,
        "Matches válidos médios por pergunta": (
            total_matches / total_rows if total_rows else math.nan
        ),
        "Taxa de Matching (%)": safe_rate(total_matches, total_nodes),
        "Modelo batch_metadata": metadata.raw_model,
        "Processing type": metadata.processing_type,
        "Ontology ID": metadata.ontology_id,
        "Batch task ID": metadata.batch_task_id,
    }


def model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def configuration_sort_key(configuration: str) -> int:
    order = {
        "Sem ontologia": 0,
        "Com ontologia": 1,
        "Desconhecida": 2,
    }
    return order.get(configuration, 99)


def matching_sort_key(version: str) -> int:
    order = {
        "Nao classificado": 0,
        "Matching original": 1,
        "Matching adaptado": 2,
    }
    return order.get(version, 99)


def write_table(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Modelo",
        "Configuração",
        "Versão do Matching",
        "Pasta de resultados",
        "Arquivo CSV",
        "Perguntas",
        "Nodes totais",
        "Edges totais",
        "Matches válidos totais",
        "Nodes médios por pergunta",
        "Edges médios por pergunta",
        "Matches válidos médios por pergunta",
        "Taxa de Matching (%)",
        "Modelo batch_metadata",
        "Processing type",
        "Ontology ID",
        "Batch task ID",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_number(value) for key, value in row.items()})


def main() -> int:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Pasta de resultados nao encontrada: {args.results_dir}")

    csv_paths = sorted(args.results_dir.rglob("*-completo.csv"))
    if not csv_paths:
        raise FileNotFoundError("Nenhum arquivo *-completo.csv foi encontrado em resultados/.")

    rows = [build_audit_row(path) for path in csv_paths]
    rows.sort(
        key=lambda row: (
            model_sort_key(str(row["Modelo"])),
            configuration_sort_key(str(row["Configuração"])),
            matching_sort_key(str(row["Versão do Matching"])),
            str(row["Pasta de resultados"]),
        )
    )

    write_table(rows, args.output)
    print(f"Tabela de auditoria gerada: {args.output}")
    print(f"Linhas geradas: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
