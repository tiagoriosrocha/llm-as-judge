from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SUBSET_ROOT = PROJECT_ROOT.parent
DEFAULT_ANALISES_DIR = SUBSET_ROOT / "resultados_analises"
REQUIRED_COLUMNS = (
    "total_graph_data_nodes",
    "total_edges",
    "total_ontology_valid_true",
)


@dataclass
class MatchingStats:
    version: str
    total_nodes: int = 0
    total_edges: int = 0
    total_matches: int = 0
    total_rows: int = 0
    source_files: int = 0

    @property
    def matching_rate(self) -> float:
        if self.total_nodes == 0:
            return math.nan
        return self.total_matches / self.total_nodes * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera o CSV da Tabela 9 a partir dos CSVs de resultados com "
            "dados de matching, nodes e edges."
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
        default=DEFAULT_ANALISES_DIR / "tabelas" / "tabela_9.csv",
        help="Caminho do CSV final da Tabela 9.",
    )
    return parser.parse_args()


def classify_matching_version(path: Path) -> str | None:
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

    adapted_markers = (
        "adaptatado",
        "adaptado",
        "adaptacao",
        "adaptation",
        "adapt",
    )
    if any(marker in lowered for marker in adapted_markers):
        return "Matching adaptado"

    return None


def iter_matching_csvs(results_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for path in sorted(results_dir.rglob("*-completo.csv")):
        version = classify_matching_version(path)
        if version is None:
            continue
        files.append((version, path))
    return files


def require_columns(fieldnames: list[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV vazio ou sem cabecalho: {path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            f"{path} nao possui as colunas obrigatorias: {', '.join(missing)}"
        )


def aggregate_file(version: str, path: Path, stats_by_version: dict[str, MatchingStats]) -> None:
    file_nodes = 0
    file_edges = 0
    file_matches = 0
    file_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, path)

        stats = stats_by_version.setdefault(version, MatchingStats(version=version))
        for row in reader:
            nodes = int(float(row["total_graph_data_nodes"]))
            edges = int(float(row["total_edges"]))
            matches = int(float(row["total_ontology_valid_true"]))

            stats.total_nodes += nodes
            stats.total_edges += edges
            stats.total_matches += matches
            stats.total_rows += 1
            file_nodes += nodes
            file_edges += edges
            file_matches += matches
            file_rows += 1

        stats.source_files += 1

    print(
        f"[{version}] {path} | linhas={file_rows} nodes={file_nodes} "
        f"edges={file_edges} matches={file_matches}"
    )


def safe_delta_percent(compared: float, base: float) -> float:
    if math.isnan(compared) or math.isnan(base) or base == 0:
        return math.nan
    return (compared - base) / base * 100


def build_table_rows(stats_by_version: dict[str, MatchingStats]) -> list[dict[str, object]]:
    original = stats_by_version.get("Matching original", MatchingStats("Matching original"))
    adapted = stats_by_version.get("Matching adaptado", MatchingStats("Matching adaptado"))

    return [
        {
            "Versão do Matching": original.version,
            "Matches válidos": original.total_matches if original.total_rows else math.nan,
            "Taxa de Matching": original.matching_rate,
        },
        {
            "Versão do Matching": adapted.version,
            "Matches válidos": adapted.total_matches if adapted.total_rows else math.nan,
            "Taxa de Matching": adapted.matching_rate,
        },
        {
            "Versão do Matching": "Ganho (%)",
            "Matches válidos": safe_delta_percent(adapted.total_matches, original.total_matches),
            "Taxa de Matching": safe_delta_percent(
                adapted.matching_rate,
                original.matching_rate,
            ),
        },
    ]


def format_number(value: object) -> object:
    if not isinstance(value, float):
        return value
    if math.isnan(value):
        return ""
    rounded = round(value, 4)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def write_table(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Versão do Matching", "Matches válidos", "Taxa de Matching"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_number(value) for key, value in row.items()})


def main() -> int:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Pasta de resultados nao encontrada: {args.results_dir}")

    files = iter_matching_csvs(args.results_dir)
    if not files:
        raise FileNotFoundError(
            "Nenhum CSV de matching original/adaptado foi encontrado em resultados/."
        )

    stats_by_version: dict[str, MatchingStats] = {}
    for version, path in files:
        aggregate_file(version, path, stats_by_version)

    rows = build_table_rows(stats_by_version)
    write_table(rows, args.output)
    for version in ("Matching original", "Matching adaptado"):
        stats = stats_by_version.get(version)
        if stats is None:
            continue
        # A taxa usa os nodes como denominador porque total_ontology_valid_true
        # nunca excede total_graph_data_nodes nos resultados disponiveis.
        print(
            f"{version}: nodes={stats.total_nodes} edges={stats.total_edges} "
            f"matches={stats.total_matches} taxa={stats.matching_rate:.4f}%"
        )
    print(f"Tabela 9 gerada: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
