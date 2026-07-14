from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SUBSET_ROOT = SCRIPT_DIR.parent
ANALISES_DIR = SUBSET_ROOT / "resultados_analises"
RESULTADOS_DIR = ANALISES_DIR / "resultados_filtrados"

DEFAULT_ANTES_DIR = RESULTADOS_DIR / "petrobras-5-2-com-ontologia-antes-adaptacao"
DEFAULT_ADAPTADO_DIR = RESULTADOS_DIR / "petrobras-5-2-com-ontologia"
DEFAULT_OUTPUT = (
    ANALISES_DIR
    / "tabelas"
    / "comparacao_5_2_ontologia_antes_vs_adaptado.csv"
)

SCORE_METRICS = [
    (
        "RAG - Answer Relevancy",
        "evaluation.rag.metrics.answer_relevancy.score",
    ),
    (
        "RAG - Faithfulness",
        "evaluation.rag.metrics.faithfulness.score",
    ),
    (
        "GraphRAG - Answer Relevancy",
        "evaluation.graph.answer_relevancy.score",
    ),
    (
        "GraphRAG - Faithfulness",
        "evaluation.graph.faithfulness.score",
    ),
]

STRUCTURAL_METRICS = [
    ("Graph data nodes", "total_graph_data_nodes"),
    ("Edges", "total_edges"),
    ("Matches validos", "total_ontology_valid_true"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara os resultados do GPT-5.2 com ontologia antes e depois "
            "da adaptacao do matching."
        )
    )
    parser.add_argument(
        "--antes-dir",
        type=Path,
        default=DEFAULT_ANTES_DIR,
        help="Pasta do conjunto anterior a adaptacao.",
    )
    parser.add_argument(
        "--adaptado-dir",
        type=Path,
        default=DEFAULT_ADAPTADO_DIR,
        help="Pasta do conjunto com matching adaptado.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV comparativo de saida.",
    )
    return parser.parse_args()


def find_result_csv(directory: Path) -> Path:
    paths = sorted(directory.glob("*-completo.csv"))
    if not paths:
        raise FileNotFoundError(f"Nenhum arquivo *-completo.csv encontrado em {directory}.")
    if len(paths) > 1:
        raise ValueError(f"Mais de um arquivo *-completo.csv encontrado em {directory}.")
    return paths[0]


def normalize_question_ids(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_results(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required_columns = {
        "question_id",
        *[column for _, column in SCORE_METRICS],
        *[column for _, column in STRUCTURAL_METRICS],
    }
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"{path} nao possui as colunas: {', '.join(missing)}")

    dataframe = dataframe.copy()
    dataframe["question_id"] = normalize_question_ids(dataframe["question_id"])
    if dataframe["question_id"].duplicated().any():
        duplicated = sorted(
            dataframe.loc[dataframe["question_id"].duplicated(), "question_id"].unique()
        )
        raise ValueError(f"{path} possui question_id duplicado: {', '.join(duplicated)}")

    for _, column in [*SCORE_METRICS, *STRUCTURAL_METRICS]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def safe_delta_percent(adaptado: float, antes: float) -> float:
    if pd.isna(adaptado) or pd.isna(antes) or antes == 0:
        return math.nan
    return (adaptado - antes) / antes * 100


def comparison_row(
    metric: str,
    aggregation: str,
    before_value: float,
    adapted_value: float,
    question_count: int,
) -> dict[str, object]:
    return {
        "metrica": metric,
        "agregacao": aggregation,
        "antes_adaptacao": before_value,
        "adaptado": adapted_value,
        "diferenca_adaptado_menos_antes": adapted_value - before_value,
        "variacao_percentual": safe_delta_percent(adapted_value, before_value),
        "questoes_comparadas": question_count,
    }


def build_comparison(
    before: pd.DataFrame,
    adapted: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, int]:
    before_ids = set(before["question_id"])
    adapted_ids = set(adapted["question_id"])
    common_ids = before_ids & adapted_ids
    if not common_ids:
        raise ValueError("Os conjuntos nao possuem question_id em comum.")

    before_common = before[before["question_id"].isin(common_ids)].copy()
    adapted_common = adapted[adapted["question_id"].isin(common_ids)].copy()
    question_count = len(common_ids)

    rows: list[dict[str, object]] = []
    for label, column in SCORE_METRICS:
        rows.append(
            comparison_row(
                metric=label,
                aggregation="media",
                before_value=float(before_common[column].mean()),
                adapted_value=float(adapted_common[column].mean()),
                question_count=question_count,
            )
        )

    for label, column in STRUCTURAL_METRICS:
        rows.append(
            comparison_row(
                metric=label,
                aggregation="total",
                before_value=float(before_common[column].sum()),
                adapted_value=float(adapted_common[column].sum()),
                question_count=question_count,
            )
        )
        rows.append(
            comparison_row(
                metric=label,
                aggregation="media_por_questao",
                before_value=float(before_common[column].mean()),
                adapted_value=float(adapted_common[column].mean()),
                question_count=question_count,
            )
        )

    before_matching_rate = (
        before_common["total_ontology_valid_true"].sum()
        / before_common["total_graph_data_nodes"].sum()
        * 100
    )
    adapted_matching_rate = (
        adapted_common["total_ontology_valid_true"].sum()
        / adapted_common["total_graph_data_nodes"].sum()
        * 100
    )
    rows.append(
        comparison_row(
            metric="Taxa de matching",
            aggregation="percentual",
            before_value=float(before_matching_rate),
            adapted_value=float(adapted_matching_rate),
            question_count=question_count,
        )
    )

    comparison = pd.DataFrame(rows)
    return comparison, len(before_ids), len(adapted_ids), question_count


def main() -> int:
    args = parse_args()
    before_path = find_result_csv(args.antes_dir)
    adapted_path = find_result_csv(args.adaptado_dir)

    before = load_results(before_path)
    adapted = load_results(adapted_path)
    comparison, before_count, adapted_count, common_count = build_comparison(
        before,
        adapted,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Tabela comparativa gerada: {args.output}")
    print(
        f"Questoes: antes={before_count}, adaptado={adapted_count}, "
        f"comparadas={common_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
