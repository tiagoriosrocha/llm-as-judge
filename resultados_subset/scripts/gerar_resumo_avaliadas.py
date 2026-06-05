from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SUBSET_ROOT = PROJECT_ROOT.parent
DEFAULT_AVALIACAO_DIR = SUBSET_ROOT / "resultados_analises"

METRIC_COLUMNS = [
    "answer_relevancy_score",
    "faithfulness_score",
    "contextual_relevancy_score",
    "contextual_recall_score",
    "correctness_score",
    "completeness_score",
    "geological_adequacy_score",
    "ontology_alignment_score",
    "rouge1_precision",
    "rouge1_recall",
    "rouge1_fmeasure",
    "rouge2_precision",
    "rouge2_recall",
    "rouge2_fmeasure",
    "rougeL_precision",
    "rougeL_recall",
    "rougeL_fmeasure",
    "exact_match",
    "token_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalcula avaliacao/resumo_avaliadas.csv a partir de "
            "avaliacao/todas_avaliadas.csv."
        )
    )
    parser.add_argument(
        "--avaliacao-dir",
        type=Path,
        default=DEFAULT_AVALIACAO_DIR,
        help="Pasta com os CSVs de avaliacao. Padrao: avaliacao/.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV detalhado de entrada. Padrao: <avaliacao-dir>/todas_avaliadas.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de resumo a atualizar. Padrao: <avaliacao-dir>/resumo_avaliadas.csv.",
    )
    return parser.parse_args()


def require_columns(dataframe: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{source_name} nao possui as colunas obrigatorias: {', '.join(missing)}"
        )


def drop_repeated_header_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    mask = (
        dataframe["arquivo_fonte"].astype(str).str.strip().eq("arquivo_fonte")
        & dataframe["tipo_resposta"].astype(str).str.strip().eq("tipo_resposta")
    )
    return dataframe.loc[~mask].copy() if mask.any() else dataframe


def load_details(input_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    require_columns(dataframe, ["arquivo_fonte", "tipo_resposta"], input_path.name)

    dataframe = drop_repeated_header_rows(dataframe)
    dataframe["arquivo_fonte"] = dataframe["arquivo_fonte"].fillna("").astype(str)
    dataframe["tipo_resposta"] = dataframe["tipo_resposta"].fillna("").astype(str)
    return dataframe


def build_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [column for column in METRIC_COLUMNS if column in dataframe.columns]
    if not available_metrics:
        raise ValueError(
            "Nenhuma coluna numerica de metrica foi encontrada em todas_avaliadas.csv."
        )

    working = dataframe.copy()
    for column in available_metrics:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    grouped = working.groupby(["arquivo_fonte", "tipo_resposta"], sort=False)
    counts = grouped.size().rename("num_linhas").to_frame()

    metrics_summary = grouped[available_metrics].agg(["mean", "median"])
    metrics_std = grouped[available_metrics].std(ddof=0)
    metrics_std.columns = pd.MultiIndex.from_product([metrics_std.columns, ["std"]])
    metrics_summary = metrics_summary.join(metrics_std)
    metrics_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in metrics_summary.columns.to_flat_index()
    ]

    return counts.join(metrics_summary).reset_index()


def write_summary(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    avaliacao_dir = args.avaliacao_dir.resolve()
    input_path = (
        args.input.resolve()
        if args.input is not None
        else avaliacao_dir / "todas_avaliadas.csv"
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else avaliacao_dir / "resumo_avaliadas.csv"
    )

    details = load_details(input_path)
    summary = build_summary(details)
    write_summary(summary, output_path)

    print(
        f"Resumo atualizado: {output_path} "
        f"({len(summary)} grupos, {len(details)} linhas avaliadas)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
