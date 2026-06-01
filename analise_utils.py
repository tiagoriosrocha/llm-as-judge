from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AVALIACAO_DIR = PROJECT_ROOT / "avaliacao"
DEFAULT_OUTPUT_DIR_NAME = "tabelas"
MODEL_ORDER = ["GPT-4.1", "GPT-5.2", "GPT-5.4"]

METRIC_COLUMNS = {
    "Correctness ↑": "correctness_score_mean",
    "Faithfulness ↑": "faithfulness_score_mean",
    "Completeness ↑": "completeness_score_mean",
    "Geological Adequacy ↑": "geological_adequacy_score_mean",
    "Ontology Alignment ↑": "ontology_alignment_score_mean",
    "Contextual Recall ↑": "contextual_recall_score_mean",
    "Contextual Relevancy ↑": "contextual_relevancy_score_mean",
    "Answer Relevancy ↑": "answer_relevancy_score_mean",
}

MATCHING_COLUMNS = {
    "total_ontology_valid_true",
    "ontology_valid_true",
    "total_matches",
    "match_rate",
    "taxa_matching",
    "ontology_match",
    "valid_match",
}


def parse_avaliacao_dir() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avaliacao-dir",
        type=Path,
        default=DEFAULT_AVALIACAO_DIR,
        help="Pasta com os CSVs de avaliacao. Padrao: avaliacao/.",
    )
    args = parser.parse_args()
    return args.avaliacao_dir.resolve()


def load_summary(avaliacao_dir: Path = DEFAULT_AVALIACAO_DIR) -> pd.DataFrame:
    path = avaliacao_dir / "resumo_metricas_por_execucao.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV de resumo nao encontrado: {path}")
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(dataframe, ["arquivo_fonte", "tipo_resposta"], path.name)
    dataframe["arquivo_fonte"] = dataframe["arquivo_fonte"].fillna("").astype(str)
    dataframe["tipo_resposta"] = dataframe["tipo_resposta"].fillna("").astype(str)
    dataframe["modelo"] = dataframe["arquivo_fonte"].map(detect_model)
    dataframe["configuracao"] = dataframe.apply(
        lambda row: detect_configuration(row["arquivo_fonte"], row["tipo_resposta"]),
        axis=1,
    )
    return dataframe


def load_details(avaliacao_dir: Path = DEFAULT_AVALIACAO_DIR) -> pd.DataFrame:
    path = avaliacao_dir / "todas_execucoes_deepeval_avaliadas.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV detalhado nao encontrado: {path}")
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(dataframe, ["arquivo_fonte", "tipo_resposta"], path.name)
    dataframe["arquivo_fonte"] = dataframe["arquivo_fonte"].fillna("").astype(str)
    dataframe["tipo_resposta"] = dataframe["tipo_resposta"].fillna("").astype(str)
    return dataframe


def detect_model(arquivo_fonte: str) -> str:
    lowered = str(arquivo_fonte).lower()
    if "4-1" in lowered:
        return "GPT-4.1"
    if "5-2" in lowered:
        return "GPT-5.2"
    if "5-4" in lowered:
        return "GPT-5.4"
    return "Desconhecido"


def detect_configuration(arquivo_fonte: str, tipo_resposta: str = "") -> str:
    lowered = str(arquivo_fonte).lower()
    response_type = str(tipo_resposta).lower()
    if response_type == "answer" or "sem-contexto-sem-ontologia" in lowered:
        return "Answer"
    if "com-ontologia" in lowered:
        return "GraphRAG + Ontologia"
    if "sem-ontologia" in lowered:
        return "GraphRAG"
    if response_type == "rag":
        return "RAG"
    if response_type == "graphrag":
        return "GraphRAG"
    return "Desconhecido"


def require_columns(dataframe: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{source_name} nao possui as colunas obrigatorias: {', '.join(missing)}"
        )


def require_metric_columns(dataframe: pd.DataFrame, display_columns: list[str]) -> None:
    require_columns(
        dataframe,
        [METRIC_COLUMNS[column] for column in display_columns],
        "resumo_metricas_por_execucao.csv",
    )


def safe_delta_percent(valor_comparado: float, valor_base: float) -> float:
    if pd.isna(valor_base) or pd.isna(valor_comparado) or valor_base == 0:
        return math.nan
    return (valor_comparado - valor_base) / valor_base * 100


def save_table(dataframe: pd.DataFrame, table_number: int, avaliacao_dir: Path) -> Path:
    output_dir = avaliacao_dir / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tabela_{table_number}.csv"
    rounded = dataframe.copy()
    numeric_columns = rounded.select_dtypes(include="number").columns
    rounded[numeric_columns] = rounded[numeric_columns].round(4)
    rounded.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Tabela {table_number} gerada: {output_path}")
    return output_path


def run_table(table_number: int, builder: Callable[[Path], pd.DataFrame]) -> None:
    avaliacao_dir = parse_avaliacao_dir()
    dataframe = builder(avaliacao_dir)
    save_table(dataframe, table_number, avaliacao_dir)


def filter_rag(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe[dataframe["tipo_resposta"].str.lower() == "rag"].copy()


def filter_graphrag_without_ontology(dataframe: pd.DataFrame) -> pd.DataFrame:
    source = dataframe["arquivo_fonte"].str.lower()
    response_type = dataframe["tipo_resposta"].str.lower()
    return dataframe[
        (response_type == "graphrag")
        & source.str.contains("sem-ontologia", na=False)
        & ~source.str.contains("sem-contexto", na=False)
    ].copy()


def filter_graphrag_with_ontology(dataframe: pd.DataFrame) -> pd.DataFrame:
    source = dataframe["arquivo_fonte"].str.lower()
    response_type = dataframe["tipo_resposta"].str.lower()
    return dataframe[
        (response_type == "graphrag")
        & source.str.contains("com-ontologia", na=False)
    ].copy()


def mean_metrics(dataframe: pd.DataFrame, display_columns: list[str]) -> dict[str, float]:
    require_metric_columns(dataframe, display_columns)
    return {
        display_column: pd.to_numeric(
            dataframe[METRIC_COLUMNS[display_column]],
            errors="coerce",
        ).mean()
        for display_column in display_columns
    }


def row_from_metrics(label_column: str, label: str, metrics: dict[str, float]) -> dict[str, object]:
    return {label_column: label, **metrics}


def delta_row(
    label_column: str,
    base: dict[str, float],
    compared: dict[str, float],
    display_columns: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {label_column: "Δ (%)"}
    for column in display_columns:
        row[column] = safe_delta_percent(compared.get(column), base.get(column))
    return row


def build_method_comparison(
    avaliacao_dir: Path,
    display_columns: list[str],
    include_delta: bool,
) -> pd.DataFrame:
    summary = load_summary(avaliacao_dir)
    rag_metrics = mean_metrics(filter_rag(summary), display_columns)
    graphrag_metrics = mean_metrics(filter_graphrag_without_ontology(summary), display_columns)
    rows = [
        row_from_metrics("Método", "RAG", rag_metrics),
        row_from_metrics("Método", "GraphRAG", graphrag_metrics),
    ]
    if include_delta:
        rows.append(delta_row("Método", rag_metrics, graphrag_metrics, display_columns))
    return pd.DataFrame(rows, columns=["Método", *display_columns])


def build_ontology_comparison(
    avaliacao_dir: Path,
    display_columns: list[str],
    include_delta: bool,
) -> pd.DataFrame:
    summary = load_summary(avaliacao_dir)
    graphrag_metrics = mean_metrics(filter_graphrag_without_ontology(summary), display_columns)
    ontology_metrics = mean_metrics(filter_graphrag_with_ontology(summary), display_columns)
    rows = [
        row_from_metrics("Configuração", "GraphRAG", graphrag_metrics),
        row_from_metrics("Configuração", "GraphRAG + Ontologia", ontology_metrics),
    ]
    if include_delta:
        rows.append(delta_row("Configuração", graphrag_metrics, ontology_metrics, display_columns))
    return pd.DataFrame(rows, columns=["Configuração", *display_columns])


def build_ontology_gain_rows(
    avaliacao_dir: Path,
    display_columns: list[str],
) -> pd.DataFrame:
    summary = load_summary(avaliacao_dir)
    graphrag_metrics = mean_metrics(filter_graphrag_without_ontology(summary), display_columns)
    ontology_metrics = mean_metrics(filter_graphrag_with_ontology(summary), display_columns)
    rows = []
    for display_column in display_columns:
        metric_name = display_column.replace(" ↑", "")
        rows.append(
            {
                "Métrica": metric_name,
                "GraphRAG": graphrag_metrics[display_column],
                "GraphRAG + Ontologia": ontology_metrics[display_column],
                "Δ (%)": safe_delta_percent(
                    ontology_metrics[display_column],
                    graphrag_metrics[display_column],
                ),
            }
        )
    return pd.DataFrame(rows, columns=["Métrica", "GraphRAG", "GraphRAG + Ontologia", "Δ (%)"])


def build_rag_vs_graphrag_by_model(avaliacao_dir: Path) -> pd.DataFrame:
    display_columns = ["Correctness ↑", "Faithfulness ↑", "Completeness ↑"]
    summary = load_summary(avaliacao_dir)
    rows = []
    for model in MODEL_ORDER:
        model_df = summary[summary["modelo"] == model]
        for method, subset in [
            ("RAG", filter_rag(model_df)),
            ("GraphRAG", filter_graphrag_without_ontology(model_df)),
        ]:
            rows.append(
                {
                    "Modelo": model,
                    "Método": method,
                    **mean_metrics(subset, display_columns),
                }
            )
    return pd.DataFrame(rows, columns=["Modelo", "Método", *display_columns])


def build_ontology_delta_by_model(
    avaliacao_dir: Path,
    display_columns: list[str],
    include_correctness_values: bool,
    only_alignment: bool = False,
) -> pd.DataFrame:
    summary = load_summary(avaliacao_dir)
    rows = []
    for model in MODEL_ORDER:
        model_df = summary[summary["modelo"] == model]
        base = mean_metrics(filter_graphrag_without_ontology(model_df), display_columns)
        compared = mean_metrics(filter_graphrag_with_ontology(model_df), display_columns)
        row: dict[str, object] = {"Modelo": model}
        if include_correctness_values:
            row["GraphRAG"] = base["Correctness ↑"]
            row["GraphRAG + Ontologia"] = compared["Correctness ↑"]
        for column in display_columns:
            delta_name = f"Δ {column}"
            if only_alignment and column == "Ontology Alignment ↑":
                delta_name = "Δ Ontology Alignment"
            row[delta_name] = safe_delta_percent(compared[column], base[column])
        rows.append(row)
    if only_alignment:
        return pd.DataFrame(rows, columns=["Modelo", "Δ Ontology Alignment"])
    columns = ["Modelo"]
    if include_correctness_values:
        columns.extend(["GraphRAG", "GraphRAG + Ontologia"])
    columns.extend([f"Δ {column}" for column in display_columns])
    return pd.DataFrame(rows, columns=columns)


def booleanish_sum(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return math.nan
    lowered = cleaned.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "sim", "s", "yes", "y", "match", "matched"}
    falsy = {"0", "false", "f", "nao", "não", "n", "no", "none", ""}
    if lowered.isin(truthy | falsy).all():
        return float(lowered.isin(truthy).sum())
    return pd.to_numeric(cleaned, errors="coerce").sum()


def normalize_rate(value: float) -> float:
    if pd.isna(value):
        return math.nan
    return value * 100 if 0 <= value <= 1 else value


def matching_version_from_path(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "adapt" in lowered:
        return "Matching adaptado"
    if "original" in lowered or "antes" in lowered or "before" in lowered:
        return "Matching original"
    return "Desconhecido"


def extract_matching_stats(path: Path) -> dict[str, object] | None:
    try:
        dataframe = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    matching_columns = MATCHING_COLUMNS.intersection(dataframe.columns)
    if not matching_columns:
        return None

    valid_count = math.nan
    for column in [
        "total_ontology_valid_true",
        "ontology_valid_true",
        "total_matches",
        "valid_match",
        "ontology_match",
    ]:
        if column not in dataframe.columns:
            continue
        if column.startswith("total_"):
            valid_count = pd.to_numeric(dataframe[column], errors="coerce").max()
        else:
            valid_count = booleanish_sum(dataframe[column])
        break

    rate = math.nan
    for column in ["match_rate", "taxa_matching"]:
        if column in dataframe.columns:
            rate = normalize_rate(pd.to_numeric(dataframe[column], errors="coerce").mean())
            break
    if pd.isna(rate) and not pd.isna(valid_count) and len(dataframe) > 0:
        rate = valid_count / len(dataframe) * 100

    return {
        "Versão do Matching": matching_version_from_path(path),
        "Matches válidos": valid_count,
        "Taxa de Matching": rate,
    }


def build_matching_table(avaliacao_dir: Path) -> pd.DataFrame:
    stats = []
    for path in avaliacao_dir.rglob("*.csv"):
        if "tabelas" in path.parts:
            continue
        extracted = extract_matching_stats(path)
        if extracted is not None:
            stats.append(extracted)

    columns = ["Versão do Matching", "Matches válidos", "Taxa de Matching"]
    if not stats:
        print(
            f"Dados de matching ontologico nao encontrados em {avaliacao_dir}. "
            "Preencha manualmente tabela_9.csv quando esses dados estiverem disponiveis."
        )
        # Preencher manualmente Matches validos e Taxa de Matching quando houver os dados.
        return pd.DataFrame(
            [
                {"Versão do Matching": "Matching original", "Matches válidos": math.nan, "Taxa de Matching": math.nan},
                {"Versão do Matching": "Matching adaptado", "Matches válidos": math.nan, "Taxa de Matching": math.nan},
                {"Versão do Matching": "Ganho (%)", "Matches válidos": math.nan, "Taxa de Matching": math.nan},
            ],
            columns=columns,
        )

    raw = pd.DataFrame(stats)
    grouped = raw.groupby("Versão do Matching", as_index=False).agg(
        {"Matches válidos": "sum", "Taxa de Matching": "mean"}
    )
    rows = []
    for version in ["Matching original", "Matching adaptado"]:
        match = grouped[grouped["Versão do Matching"] == version]
        if match.empty:
            rows.append({"Versão do Matching": version, "Matches válidos": math.nan, "Taxa de Matching": math.nan})
        else:
            rows.append(match.iloc[0].to_dict())
    original, adapted = rows
    rows.append(
        {
            "Versão do Matching": "Ganho (%)",
            "Matches válidos": safe_delta_percent(
                adapted["Matches válidos"],
                original["Matches válidos"],
            ),
            "Taxa de Matching": safe_delta_percent(
                adapted["Taxa de Matching"],
                original["Taxa de Matching"],
            ),
        }
    )
    return pd.DataFrame(rows, columns=columns)
