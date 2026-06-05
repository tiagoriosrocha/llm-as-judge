from __future__ import annotations

from pathlib import Path

import pandas as pd

from analise_utils import (
    DEFAULT_OUTPUT_DIR_NAME,
    detect_analysis_group,
    detect_configuration,
    detect_model,
    run_table,
)


def load_available_data(avaliacao_dir: Path) -> pd.DataFrame:
    """
    Carrega preferencialmente o arquivo detalhado, pois ele permite contar
    perguntas distintas por question_id.

    Se o arquivo detalhado não existir, usa o resumo por execução.
    """
    detailed_path = avaliacao_dir / "todas_avaliadas.csv"
    summary_path = avaliacao_dir / "resumo_avaliadas.csv"

    if detailed_path.exists():
        df = pd.read_csv(detailed_path, encoding="utf-8-sig")
        source_file = detailed_path.name
    elif summary_path.exists():
        df = pd.read_csv(summary_path, encoding="utf-8-sig")
        source_file = summary_path.name
    else:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {avaliacao_dir}. "
            "Esperado: todas_avaliadas.csv ou resumo_avaliadas.csv"
        )

    required_columns = ["arquivo_fonte", "tipo_resposta"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"O arquivo {source_file} não possui as colunas obrigatórias: {missing}"
        )

    df["arquivo_fonte"] = df["arquivo_fonte"].fillna("").astype(str)
    df["tipo_resposta"] = df["tipo_resposta"].fillna("").astype(str).str.lower()

    df["modelo"] = df["arquivo_fonte"].apply(detect_model)
    df["configuracao"] = df.apply(
        lambda row: detect_configuration(row["arquivo_fonte"], row["tipo_resposta"]),
        axis=1,
    )
    df["grupo_analise"] = df.apply(
        lambda row: detect_analysis_group(row["arquivo_fonte"], row["tipo_resposta"]),
        axis=1,
    )

    df["arquivo_origem_contagem"] = source_file

    return df


def build_table(avaliacao_dir: Path) -> pd.DataFrame:
    df = load_available_data(avaliacao_dir)

    group_cols = [
        "modelo",
        "grupo_analise",
        "configuracao",
        "tipo_resposta",
        "arquivo_fonte",
    ]

    aggregation = {
        "arquivo_origem_contagem": "first",
    }

    if "question_id" in df.columns:
        result = (
            df.groupby(group_cols, dropna=False)
            .agg(
                num_linhas=("question_id", "size"),
                num_perguntas_distintas=("question_id", "nunique"),
                arquivo_origem_contagem=("arquivo_origem_contagem", "first"),
            )
            .reset_index()
        )
    elif "num_linhas" in df.columns:
        result = (
            df.groupby(group_cols, dropna=False)
            .agg(
                num_linhas=("num_linhas", "sum"),
                num_perguntas_distintas=("num_linhas", "sum"),
                arquivo_origem_contagem=("arquivo_origem_contagem", "first"),
            )
            .reset_index()
        )
    else:
        result = (
            df.groupby(group_cols, dropna=False)
            .agg(
                num_linhas=("arquivo_fonte", "size"),
                arquivo_origem_contagem=("arquivo_origem_contagem", "first"),
            )
            .reset_index()
        )
        result["num_perguntas_distintas"] = result["num_linhas"]

    result = result[
        [
            "modelo",
            "grupo_analise",
            "configuracao",
            "tipo_resposta",
            "num_linhas",
            "num_perguntas_distintas",
            "arquivo_fonte",
            "arquivo_origem_contagem",
        ]
    ]

    model_order = {
        "GPT-4o": 0,
        "GPT-4.1": 1,
        "GPT-5.2": 2,
        "GPT-5.4": 3,
        "GPT-o3": 4,
        "Desconhecido": 99,
    }

    group_order = {
        "Answer": 0,
        "RAG": 1,
        "GraphRAG": 2,
        "GraphRAG + Ontologia": 3,
        "Desconhecido": 99,
    }

    result["_model_order"] = result["modelo"].map(model_order).fillna(50)
    result["_group_order"] = result["grupo_analise"].map(group_order).fillna(50)

    result = (
        result.sort_values(
            ["_model_order", "_group_order", "tipo_resposta", "arquivo_fonte"]
        )
        .drop(columns=["_model_order", "_group_order"])
        .reset_index(drop=True)
    )

    save_coverage_table(result, avaliacao_dir)

    return result


def save_coverage_table(result: pd.DataFrame, avaliacao_dir: Path) -> None:
    """
    Gera uma tabela extra de cobertura por modelo para verificar rapidamente
    se todos os modelos possuem Answer, RAG, GraphRAG e GraphRAG + Ontologia.
    """
    output_dir = avaliacao_dir / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = (
        result.pivot_table(
            index="modelo",
            columns="grupo_analise",
            values="num_perguntas_distintas",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    expected_columns = [
        "modelo",
        "Answer",
        "RAG",
        "GraphRAG",
        "GraphRAG + Ontologia",
    ]

    for col in expected_columns:
        if col not in coverage.columns:
            coverage[col] = 0

    coverage = coverage[expected_columns]

    coverage["tem_rag"] = coverage["RAG"] > 0
    coverage["tem_graphrag"] = coverage["GraphRAG"] > 0
    coverage["tem_graphrag_ontologia"] = coverage["GraphRAG + Ontologia"] > 0
    coverage["comparacao_h1_ok"] = coverage["tem_rag"] & coverage["tem_graphrag"]
    coverage["comparacao_ontologia_ok"] = (
        coverage["tem_graphrag"] & coverage["tem_graphrag_ontologia"]
    )

    coverage_path = output_dir / "tabela_0_cobertura_por_modelo.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    print(f"Tabela de cobertura gerada: {coverage_path}")


if __name__ == "__main__":
    run_table(0, build_table)