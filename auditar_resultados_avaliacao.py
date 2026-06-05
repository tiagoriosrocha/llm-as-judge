from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_MODELS = ["GPT-4o", "GPT-4.1", "GPT-5-mini", "GPT-5.2", "GPT-5.4", "GPT-o3"]
EXPECTED_GROUPS = ["Answer", "RAG", "GraphRAG", "GraphRAG + Ontologia"]
EXPECTED_QUESTIONS_PER_GROUP = 312
TOLERANCE = 0.0001

# Fontes consolidadas: juntam todos os resultados parciais disponiveis.
DETAILS_FILE = "todas_avaliadas.csv"
SUMMARY_FILE = "resumo_avaliadas.csv"

DEEPEVAL_METRIC_COLUMNS = [
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

REQUIRED_DETAIL_COLUMNS = [*DEEPEVAL_METRIC_COLUMNS, "eval_error"]

SUMMARY_COMPARISON_METRICS = [
    "answer_relevancy_score",
    "faithfulness_score",
    "contextual_relevancy_score",
    "contextual_recall_score",
    "correctness_score",
    "completeness_score",
    "geological_adequacy_score",
    "ontology_alignment_score",
    "token_f1",
]

STRUCTURAL_COLUMNS = [
    "total_graph_data_nodes",
    "total_edges",
    "total_ontology_valid_true",
]

RESULT_FILE_COLUMNS = [
    "modelo",
    "grupo_detectado",
    "pasta",
    "caminho_csv",
    "nome_arquivo",
    "tamanho_bytes",
    "hash_md5",
    "batch_id",
    "quantidade_linhas",
    "quantidade_question_id_distintos",
    "possui_colunas_de_metricas_deepeval",
    "possui_colunas_estruturais_matching",
    "total_graph_data_nodes",
    "total_edges",
    "total_ontology_valid_true",
    "taxa_matching",
]


def detect_model(path_or_text: str | Path) -> str:
    text = str(path_or_text).lower().replace("\\", "/")
    patterns = [
        ("petrobras-4-0", "GPT-4o"),
        ("petrobras-4-1", "GPT-4.1"),
        ("petrobras-5-mini", "GPT-5-mini"),
        ("petrobras-5-2", "GPT-5.2"),
        ("petrobras-5-4", "GPT-5.4"),
        ("petrobras-0-3", "GPT-o3"),
    ]
    for pattern, model in patterns:
        if pattern in text:
            return model
    return "Desconhecido"


def detect_analysis_group(path_or_text: str | Path, tipo_resposta: str | None = None) -> str:
    text = str(path_or_text).lower().replace("\\", "/")
    response_type = str(tipo_resposta or "").strip().lower()

    if response_type == "answer" or "sem-contexto-sem-ontologia" in text:
        return "Answer"
    if response_type == "rag" and "com-ontologia" in text:
        return "RAG dentro de processamento com ontologia"
    if response_type == "rag" and "sem-ontologia" in text and "sem-contexto" not in text:
        return "RAG"
    if response_type == "graphrag" and "com-ontologia" in text:
        return "GraphRAG + Ontologia"
    if response_type == "graphrag" and "sem-ontologia" in text and "sem-contexto" not in text:
        return "GraphRAG"

    # Fallback used for file-level audits, where the CSV often contains both RAG
    # and GraphRAG columns but no tipo_resposta column.
    if not response_type:
        if "com-ontologia" in text:
            return "GraphRAG + Ontologia"
        if "sem-ontologia" in text and "sem-contexto" not in text:
            return "GraphRAG"

    return "Desconhecido"


def extract_batch_id(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    match = re.search(r"batch[_-]([0-9a-fA-F-]{32,36})", name)
    if match:
        return match.group(1)
    match = re.search(r"batch[_-]([^-_.]+)", name)
    return match.group(1) if match else ""


def compute_file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_safely(path: Path, errors: list[str]) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        return drop_repeated_header_rows(dataframe)
    except Exception as exc:
        errors.append(f"Falha ao ler {path}: {exc}")
        return pd.DataFrame()


def drop_repeated_header_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas de cabecalho repetidas apos concatenacao de CSVs parciais."""
    if dataframe.empty:
        return dataframe
    if {"arquivo_fonte", "tipo_resposta"}.issubset(dataframe.columns):
        mask = (
            dataframe["arquivo_fonte"].astype(str).str.strip().eq("arquivo_fonte")
            & dataframe["tipo_resposta"].astype(str).str.strip().eq("tipo_resposta")
        )
        if mask.any():
            return dataframe.loc[~mask].copy()
    return dataframe


def write_csv(dataframe: pd.DataFrame, output_path: Path, columns: list[str] | None = None) -> None:
    if columns is not None:
        dataframe = dataframe.reindex(columns=columns)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def unique_join(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return " | ".join(cleaned)


def numeric_sum(dataframe: pd.DataFrame, column: str) -> float:
    if column not in dataframe.columns:
        return math.nan
    return float(pd.to_numeric(dataframe[column], errors="coerce").sum())


def safe_percentage(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator * 100)


def has_deepeval_metric_columns(dataframe: pd.DataFrame) -> bool:
    direct_columns = set(DEEPEVAL_METRIC_COLUMNS).intersection(dataframe.columns)
    if direct_columns:
        return True
    metric_like = [
        column
        for column in dataframe.columns
        if "evaluation." in column.lower() and ".score" in column.lower()
    ]
    return bool(metric_like)


def is_missing(series: pd.Series) -> pd.Series:
    missing = series.isna()
    as_text = series.astype(str).str.strip()
    return missing | as_text.eq("") | as_text.str.lower().isin({"nan", "none", "null"})


def audit_coverage(avaliacao_dir: Path, output_dir: Path, errors: list[str]) -> pd.DataFrame:
    path = avaliacao_dir / DETAILS_FILE
    dataframe = read_csv_safely(path, errors)
    columns = [
        "modelo",
        "Answer",
        "RAG",
        "GraphRAG",
        "GraphRAG + Ontologia",
        "total_linhas",
        "comparacao_h1_ok",
        "comparacao_ontologia_ok",
        "status",
    ]
    if dataframe.empty:
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_cobertura_por_modelo.csv", columns)
        return result

    required = {"arquivo_fonte", "tipo_resposta", "question_id"}
    missing_required = required - set(dataframe.columns)
    if missing_required:
        errors.append(f"{path} nao possui colunas obrigatorias: {sorted(missing_required)}")
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_cobertura_por_modelo.csv", columns)
        return result

    working = dataframe.copy()
    working["modelo"] = working["arquivo_fonte"].map(detect_model)
    working["grupo"] = working.apply(
        lambda row: detect_analysis_group(row["arquivo_fonte"], row["tipo_resposta"]),
        axis=1,
    )

    models = list(EXPECTED_MODELS)
    unknown_or_extra = sorted(set(working["modelo"]) - set(models))
    models.extend(unknown_or_extra)

    rows: list[dict[str, Any]] = []
    for model in models:
        model_df = working[working["modelo"] == model]
        row: dict[str, Any] = {"modelo": model}
        for group in EXPECTED_GROUPS:
            group_df = model_df[model_df["grupo"] == group]
            row[group] = int(group_df["question_id"].nunique(dropna=True))
        row["total_linhas"] = int(len(model_df))
        row["comparacao_h1_ok"] = row["RAG"] == EXPECTED_QUESTIONS_PER_GROUP and row["GraphRAG"] == EXPECTED_QUESTIONS_PER_GROUP
        row["comparacao_ontologia_ok"] = row["GraphRAG"] == EXPECTED_QUESTIONS_PER_GROUP and row["GraphRAG + Ontologia"] == EXPECTED_QUESTIONS_PER_GROUP
        row["status"] = (
            "OK"
            if all(row[group] == EXPECTED_QUESTIONS_PER_GROUP for group in EXPECTED_GROUPS)
            else "INCOMPLETO"
        )
        rows.append(row)

    result = pd.DataFrame(rows, columns=columns)
    write_csv(result, output_dir / "auditoria_cobertura_por_modelo.csv", columns)
    return result


def audit_result_files(resultados_dir: Path, output_dir: Path, errors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(resultados_dir.rglob("*-completo.csv")):
        dataframe = read_csv_safely(path, errors)
        total_nodes = numeric_sum(dataframe, "total_graph_data_nodes")
        total_edges = numeric_sum(dataframe, "total_edges")
        total_valid = numeric_sum(dataframe, "total_ontology_valid_true")
        question_count = (
            int(dataframe["question_id"].nunique(dropna=True))
            if "question_id" in dataframe.columns
            else math.nan
        )

        try:
            relative_parent = path.parent.relative_to(resultados_dir)
        except ValueError:
            relative_parent = path.parent

        row = {
            "modelo": detect_model(path),
            "grupo_detectado": detect_analysis_group(path),
            "pasta": relative_parent.as_posix(),
            "caminho_csv": path.as_posix(),
            "nome_arquivo": path.name,
            "tamanho_bytes": path.stat().st_size if path.exists() else math.nan,
            "hash_md5": compute_file_hash(path) if path.exists() else "",
            "batch_id": extract_batch_id(path),
            "quantidade_linhas": int(len(dataframe)),
            "quantidade_question_id_distintos": question_count,
            "possui_colunas_de_metricas_deepeval": has_deepeval_metric_columns(dataframe),
            "possui_colunas_estruturais_matching": all(column in dataframe.columns for column in STRUCTURAL_COLUMNS),
            "total_graph_data_nodes": total_nodes,
            "total_edges": total_edges,
            "total_ontology_valid_true": total_valid,
            "taxa_matching": safe_percentage(total_valid, total_nodes),
        }
        rows.append(row)

    result = pd.DataFrame(rows, columns=RESULT_FILE_COLUMNS)
    write_csv(result, output_dir / "auditoria_arquivos_resultados.csv", RESULT_FILE_COLUMNS)
    return result


def audit_duplicate_batches(result_files: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    columns = ["batch_id", "quantidade_ocorrencias", "modelos", "grupos", "caminhos"]
    if result_files.empty or "batch_id" not in result_files.columns:
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_batches_duplicados.csv", columns)
        return result

    rows: list[dict[str, Any]] = []
    valid = result_files[result_files["batch_id"].fillna("").astype(str).str.strip() != ""]
    for batch_id, group in valid.groupby("batch_id", dropna=False):
        unique_paths = group["caminho_csv"].nunique(dropna=True)
        unique_groups = group["grupo_detectado"].nunique(dropna=True)
        unique_folders = group["pasta"].nunique(dropna=True)
        if len(group) > 1 and (unique_paths > 1 or unique_groups > 1 or unique_folders > 1):
            rows.append(
                {
                    "batch_id": batch_id,
                    "quantidade_ocorrencias": int(len(group)),
                    "modelos": unique_join(group["modelo"]),
                    "grupos": unique_join(group["grupo_detectado"]),
                    "caminhos": unique_join(group["caminho_csv"]),
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    write_csv(result, output_dir / "auditoria_batches_duplicados.csv", columns)
    return result


def audit_duplicate_hashes(result_files: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    columns = ["hash_md5", "quantidade_ocorrencias", "modelos", "grupos", "caminhos", "tamanhos"]
    if result_files.empty or "hash_md5" not in result_files.columns:
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_hashes_duplicados.csv", columns)
        return result

    rows: list[dict[str, Any]] = []
    valid = result_files[result_files["hash_md5"].fillna("").astype(str).str.strip() != ""]
    for hash_md5, group in valid.groupby("hash_md5", dropna=False):
        if group["caminho_csv"].nunique(dropna=True) > 1:
            rows.append(
                {
                    "hash_md5": hash_md5,
                    "quantidade_ocorrencias": int(len(group)),
                    "modelos": unique_join(group["modelo"]),
                    "grupos": unique_join(group["grupo_detectado"]),
                    "caminhos": unique_join(group["caminho_csv"]),
                    "tamanhos": unique_join(group["tamanho_bytes"].astype(str)),
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    write_csv(result, output_dir / "auditoria_hashes_duplicados.csv", columns)
    return result


def audit_missing_metrics(avaliacao_dir: Path, output_dir: Path, errors: list[str]) -> pd.DataFrame:
    path = avaliacao_dir / DETAILS_FILE
    dataframe = read_csv_safely(path, errors)
    columns = [
        "tipo_registro",
        "modelo",
        "grupo",
        "tipo_resposta",
        "question_id",
        "coluna",
        "valor_ausente",
        "eval_error",
        "total_ausentes",
        "percentual_ausente",
    ]
    if dataframe.empty:
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_metricas_ausentes.csv", columns)
        return result

    rows: list[dict[str, Any]] = []
    total_rows = len(dataframe)
    for column in REQUIRED_DETAIL_COLUMNS:
        if column not in dataframe.columns:
            total_missing = total_rows
            mask = pd.Series([True] * total_rows, index=dataframe.index)
        elif column == "eval_error":
            total_missing = 0
            mask = pd.Series([False] * total_rows, index=dataframe.index)
        else:
            mask = is_missing(dataframe[column])
            total_missing = int(mask.sum())

        rows.append(
            {
                "tipo_registro": "resumo_coluna",
                "modelo": "",
                "grupo": "",
                "tipo_resposta": "",
                "question_id": "",
                "coluna": column,
                "valor_ausente": "",
                "eval_error": "",
                "total_ausentes": total_missing,
                "percentual_ausente": safe_percentage(total_missing, total_rows),
            }
        )

        if total_missing == 0:
            continue

        for index, row in dataframe[mask].iterrows():
            source = row.get("arquivo_fonte", "")
            response_type = row.get("tipo_resposta", "")
            rows.append(
                {
                    "tipo_registro": "detalhe",
                    "modelo": detect_model(source),
                    "grupo": detect_analysis_group(source, response_type),
                    "tipo_resposta": response_type,
                    "question_id": row.get("question_id", ""),
                    "coluna": column,
                    "valor_ausente": True,
                    "eval_error": row.get("eval_error", ""),
                    "total_ausentes": "",
                    "percentual_ausente": "",
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    write_csv(result, output_dir / "auditoria_metricas_ausentes.csv", columns)
    return result


def audit_summary_consistency(avaliacao_dir: Path, output_dir: Path, errors: list[str]) -> pd.DataFrame:
    summary_path = avaliacao_dir / SUMMARY_FILE
    details_path = avaliacao_dir / DETAILS_FILE
    summary = read_csv_safely(summary_path, errors)
    details = read_csv_safely(details_path, errors)
    columns = [
        "arquivo_fonte",
        "tipo_resposta",
        "metrica",
        "valor_resumo",
        "valor_recalculado",
        "diferenca_absoluta",
        "status",
    ]
    if summary.empty or details.empty:
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_consistencia_resumo_vs_detalhado.csv", columns)
        return result

    key_columns = ["arquivo_fonte", "tipo_resposta"]
    for dataframe, name in [(summary, SUMMARY_FILE), (details, DETAILS_FILE)]:
        missing = [column for column in key_columns if column not in dataframe.columns]
        if missing:
            errors.append(f"{name} nao possui colunas obrigatorias: {missing}")
            result = pd.DataFrame(columns=columns)
            write_csv(result, output_dir / "auditoria_consistencia_resumo_vs_detalhado.csv", columns)
            return result

    summary_index = summary.set_index(key_columns, drop=False)
    detail_groups = details.groupby(key_columns, dropna=False)
    all_keys = sorted(set(summary_index.index).union(set(detail_groups.groups.keys())))

    rows: list[dict[str, Any]] = []
    for key in all_keys:
        key_tuple = key if isinstance(key, tuple) else (key, "")
        source, response_type = key_tuple
        summary_rows = summary_index.loc[[key_tuple]] if key_tuple in summary_index.index else pd.DataFrame()
        detail_rows = details.loc[detail_groups.groups[key_tuple]] if key_tuple in detail_groups.groups else pd.DataFrame()

        for metric in SUMMARY_COMPARISON_METRICS:
            summary_column = f"{metric}_mean"
            status = "OK"
            summary_value = math.nan
            recalculated_value = math.nan
            difference = math.nan

            if summary_rows.empty:
                status = "AUSENTE_NO_RESUMO"
            elif detail_rows.empty:
                status = "AUSENTE_NO_DETALHADO"
                if summary_column in summary_rows.columns:
                    summary_value = pd.to_numeric(summary_rows[summary_column], errors="coerce").mean()
            elif summary_column not in summary_rows.columns or metric not in detail_rows.columns:
                status = "COLUNA_AUSENTE"
                if summary_column in summary_rows.columns:
                    summary_value = pd.to_numeric(summary_rows[summary_column], errors="coerce").mean()
                if metric in detail_rows.columns:
                    recalculated_value = pd.to_numeric(detail_rows[metric], errors="coerce").mean()
            else:
                summary_value = pd.to_numeric(summary_rows[summary_column], errors="coerce").mean()
                recalculated_value = pd.to_numeric(detail_rows[metric], errors="coerce").mean()
                if pd.isna(summary_value) and pd.isna(recalculated_value):
                    difference = 0.0
                else:
                    difference = abs(float(summary_value) - float(recalculated_value))
                status = "OK" if difference <= TOLERANCE else "DIVERGENTE"

            rows.append(
                {
                    "arquivo_fonte": source,
                    "tipo_resposta": response_type,
                    "metrica": metric,
                    "valor_resumo": summary_value,
                    "valor_recalculado": recalculated_value,
                    "diferenca_absoluta": difference,
                    "status": status,
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    write_csv(result, output_dir / "auditoria_consistencia_resumo_vs_detalhado.csv", columns)
    return result


def audit_structural_matching(result_files: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    columns = [
        "modelo",
        "grupo",
        "caminho_csv",
        "batch_id",
        "linhas",
        "nodes_totais",
        "edges_totais",
        "matches_validos_totais",
        "nodes_medios_por_pergunta",
        "edges_medios_por_pergunta",
        "matches_validos_medios_por_pergunta",
        "taxa_matching",
        "alertas",
    ]
    if result_files.empty:
        result = pd.DataFrame(columns=columns)
        write_csv(result, output_dir / "auditoria_matching_estrutural.csv", columns)
        return result

    structural = result_files[result_files["possui_colunas_estruturais_matching"] == True].copy()
    rows: list[dict[str, Any]] = []
    for _, row in structural.iterrows():
        denominator = row.get("quantidade_question_id_distintos")
        if pd.isna(denominator) or denominator == 0:
            denominator = row.get("quantidade_linhas", 0)
        denominator = float(denominator) if denominator else math.nan

        nodes = float(row.get("total_graph_data_nodes", math.nan))
        edges = float(row.get("total_edges", math.nan))
        matches = float(row.get("total_ontology_valid_true", math.nan))
        alerts: list[str] = []
        if nodes == 0:
            alerts.append("nodes_totais_zero")
        if not pd.isna(matches) and not pd.isna(nodes) and matches > nodes:
            alerts.append("matches_validos_maior_que_nodes")
        rate = safe_percentage(matches, nodes)
        if not pd.isna(rate) and rate > 100:
            alerts.append("taxa_matching_maior_que_100")

        rows.append(
            {
                "modelo": row.get("modelo", ""),
                "grupo": row.get("grupo_detectado", ""),
                "caminho_csv": row.get("caminho_csv", ""),
                "batch_id": row.get("batch_id", ""),
                "linhas": row.get("quantidade_linhas", 0),
                "nodes_totais": nodes,
                "edges_totais": edges,
                "matches_validos_totais": matches,
                "nodes_medios_por_pergunta": nodes / denominator if denominator and not pd.isna(denominator) else math.nan,
                "edges_medios_por_pergunta": edges / denominator if denominator and not pd.isna(denominator) else math.nan,
                "matches_validos_medios_por_pergunta": matches / denominator if denominator and not pd.isna(denominator) else math.nan,
                "taxa_matching": rate,
                "alertas": ";".join(alerts),
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        write_csv(result, output_dir / "auditoria_matching_estrutural.csv", columns)
        return result

    triple_columns = ["nodes_totais", "edges_totais", "matches_validos_totais"]
    duplicated_triples = result.duplicated(triple_columns, keep=False)
    for index in result[duplicated_triples].index:
        current = result.at[index, "alertas"]
        suffix = "totais_estruturais_repetidos"
        result.at[index, "alertas"] = f"{current};{suffix}" if current else suffix

    valid_batches = result[result["batch_id"].fillna("").astype(str).str.strip() != ""]
    duplicate_batch_ids = set()
    for batch_id, group in valid_batches.groupby("batch_id", dropna=False):
        if group["grupo"].nunique(dropna=True) > 1:
            duplicate_batch_ids.add(batch_id)
    for index in result[result["batch_id"].isin(duplicate_batch_ids)].index:
        current = result.at[index, "alertas"]
        suffix = "batch_id_em_multiplas_configuracoes"
        result.at[index, "alertas"] = f"{current};{suffix}" if current else suffix

    write_csv(result, output_dir / "auditoria_matching_estrutural.csv", columns)
    return result


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_Sem registros._"
    try:
        return dataframe.to_markdown(index=False)
    except Exception:
        return dataframe.to_string(index=False)


def count_missing_metric_details(missing_metrics: pd.DataFrame) -> int:
    if missing_metrics.empty or "tipo_registro" not in missing_metrics.columns:
        return 0
    return int((missing_metrics["tipo_registro"] == "detalhe").sum())


def get_summary_value(missing_metrics: pd.DataFrame, column: str) -> int:
    if missing_metrics.empty:
        return 0
    rows = missing_metrics[
        (missing_metrics["tipo_registro"] == "resumo_coluna")
        & (missing_metrics["coluna"] == column)
    ]
    if rows.empty:
        return 0
    return int(pd.to_numeric(rows["total_ausentes"], errors="coerce").fillna(0).sum())


def write_markdown_report(
    output_dir: Path,
    coverage: pd.DataFrame,
    result_files: pd.DataFrame,
    duplicate_batches: pd.DataFrame,
    duplicate_hashes: pd.DataFrame,
    missing_metrics: pd.DataFrame,
    summary_consistency: pd.DataFrame,
    structural_matching: pd.DataFrame,
    errors: list[str],
) -> dict[str, int]:
    incomplete_models = coverage[coverage["status"] != "OK"] if not coverage.empty else pd.DataFrame()
    missing_metric_details = count_missing_metric_details(missing_metrics)
    divergent_summary = (
        summary_consistency[summary_consistency["status"] != "OK"]
        if not summary_consistency.empty
        else pd.DataFrame()
    )
    suspicious_matching = (
        structural_matching[structural_matching["alertas"].fillna("").astype(str).str.strip() != ""]
        if not structural_matching.empty and "alertas" in structural_matching.columns
        else pd.DataFrame()
    )

    critical_counts = {
        "batches_duplicados": int(len(duplicate_batches)),
        "hashes_duplicados": int(len(duplicate_hashes)),
        "modelos_incompletos": int(len(incomplete_models)),
        "metricas_ausentes": missing_metric_details,
        "divergencias_resumo": int(len(divergent_summary)),
        "matching_suspeito": int(len(suspicious_matching)),
        "erros_leitura": int(len(errors)),
    }
    has_critical_alerts = any(value > 0 for value in critical_counts.values())

    table_9_safe = critical_counts["matching_suspeito"] == 0 and not structural_matching.empty
    final_tables_ready = not has_critical_alerts

    missing_summary = missing_metrics[
        missing_metrics.get("tipo_registro", pd.Series(dtype=str)) == "resumo_coluna"
    ].copy()
    if not missing_summary.empty:
        missing_summary = missing_summary[
            ["coluna", "total_ausentes", "percentual_ausente"]
        ].sort_values("total_ausentes", ascending=False)

    suspected_files: set[str] = set()
    for dataframe, column in [
        (duplicate_hashes, "caminhos"),
        (duplicate_batches, "caminhos"),
        (suspicious_matching, "caminho_csv"),
    ]:
        if dataframe.empty or column not in dataframe.columns:
            continue
        for value in dataframe[column].dropna().astype(str):
            for path in value.split(" | "):
                if path.strip():
                    suspected_files.add(path.strip())

    report_lines = [
        "# Relatorio de auditoria",
        "",
        "## 1. Resumo executivo",
        "",
        f"- Conclusao: {'AUDITORIA COM ALERTAS' if has_critical_alerts else 'AUDITORIA OK'}",
        f"- Tabela 9 pode ser usada com seguranca: {'sim' if table_9_safe else 'nao'}",
        f"- Dados prontos para gerar as tabelas finais do artigo: {'sim' if final_tables_ready else 'nao'}",
        f"- Erros de leitura/processamento registrados: {len(errors)}",
        "",
        "## 2. Quantidade de modelos encontrados",
        "",
        str(int(coverage["modelo"].nunique())) if not coverage.empty else "0",
        "",
        "## 3. Quantidade total de arquivos `*-completo.csv`",
        "",
        str(int(len(result_files))),
        "",
        "## 4. Cobertura por modelo",
        "",
        dataframe_to_markdown(coverage),
        "",
        "## 5. Alertas criticos",
        "",
        f"- Batches duplicados: {critical_counts['batches_duplicados']}",
        f"- Hashes duplicados: {critical_counts['hashes_duplicados']}",
        f"- Modelos incompletos: {critical_counts['modelos_incompletos']}",
        f"- Metricas ausentes: {critical_counts['metricas_ausentes']}",
        f"- Divergencias entre resumo e detalhado: {critical_counts['divergencias_resumo']}",
        f"- Matching estrutural suspeito: {critical_counts['matching_suspeito']}",
        f"- Erros de leitura/processamento: {critical_counts['erros_leitura']}",
        "",
        "### Modelos incompletos",
        "",
        dataframe_to_markdown(incomplete_models),
        "",
        "### Batches duplicados",
        "",
        dataframe_to_markdown(duplicate_batches),
        "",
        "### Hashes duplicados",
        "",
        dataframe_to_markdown(duplicate_hashes),
        "",
        "### Resumo de metricas ausentes por coluna",
        "",
        dataframe_to_markdown(missing_summary),
        "",
        "### Divergencias entre resumo e detalhado",
        "",
        dataframe_to_markdown(divergent_summary.head(100)),
        "",
        "### Matching estrutural suspeito",
        "",
        dataframe_to_markdown(suspicious_matching),
        "",
        "## 6. Lista de arquivos suspeitos",
        "",
    ]

    if suspected_files:
        report_lines.extend(f"- {path}" for path in sorted(suspected_files))
    else:
        report_lines.append("_Nenhum arquivo suspeito identificado pelos criterios da auditoria._")

    report_lines.extend(
        [
            "",
            "## 7. Erros registrados",
            "",
        ]
    )
    if errors:
        report_lines.extend(f"- {error}" for error in errors)
    else:
        report_lines.append("_Nenhum erro registrado._")

    report_lines.extend(
        [
            "",
            "## 8. Conclusao",
            "",
            "AUDITORIA COM ALERTAS" if has_critical_alerts else "AUDITORIA OK",
            "",
            (
                "Ha alertas criticos a revisar antes de usar a Tabela 9 e antes de "
                "gerar as tabelas finais do artigo."
                if has_critical_alerts
                else "Todos os criterios criticos passaram; os dados estao prontos para as tabelas finais do artigo."
            ),
        ]
    )

    output_path = output_dir / "relatorio_auditoria.md"
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return critical_counts


def print_terminal_summary(critical_counts: dict[str, int], report_path: Path) -> None:
    has_alerts = any(value > 0 for value in critical_counts.values())
    print("\nResumo da auditoria")
    print(f"Status: {'AUDITORIA COM ALERTAS' if has_alerts else 'AUDITORIA OK'}")
    for key, value in critical_counts.items():
        print(f"- {key}: {value}")
    print(f"Relatorio: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita resultados de avaliacao RAG/GraphRAG.")
    parser.add_argument("--resultados-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--avaliacao-dir", type=Path, default=Path("avaliacao"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resultados_dir = args.resultados_dir
    avaliacao_dir = args.avaliacao_dir
    output_dir = avaliacao_dir / "auditoria"
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    if not resultados_dir.exists():
        errors.append(f"Pasta de resultados nao encontrada: {resultados_dir}")
    if not avaliacao_dir.exists():
        errors.append(f"Pasta de avaliacao nao encontrada: {avaliacao_dir}")

    coverage = audit_coverage(avaliacao_dir, output_dir, errors)
    result_files = audit_result_files(resultados_dir, output_dir, errors)
    duplicate_batches = audit_duplicate_batches(result_files, output_dir)
    duplicate_hashes = audit_duplicate_hashes(result_files, output_dir)
    missing_metrics = audit_missing_metrics(avaliacao_dir, output_dir, errors)
    summary_consistency = audit_summary_consistency(avaliacao_dir, output_dir, errors)
    structural_matching = audit_structural_matching(result_files, output_dir)

    critical_counts = write_markdown_report(
        output_dir=output_dir,
        coverage=coverage,
        result_files=result_files,
        duplicate_batches=duplicate_batches,
        duplicate_hashes=duplicate_hashes,
        missing_metrics=missing_metrics,
        summary_consistency=summary_consistency,
        structural_matching=structural_matching,
        errors=errors,
    )
    print_terminal_summary(critical_counts, output_dir / "relatorio_auditoria.md")


if __name__ == "__main__":
    main()
