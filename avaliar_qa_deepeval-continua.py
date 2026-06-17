#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import math
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

INPUT_COLUMNS = [
    "arquivo_fonte",
    "tipo_resposta",
    "question_id",
    "question",
    "expected_question",
    "context",
    "llm_answer",
]

REQUIRED_INPUT_COLUMNS = [
    column for column in INPUT_COLUMNS if column != "tipo_resposta"
]

RESULT_COLUMNS = [
    "answer_relevancy_score",
    "answer_relevancy_reason",
    "faithfulness_score",
    "faithfulness_reason",
    "contextual_relevancy_score",
    "contextual_relevancy_reason",
    "contextual_recall_score",
    "contextual_recall_reason",
    "correctness_score",
    "correctness_reason",
    "completeness_score",
    "completeness_reason",
    "geological_adequacy_score",
    "geological_adequacy_reason",
    "ontology_alignment_score",
    "ontology_alignment_reason",
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
    "eval_error",
]

NUMERIC_COLUMNS = [
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

REASON_COLUMNS = [
    "answer_relevancy_reason",
    "faithfulness_reason",
    "contextual_relevancy_reason",
    "contextual_recall_reason",
    "correctness_reason",
    "completeness_reason",
    "geological_adequacy_reason",
    "ontology_alignment_reason",
    "eval_error",
]


@dataclass
class MetricAdapter:
    label: str
    score_column: str
    reason_column: str
    metric: Any
    requires_context: bool = False
    requires_expected: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continua a avaliacao de um CSV consolidado de QA com DeepEval, "
            "GEval e metricas classicas a partir dos resultados ja salvos."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/todas_execucoes_deepeval.csv"),
        help="Arquivo CSV de entrada com as colunas originais do QA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("avaliacao/todas_execucoes_deepeval_avaliadas.csv"),
        help="Arquivo CSV de saida com as colunas originais e as metricas por linha.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("avaliacao/resumo_metricas_por_execucao.csv"),
        help="Arquivo CSV agregado por arquivo_fonte.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o numero de novas linhas processadas para testes.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help=(
            "Registra mensagem de progresso a cada N linhas. "
            "O CSV de saida e salvo a cada resultado. Padrao: 1."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Sobrescreve o modelo do judge Azure definido em LLM_MODEL no .env. "
            "Exemplo: gpt-5-4-petrobras."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de log no terminal. Se omitido, usa LOG_LEVEL do .env.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def resolve_existing_input_path(path: Path) -> Path:
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(PROJECT_ROOT / path)
        if len(path.parts) == 1:
            candidates.append(PROJECT_ROOT / "output" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Arquivo de entrada nao encontrado. Caminhos testados: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def ensure_distinct_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    unique = set(resolved)
    if len(unique) != len(resolved):
        raise ValueError("Os caminhos de entrada/saida nao podem apontar para o mesmo arquivo.")


def load_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pandas as pd
        from rouge_score import rouge_scorer
        from tqdm import tqdm
    except ImportError as exc:
        missing = getattr(exc, "name", "dependencia desconhecida")
        raise RuntimeError(
            f"Dependencia ausente: {missing}. Instale com `pip install -r requirements.txt`."
        ) from exc

    return pd, rouge_scorer, tqdm


def load_project_dependencies() -> tuple[Any, Any]:
    try:
        from config import Config
        from deepeval_azure_model import AzureDeepEvalModel
    except ImportError as exc:
        missing = getattr(exc, "name", "dependencia do projeto")
        raise RuntimeError(
            f"Dependencia ausente: {missing}. Instale com `pip install -r requirements.txt`."
        ) from exc

    return Config, AzureDeepEvalModel


def load_deepeval_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        missing = getattr(exc, "name", "deepeval")
        raise RuntimeError(
            f"Dependencia ausente: {missing}. Instale com `pip install -r requirements.txt`."
        ) from exc

    return (
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        SingleTurnParams,
        LLMTestCase,
    )


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_for_qa(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", safe_text(text)).lower()
    without_punctuation = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def compute_exact_match(expected: str, actual: str) -> float:
    return float(normalize_for_qa(expected) == normalize_for_qa(actual))


def compute_token_f1(expected: str, actual: str) -> float:
    expected_tokens = normalize_for_qa(expected).split()
    actual_tokens = normalize_for_qa(actual).split()

    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0

    overlap = Counter(expected_tokens) & Counter(actual_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(actual_tokens)
    recall = common / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def make_test_case(
    row: dict[str, Any],
    LLMTestCase: Any,
) -> tuple[Any, str, str, str, list[str]]:
    question = safe_text(row.get("question"))
    actual_output = safe_text(row.get("llm_answer"))
    expected_output = safe_text(row.get("expected_question"))
    context = safe_text(row.get("context"))
    retrieval_context = [context] if context else []

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        context=retrieval_context,
        retrieval_context=retrieval_context,
    )
    return test_case, question, actual_output, expected_output, retrieval_context


def build_metric_adapters(azure_judge_model: Any) -> tuple[list[MetricAdapter], Any]:
    (
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        SingleTurnParams,
        LLMTestCase,
    ) = load_deepeval_dependencies()

    common_native_kwargs: dict[str, Any] = {
        "include_reason": True,
        "async_mode": False,
        "verbose_mode": False,
        "model": azure_judge_model,
    }
    common_geval_kwargs: dict[str, Any] = {
        "async_mode": False,
        "verbose_mode": False,
        "model": azure_judge_model,
    }

    adapters = [
        MetricAdapter(
            label="AnswerRelevancyMetric",
            score_column="answer_relevancy_score",
            reason_column="answer_relevancy_reason",
            metric=AnswerRelevancyMetric(**common_native_kwargs),
        ),
        MetricAdapter(
            label="FaithfulnessMetric",
            score_column="faithfulness_score",
            reason_column="faithfulness_reason",
            metric=FaithfulnessMetric(**common_native_kwargs),
            requires_context=True,
        ),
        MetricAdapter(
            label="ContextualRelevancyMetric",
            score_column="contextual_relevancy_score",
            reason_column="contextual_relevancy_reason",
            metric=ContextualRelevancyMetric(**common_native_kwargs),
            requires_context=True,
        ),
        MetricAdapter(
            label="ContextualRecallMetric",
            score_column="contextual_recall_score",
            reason_column="contextual_recall_reason",
            metric=ContextualRecallMetric(**common_native_kwargs),
            requires_context=True,
            requires_expected=True,
        ),
        MetricAdapter(
            label="GEval Correctness",
            score_column="correctness_score",
            reason_column="correctness_reason",
            metric=GEval(
                name="Correctness",
                evaluation_steps=[
                    "Compare the actual output against the expected output for the given input.",
                    "Check whether the actual output contains factual contradictions relative to the expected output.",
                    "Penalize clear factual mistakes heavily.",
                    "Do not penalize harmless stylistic or wording differences when the meaning stays correct.",
                ],
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                **common_geval_kwargs,
            ),
            requires_expected=True,
        ),
        MetricAdapter(
            label="GEval Completeness",
            score_column="completeness_score",
            reason_column="completeness_reason",
            metric=GEval(
                name="Completeness",
                evaluation_steps=[
                    "Identify the important facts, entities, and relations present in the expected output for the given input.",
                    "Check whether the actual output covers all essential elements from the expected output.",
                    "Penalize omissions of important information strongly.",
                    "Do not penalize concise wording if all essential content is still covered.",
                ],
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                **common_geval_kwargs,
            ),
            requires_expected=True,
        ),
        MetricAdapter(
            label="GEval Geological Adequacy",
            score_column="geological_adequacy_score",
            reason_column="geological_adequacy_reason",
            metric=GEval(
                name="Geological Adequacy",
                evaluation_steps=[
                    "Assess whether the actual output uses geological terminology and concepts appropriately for the input.",
                    "Compare the geological entities, relations, and interpretations in the actual output with the expected output.",
                    "Penalize misuse of geological concepts, lithologies, facies, stratigraphic terms, reservoir notions, or domain relations.",
                    "Reward precise, domain-appropriate, and semantically coherent geological language.",
                ],
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                **common_geval_kwargs,
            ),
            requires_expected=True,
        ),
        MetricAdapter(
            label="GEval Ontology Alignment",
            score_column="ontology_alignment_score",
            reason_column="ontology_alignment_reason",
            metric=GEval(
                name="Ontology Alignment",
                evaluation_steps=[
                    "Compare the actual output with the expected output for the given input, focusing on semantic alignment rather than wording overlap.",
                    "Check whether the actual output stays aligned with the expected geological concepts, entities, classes, and relations.",
                    "Penalize concept confusion, wrong semantic roles, inverted relations, or misuse of domain-specific meanings.",
                    "When the actual output is concise, judge whether it still preserves the intended geological semantics.",
                ],
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                **common_geval_kwargs,
            ),
            requires_expected=True,
        ),
    ]
    return adapters, LLMTestCase


def initialize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {column: row.get(column, "") for column in INPUT_COLUMNS}
    for column in RESULT_COLUMNS:
        result[column] = ""
    for column in NUMERIC_COLUMNS:
        result[column] = math.nan
    return result


def metric_reason_or_empty(metric: Any) -> str:
    return safe_text(getattr(metric, "reason", ""))


def metric_score_or_nan(metric: Any) -> float:
    score = getattr(metric, "score", math.nan)
    if score is None:
        return math.nan
    try:
        return float(score)
    except (TypeError, ValueError):
        return math.nan


def evaluate_metric(
    adapter: MetricAdapter,
    test_case: Any,
    result_row: dict[str, Any],
    has_context: bool,
    has_expected: bool,
    errors: list[str],
) -> None:
    if adapter.requires_context and not has_context:
        result_row[adapter.score_column] = math.nan
        result_row[adapter.reason_column] = "Skipped: empty context."
        return

    if adapter.requires_expected and not has_expected:
        result_row[adapter.score_column] = math.nan
        result_row[adapter.reason_column] = "Skipped: empty expected_question."
        return

    try:
        adapter.metric.measure(test_case)
        result_row[adapter.score_column] = metric_score_or_nan(adapter.metric)
        result_row[adapter.reason_column] = metric_reason_or_empty(adapter.metric)
    except Exception as exc:  # pragma: no cover - depends on runtime/model/provider
        result_row[adapter.score_column] = math.nan
        result_row[adapter.reason_column] = ""
        errors.append(f"{adapter.label}: {exc}")


def evaluate_classic_metrics(
    expected_output: str,
    actual_output: str,
    rouge_scorer_instance: Any,
    result_row: dict[str, Any],
    errors: list[str],
) -> None:
    try:
        rouge_scores = rouge_scorer_instance.score(expected_output, actual_output)
        result_row["rouge1_precision"] = float(rouge_scores["rouge1"].precision)
        result_row["rouge1_recall"] = float(rouge_scores["rouge1"].recall)
        result_row["rouge1_fmeasure"] = float(rouge_scores["rouge1"].fmeasure)
        result_row["rouge2_precision"] = float(rouge_scores["rouge2"].precision)
        result_row["rouge2_recall"] = float(rouge_scores["rouge2"].recall)
        result_row["rouge2_fmeasure"] = float(rouge_scores["rouge2"].fmeasure)
        result_row["rougeL_precision"] = float(rouge_scores["rougeL"].precision)
        result_row["rougeL_recall"] = float(rouge_scores["rougeL"].recall)
        result_row["rougeL_fmeasure"] = float(rouge_scores["rougeL"].fmeasure)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"ROUGE: {exc}")

    try:
        result_row["exact_match"] = compute_exact_match(expected_output, actual_output)
        result_row["token_f1"] = compute_token_f1(expected_output, actual_output)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"EM/F1: {exc}")


def evaluate_row(
    row: dict[str, Any],
    metric_adapters: list[MetricAdapter],
    LLMTestCase: Any,
    rouge_scorer_instance: Any,
) -> dict[str, Any]:
    result_row = initialize_result_row(row)
    errors: list[str] = []

    try:
        test_case, _question, actual_output, expected_output, retrieval_context = make_test_case(
            row,
            LLMTestCase,
        )
    except Exception as exc:  # pragma: no cover - defensive
        result_row["eval_error"] = f"LLMTestCase: {exc}"
        return result_row

    has_context = bool(retrieval_context)
    has_expected = bool(expected_output)

    for adapter in metric_adapters:
        evaluate_metric(
            adapter=adapter,
            test_case=test_case,
            result_row=result_row,
            has_context=has_context,
            has_expected=has_expected,
            errors=errors,
        )

    evaluate_classic_metrics(
        expected_output=expected_output,
        actual_output=actual_output,
        rouge_scorer_instance=rouge_scorer_instance,
        result_row=result_row,
        errors=errors,
    )

    result_row["eval_error"] = " | ".join(errors)
    return result_row


def validate_input_columns(dataframe: Any) -> None:
    missing_columns = [
        column for column in REQUIRED_INPUT_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "O CSV de entrada nao possui todas as colunas esperadas. Faltando: "
            + ", ".join(missing_columns)
        )


def validate_result_columns(dataframe: Any) -> None:
    expected_columns = INPUT_COLUMNS + RESULT_COLUMNS
    missing_columns = [
        column for column in expected_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "O CSV de resultados existente nao possui todas as colunas esperadas. "
            "Faltando: " + ", ".join(missing_columns)
        )


def build_results_dataframe(pd: Any, rows: list[dict[str, Any]]) -> Any:
    dataframe = pd.DataFrame(rows, columns=INPUT_COLUMNS + RESULT_COLUMNS)
    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    for column in REASON_COLUMNS + INPUT_COLUMNS:
        dataframe[column] = dataframe[column].fillna("").astype(str)
    return dataframe


def build_summary_dataframe(pd: Any, results_df: Any) -> Any:
    grouped = results_df.groupby(["arquivo_fonte", "tipo_resposta"], sort=False)
    counts_df = grouped.size().rename("num_linhas").to_frame()
    metrics_summary = grouped[NUMERIC_COLUMNS].agg(["mean", "median"])
    metrics_std = grouped[NUMERIC_COLUMNS].std(ddof=0)
    metrics_std.columns = pd.MultiIndex.from_product([metrics_std.columns, ["std"]])
    metrics_summary = metrics_summary.join(metrics_std)
    metrics_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in metrics_summary.columns.to_flat_index()
    ]
    return counts_df.join(metrics_summary).reset_index()


def write_csv_atomically(dataframe: Any, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)


def save_outputs(
    pd: Any,
    rows: list[dict[str, Any]],
    output_path: Path,
    summary_path: Path,
) -> tuple[Any, Any]:
    results_df = build_results_dataframe(pd, rows)
    summary_df = build_summary_dataframe(pd, results_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv_atomically(results_df, output_path)
    write_csv_atomically(summary_df, summary_path)

    return results_df, summary_df


def read_input_dataframe(pd: Any, input_path: Path) -> Any:
    dataframe = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    validate_input_columns(dataframe)
    for column in INPUT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""

    return dataframe


def read_existing_results_dataframe(pd: Any, output_path: Path) -> Any:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return build_results_dataframe(pd, [])

    dataframe = pd.read_csv(
        output_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    validate_result_columns(dataframe)
    return dataframe[INPUT_COLUMNS + RESULT_COLUMNS].copy()


def determine_resume_index(input_df: Any, existing_results_df: Any) -> int:
    completed_count = len(existing_results_df)
    if completed_count == 0:
        return 0
    if completed_count > len(input_df):
        raise ValueError(
            "O CSV de resultados possui mais linhas do que o CSV de entrada "
            f"({completed_count} > {len(input_df)})."
        )

    completed_input = existing_results_df[INPUT_COLUMNS].reset_index(drop=True)
    expected_prefix = input_df.iloc[:completed_count][INPUT_COLUMNS].reset_index(
        drop=True
    )
    matching_rows = completed_input.eq(expected_prefix).all(axis=1)
    if not bool(matching_rows.all()):
        mismatch_index = int(matching_rows[~matching_rows].index[0])
        raise ValueError(
            "O CSV de resultados nao corresponde ao inicio do CSV de entrada. "
            f"Primeira divergencia na linha de dados {mismatch_index + 1}. "
            "A retomada foi cancelada para evitar duplicacao ou mistura de dados."
        )

    return completed_count


def select_pending_dataframe(
    dataframe: Any,
    resume_index: int,
    limit: int | None,
) -> Any:
    pending_df = dataframe.iloc[resume_index:]
    if limit is not None:
        pending_df = pending_df.head(max(limit, 0))
    return pending_df.copy()


def main() -> int:
    args = parse_args()
    try:
        pd, rouge_scorer, tqdm = load_runtime_dependencies()
        Config, AzureDeepEvalModel = load_project_dependencies()
        config_log_level = args.log_level or Config.LOG_LEVEL
        configure_logging(config_log_level)
        Config.validate()
    except RuntimeError as exc:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        logging.error("%s", exc)
        return 1
    except ValueError as exc:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        logging.error("%s", exc)
        return 1

    if args.save_every <= 0:
        logging.error("--save-every deve ser maior que zero.")
        return 1

    try:
        input_path = resolve_existing_input_path(args.input)
        output_path = resolve_output_path(args.output).resolve()
        summary_path = resolve_output_path(args.summary).resolve()
        ensure_distinct_paths(input_path, output_path, summary_path)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Lendo dataset de entrada: %s", input_path)
    try:
        dataframe = read_input_dataframe(pd, input_path)
        existing_results_df = read_existing_results_dataframe(pd, output_path)
        resume_index = determine_resume_index(dataframe, existing_results_df)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logging.error("%s", exc)
        return 1

    pending_df = select_pending_dataframe(dataframe, resume_index, args.limit)
    evaluated_rows = existing_results_df.to_dict(orient="records")

    logging.info("Linhas totais na entrada: %d", len(dataframe))
    logging.info("Linhas ja avaliadas e preservadas: %d", resume_index)
    if resume_index:
        last_row = existing_results_df.iloc[-1]
        logging.info(
            "Ultimo item concluido: question_id=%s, tipo_resposta=%s, arquivo_fonte=%s",
            safe_text(last_row.get("question_id")),
            safe_text(last_row.get("tipo_resposta")),
            safe_text(last_row.get("arquivo_fonte")),
        )
    logging.info("Novas linhas selecionadas para avaliacao: %d", len(pending_df))

    if pending_df.empty:
        logging.info("Nao ha novas linhas para avaliar. Atualizando apenas o resumo.")
        results_df, summary_df = save_outputs(
            pd=pd,
            rows=evaluated_rows,
            output_path=output_path,
            summary_path=summary_path,
        )
        logging.info("Linhas avaliadas: %d", len(results_df))
        logging.info("Execucoes agregadas: %d", len(summary_df))
        return 0

    try:
        azure_judge_model = AzureDeepEvalModel(model_override=args.judge_model)
        metric_adapters, LLMTestCase = build_metric_adapters(azure_judge_model)
    except (RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Judge Azure configurado com modelo: %s", azure_judge_model.get_model_name())
    logging.info(
        "Os CSVs de avaliacao serao atualizados apos cada linha processada: %s e %s",
        output_path,
        summary_path,
    )
    if args.judge_model:
        logging.info("Override de modelo solicitado via CLI: %s", args.judge_model)

    rouge_scorer_instance = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rows = pending_df.to_dict(orient="records")

    for new_index, row in enumerate(
        tqdm(rows, desc="Avaliando linhas", unit="linha"),
        start=1,
    ):
        evaluated_rows.append(
            evaluate_row(
                row=row,
                metric_adapters=metric_adapters,
                LLMTestCase=LLMTestCase,
                rouge_scorer_instance=rouge_scorer_instance,
            )
        )

        save_outputs(
            pd=pd,
            rows=evaluated_rows,
            output_path=output_path,
            summary_path=summary_path,
        )
        total_completed = resume_index + new_index
        logging.debug(
            "CSVs de avaliacao atualizados apos a linha total %s.",
            total_completed,
        )

        if new_index % args.save_every == 0:
            logging.info(
                "Progresso salvo: %s novas linhas, %s linhas no total.",
                new_index,
                total_completed,
            )

    logging.info("Salvando arquivos finais...")
    results_df, summary_df = save_outputs(
        pd=pd,
        rows=evaluated_rows,
        output_path=output_path,
        summary_path=summary_path,
    )

    logging.info("Arquivo avaliado salvo em: %s", output_path)
    logging.info("Resumo agregado salvo em: %s", summary_path)
    logging.info("Linhas avaliadas: %d", len(results_df))
    logging.info("Execucoes agregadas: %d", len(summary_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
