#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "avaliacao"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "avaliacao" / "significancia"

# Edite esta lista para executar o script sem argumentos de linha de comando.
# Nomes simples sao procurados dentro da pasta avaliacao/.
ARQUIVOS_ENTRADA = [
    "todas_execucoes_deepeval_avaliadas_execucao15.csv",
    # "todas_execucoes_deepeval_avaliadas_execucao2.csv",
]

DEFAULT_METRIC_COLUMNS = [
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

GROUP_COLUMNS = ["arquivo_fonte", "tipo_resposta"]

DESCRIPTIVE_COLUMNS = [
    "arquivo_fonte",
    "tipo_resposta",
    "metrica",
    "n_perguntas",
    "n_valores",
    "media",
    "desvio_padrao",
    "mediana",
    "q1",
    "q3",
    "minimo",
    "maximo",
]

COMPARISON_COLUMNS = [
    "arquivo_fonte",
    "metrica",
    "tipo_a",
    "tipo_b",
    "n_pares",
    "n_diferencas_nao_zero",
    "media_tipo_a",
    "media_tipo_b",
    "mediana_tipo_a",
    "mediana_tipo_b",
    "delta_medio_b_menos_a",
    "delta_mediano_b_menos_a",
    "vitorias_tipo_a",
    "vitorias_tipo_b",
    "empates",
    "teste_principal",
    "estatistica_wilcoxon",
    "z_wilcoxon",
    "p_valor",
    "p_valor_teste_sinais",
    "efeito_rank_biserial",
    "p_valor_ajustado_holm",
    "alpha",
    "significativo",
    "direcao",
    "status",
]

COVERAGE_COLUMNS = [
    "arquivo_entrada",
    "linhas_lidas",
    "linhas_cabecalho_removidas",
    "grupos",
    "perguntas_distintas",
    "duplicatas_chave",
    "metricas_disponiveis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa significancia estatistica entre tipos de resposta dentro "
            "de cada arquivo_fonte, pareando os resultados por question_id."
        )
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Lista explicita dos CSVs que devem ser analisados. "
            "Substitui ARQUIVOS_ENTRADA; curingas nao sao aceitos."
        ),
    )
    input_group.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help=(
            "Arquivo texto com um caminho de CSV por linha. "
            "Substitui ARQUIVOS_ENTRADA."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta para os CSVs e o relatorio da analise.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metricas a analisar. Por padrao, usa todas as metricas conhecidas.",
    )
    parser.add_argument(
        "--pair-column",
        default="question_id",
        help="Coluna usada para parear os tipos. Padrao: question_id.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nivel de significancia. Padrao: 0.05.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=10,
        help="Numero minimo de pares validos para executar o teste. Padrao: 10.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=12,
        help=(
            "Casas decimais usadas ao arredondar diferencas pareadas antes "
            "do Wilcoxon. Padrao: 12."
        ),
    )
    parser.add_argument(
        "--correction-scope",
        choices=["global", "arquivo_fonte", "arquivo_metrica"],
        default="arquivo_fonte",
        help=(
            "Escopo da correcao de Holm. Padrao: arquivo_fonte, corrigindo "
            "todas as metricas e comparacoes feitas dentro de cada fonte."
        ),
    )
    return parser.parse_args()


def read_input_list(path: Path) -> list[Path]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Lista de entradas nao encontrada: {resolved}")

    paths: list[Path] = []
    for line in resolved.read_text(encoding="utf-8-sig").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            paths.append(Path(cleaned))
    return paths


def resolve_explicit_input_paths(input_paths: list[Path]) -> list[Path]:
    resolved_paths: list[Path] = []
    for path in input_paths:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = (
                DEFAULT_INPUT_DIR / candidate
                if len(candidate.parts) == 1
                else PROJECT_ROOT / candidate
            )
        if any(character in str(candidate) for character in "*?[]"):
            raise ValueError(
                f"Curingas nao sao permitidos nos arquivos de entrada: {path}"
            )
        resolved_paths.append(candidate.resolve())

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def collect_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.inputs is not None:
        requested_paths = list(args.inputs)
    elif args.input_list is not None:
        requested_paths = read_input_list(args.input_list)
    else:
        requested_paths = [Path(path) for path in ARQUIVOS_ENTRADA]

    if not requested_paths:
        raise ValueError(
            "ARQUIVOS_ENTRADA esta vazia. Preencha a lista no inicio do script "
            "ou informe --inputs/--input-list."
        )

    paths = resolve_explicit_input_paths(requested_paths)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Arquivos de entrada nao encontrados: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def require_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{source_name} nao possui as colunas obrigatorias: "
            + ", ".join(missing)
        )


def drop_repeated_header_rows(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if dataframe.empty:
        return dataframe, 0

    mask = (
        dataframe["arquivo_fonte"].astype(str).str.strip().eq("arquivo_fonte")
        & dataframe["tipo_resposta"].astype(str).str.strip().eq("tipo_resposta")
    )
    removed = int(mask.sum())
    return (dataframe.loc[~mask].copy() if removed else dataframe), removed


def load_inputs(
    paths: list[Path],
    pair_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframes: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    required = [*GROUP_COLUMNS, pair_column]

    for path in paths:
        dataframe = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )
        require_columns(dataframe, required, path.name)
        dataframe, removed_headers = drop_repeated_header_rows(dataframe)

        for column in required:
            dataframe[column] = dataframe[column].fillna("").astype(str).str.strip()
        dataframe = dataframe[
            (dataframe["arquivo_fonte"] != "")
            & (dataframe["tipo_resposta"] != "")
            & (dataframe[pair_column] != "")
        ].copy()
        available_metrics = [
            metric for metric in DEFAULT_METRIC_COLUMNS if metric in dataframe.columns
        ]
        duplicate_count = int(
            dataframe.duplicated([*GROUP_COLUMNS, pair_column], keep=False).sum()
        )
        coverage_rows.append(
            {
                "arquivo_entrada": str(path),
                "linhas_lidas": int(len(dataframe)),
                "linhas_cabecalho_removidas": removed_headers,
                "grupos": int(dataframe.groupby(GROUP_COLUMNS).ngroups),
                "perguntas_distintas": int(dataframe[pair_column].nunique()),
                "duplicatas_chave": duplicate_count,
                "metricas_disponiveis": len(available_metrics),
            }
        )
        dataframes.append(dataframe)

    combined = pd.concat(dataframes, ignore_index=True, sort=False)
    coverage = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)
    return combined, coverage


def select_metrics(
    dataframe: pd.DataFrame,
    requested_metrics: list[str] | None,
) -> list[str]:
    candidates = requested_metrics or DEFAULT_METRIC_COLUMNS
    missing = [metric for metric in candidates if metric not in dataframe.columns]
    if requested_metrics and missing:
        raise ValueError(
            "Metricas solicitadas ausentes nos CSVs: " + ", ".join(missing)
        )

    selected = [metric for metric in candidates if metric in dataframe.columns]
    if not selected:
        raise ValueError("Nenhuma metrica numerica conhecida foi encontrada.")

    usable = []
    for metric in selected:
        if pd.to_numeric(dataframe[metric], errors="coerce").notna().any():
            usable.append(metric)
    if not usable:
        raise ValueError("As metricas encontradas nao possuem valores numericos.")
    return usable


def build_question_metric_table(
    dataframe: pd.DataFrame,
    pair_column: str,
    metrics: list[str],
) -> pd.DataFrame:
    working = dataframe[[*GROUP_COLUMNS, pair_column, *metrics]].copy()
    for metric in metrics:
        working[metric] = pd.to_numeric(working[metric], errors="coerce")

    # Se a mesma pergunta aparecer em mais de um arquivo de entrada, as
    # repeticoes do judge sao agregadas antes da comparacao entre tipos.
    return (
        working.groupby(
            [*GROUP_COLUMNS, pair_column],
            sort=False,
            dropna=False,
            as_index=False,
        )[metrics]
        .mean()
    )


def build_descriptive_summary(
    question_metrics: pd.DataFrame,
    pair_column: str,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = question_metrics.groupby(GROUP_COLUMNS, sort=False, dropna=False)

    for (source, response_type), group in grouped:
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "arquivo_fonte": source,
                    "tipo_resposta": response_type,
                    "metrica": metric,
                    "n_perguntas": int(group[pair_column].nunique()),
                    "n_valores": int(len(values)),
                    "media": float(values.mean()),
                    "desvio_padrao": float(values.std(ddof=1))
                    if len(values) > 1
                    else math.nan,
                    "mediana": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "minimo": float(values.min()),
                    "maximo": float(values.max()),
                }
            )

    return pd.DataFrame(rows, columns=DESCRIPTIVE_COLUMNS)


def exact_two_sided_sign_test(positive: int, negative: int) -> float:
    total = positive + negative
    if total == 0:
        return 1.0

    smaller = min(positive, negative)
    log_probabilities = [
        (
            math.lgamma(total + 1)
            - math.lgamma(k + 1)
            - math.lgamma(total - k + 1)
            - total * math.log(2.0)
        )
        for k in range(smaller + 1)
    ]
    maximum_log = max(log_probabilities)
    lower_tail = math.exp(maximum_log) * sum(
        math.exp(log_probability - maximum_log)
        for log_probability in log_probabilities
    )
    return min(1.0, 2.0 * lower_tail)


def wilcoxon_signed_rank(differences: np.ndarray) -> dict[str, float | int]:
    finite = differences[np.isfinite(differences)]
    nonzero = finite[finite != 0]
    nonzero_count = int(len(nonzero))
    if nonzero_count == 0:
        return {
            "n_nonzero": 0,
            "statistic": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
        }

    absolute = np.abs(nonzero)
    ranks = pd.Series(absolute).rank(method="average").to_numpy(dtype=float)
    w_positive = float(ranks[nonzero > 0].sum())
    w_negative = float(ranks[nonzero < 0].sum())
    statistic = min(w_positive, w_negative)

    _, tie_counts = np.unique(absolute, return_counts=True)
    tie_correction = float(sum(count**3 - count for count in tie_counts))
    n = float(nonzero_count)
    expected = n * (n + 1.0) / 4.0
    variance = (
        n * (n + 1.0) * (2.0 * n + 1.0) - tie_correction / 2.0
    ) / 24.0

    if variance <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        centered = w_positive - expected
        continuity = 0.5 * np.sign(centered) if centered != 0 else 0.0
        z_score = float((centered - continuity) / math.sqrt(variance))
        p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0)))

    rank_total = w_positive + w_negative
    rank_biserial = (
        float((w_positive - w_negative) / rank_total)
        if rank_total
        else 0.0
    )
    return {
        "n_nonzero": nonzero_count,
        "statistic": statistic,
        "z": z_score,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
    }


def comparison_direction(
    delta: float,
    significant: bool,
    type_a: str,
    type_b: str,
) -> str:
    if not significant:
        return "sem_diferenca_significativa"
    if delta > 0:
        return f"{type_b}_maior"
    if delta < 0:
        return f"{type_a}_maior"
    return "sem_direcao"


def build_pairwise_comparisons(
    question_metrics: pd.DataFrame,
    pair_column: str,
    metrics: list[str],
    min_pairs: int,
    round_decimals: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source, source_df in question_metrics.groupby(
        "arquivo_fonte",
        sort=False,
        dropna=False,
    ):
        response_types = sorted(source_df["tipo_resposta"].dropna().unique())
        for type_a, type_b in combinations(response_types, 2):
            type_a_df = source_df[source_df["tipo_resposta"] == type_a]
            type_b_df = source_df[source_df["tipo_resposta"] == type_b]

            for metric in metrics:
                paired = type_a_df[[pair_column, metric]].merge(
                    type_b_df[[pair_column, metric]],
                    on=pair_column,
                    how="inner",
                    suffixes=("_a", "_b"),
                )
                values_a = pd.to_numeric(
                    paired[f"{metric}_a"],
                    errors="coerce",
                )
                values_b = pd.to_numeric(
                    paired[f"{metric}_b"],
                    errors="coerce",
                )
                valid = values_a.notna() & values_b.notna()
                values_a = values_a[valid].to_numpy(dtype=float)
                values_b = values_b[valid].to_numpy(dtype=float)
                differences = np.round(
                    values_b - values_a,
                    decimals=round_decimals,
                )
                pair_count = int(len(differences))

                base_row: dict[str, Any] = {
                    "arquivo_fonte": source,
                    "metrica": metric,
                    "tipo_a": type_a,
                    "tipo_b": type_b,
                    "n_pares": pair_count,
                    "n_diferencas_nao_zero": int(np.count_nonzero(differences)),
                    "media_tipo_a": float(np.mean(values_a))
                    if pair_count
                    else math.nan,
                    "media_tipo_b": float(np.mean(values_b))
                    if pair_count
                    else math.nan,
                    "mediana_tipo_a": float(np.median(values_a))
                    if pair_count
                    else math.nan,
                    "mediana_tipo_b": float(np.median(values_b))
                    if pair_count
                    else math.nan,
                    "delta_medio_b_menos_a": float(np.mean(differences))
                    if pair_count
                    else math.nan,
                    "delta_mediano_b_menos_a": float(np.median(differences))
                    if pair_count
                    else math.nan,
                    "vitorias_tipo_a": int(np.sum(differences < 0)),
                    "vitorias_tipo_b": int(np.sum(differences > 0)),
                    "empates": int(np.sum(differences == 0)),
                    "teste_principal": (
                        "wilcoxon_signed_rank_aproximacao_normal_"
                        "zeros_descartados_correcao_continuidade"
                    ),
                    "estatistica_wilcoxon": math.nan,
                    "z_wilcoxon": math.nan,
                    "p_valor": math.nan,
                    "p_valor_teste_sinais": math.nan,
                    "efeito_rank_biserial": math.nan,
                    "p_valor_ajustado_holm": math.nan,
                    "alpha": math.nan,
                    "significativo": False,
                    "direcao": "",
                    "status": "pares_insuficientes",
                }

                if pair_count >= min_pairs:
                    wilcoxon = wilcoxon_signed_rank(differences)
                    positive = int(np.sum(differences > 0))
                    negative = int(np.sum(differences < 0))
                    base_row.update(
                        {
                            "n_diferencas_nao_zero": wilcoxon["n_nonzero"],
                            "estatistica_wilcoxon": wilcoxon["statistic"],
                            "z_wilcoxon": wilcoxon["z"],
                            "p_valor": wilcoxon["p_value"],
                            "p_valor_teste_sinais": exact_two_sided_sign_test(
                                positive,
                                negative,
                            ),
                            "efeito_rank_biserial": wilcoxon["rank_biserial"],
                            "status": (
                                "sem_variacao"
                                if wilcoxon["n_nonzero"] == 0
                                else "ok"
                            ),
                        }
                    )

                rows.append(base_row)

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(math.nan, index=p_values.index, dtype=float)
    valid = pd.to_numeric(p_values, errors="coerce").dropna().sort_values()
    total = len(valid)
    running_max = 0.0

    for rank, (index, p_value) in enumerate(valid.items()):
        candidate = min(1.0, float(p_value) * (total - rank))
        running_max = max(running_max, candidate)
        adjusted.at[index] = running_max
    return adjusted


def apply_holm_correction(
    comparisons: pd.DataFrame,
    scope: str,
    alpha: float,
) -> pd.DataFrame:
    result = comparisons.copy()
    if result.empty:
        return result

    if scope == "global":
        result["p_valor_ajustado_holm"] = holm_adjust(result["p_valor"])
    else:
        group_columns = (
            ["arquivo_fonte"]
            if scope == "arquivo_fonte"
            else ["arquivo_fonte", "metrica"]
        )
        result["p_valor_ajustado_holm"] = (
            result.groupby(group_columns, dropna=False, group_keys=False)["p_valor"]
            .apply(holm_adjust)
            .reindex(result.index)
        )

    result["alpha"] = alpha
    result["significativo"] = (
        pd.to_numeric(result["p_valor_ajustado_holm"], errors="coerce") < alpha
    )
    result["direcao"] = result.apply(
        lambda row: comparison_direction(
            delta=row["delta_medio_b_menos_a"],
            significant=bool(row["significativo"]),
            type_a=str(row["tipo_a"]),
            type_b=str(row["tipo_b"]),
        ),
        axis=1,
    )
    return result[COMPARISON_COLUMNS]


def build_report(
    paths: list[Path],
    metrics: list[str],
    descriptive: pd.DataFrame,
    comparisons: pd.DataFrame,
    alpha: float,
    correction_scope: str,
    pair_column: str,
    round_decimals: int,
) -> str:
    valid_tests = comparisons[comparisons["p_valor"].notna()]
    significant = comparisons[comparisons["significativo"] == True]
    insufficient = comparisons[comparisons["status"] == "pares_insuficientes"]

    lines = [
        "# Analise de significancia",
        "",
        "## Metodologia",
        "",
        f"- Arquivos analisados: {len(paths)}",
        f"- Metricas analisadas: {len(metrics)}",
        f"- Pareamento: `{pair_column}`",
        "- Agrupamento: `arquivo_fonte` e `tipo_resposta`",
        "- Comparacao: todos os pares de tipos existentes em cada `arquivo_fonte`",
        (
            "- Teste principal: Wilcoxon signed-rank bilateral, aproximacao "
            "normal, zeros descartados e correcao de continuidade de 0.5"
        ),
        "- Teste complementar: teste exato bilateral dos sinais",
        f"- Diferencas pareadas arredondadas para {round_decimals} casas decimais",
        f"- Correcao de multiplas comparacoes: Holm ({correction_scope})",
        f"- Alpha: {alpha}",
        (
            "- Repeticoes da mesma pergunta, fonte e tipo em arquivos diferentes "
            "foram agregadas pela media antes dos testes."
        ),
        (
            "- O p-valor do Wilcoxon e assintotico; em amostras pequenas ou "
            "com muitos empates, consulte tambem o teste exato dos sinais."
        ),
        "",
        "## Resultados",
        "",
        f"- Grupos descritivos: {len(descriptive)}",
        f"- Testes executados: {len(valid_tests)}",
        f"- Comparacoes significativas apos Holm: {len(significant)}",
        f"- Comparacoes com pares insuficientes: {len(insufficient)}",
        "",
        "## Comparacoes significativas",
        "",
    ]

    if significant.empty:
        lines.append("_Nenhuma comparacao significativa apos a correcao de Holm._")
    else:
        display_columns = [
            "arquivo_fonte",
            "metrica",
            "tipo_a",
            "tipo_b",
            "n_pares",
            "delta_medio_b_menos_a",
            "p_valor_ajustado_holm",
            "efeito_rank_biserial",
            "direcao",
        ]
        try:
            lines.append(significant[display_columns].to_markdown(index=False))
        except Exception:
            lines.append(significant[display_columns].to_string(index=False))

    lines.extend(
        [
            "",
            "## Arquivos de saida",
            "",
            "- `cobertura_entradas.csv`",
            "- `resumo_descritivo.csv`",
            "- `comparacoes_significancia.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv_atomically(dataframe: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def save_outputs(
    output_dir: Path,
    coverage: pd.DataFrame,
    descriptive: pd.DataFrame,
    comparisons: pd.DataFrame,
    report: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomically(coverage, output_dir / "cobertura_entradas.csv")
    write_csv_atomically(descriptive, output_dir / "resumo_descritivo.csv")
    write_csv_atomically(
        comparisons,
        output_dir / "comparacoes_significancia.csv",
    )

    report_path = output_dir / "relatorio_significancia.md"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(report + "\n", encoding="utf-8")
    temporary_report.replace(report_path)


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha deve estar entre 0 e 1.")
    if args.min_pairs < 2:
        raise ValueError("--min-pairs deve ser pelo menos 2.")
    if args.round_decimals < 0:
        raise ValueError("--round-decimals deve ser zero ou maior.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        paths = collect_input_paths(args)
        combined, coverage = load_inputs(paths, args.pair_column)
        metrics = select_metrics(combined, args.metrics)
        question_metrics = build_question_metric_table(
            combined,
            args.pair_column,
            metrics,
        )
        descriptive = build_descriptive_summary(
            question_metrics,
            args.pair_column,
            metrics,
        )
        comparisons = build_pairwise_comparisons(
            question_metrics,
            args.pair_column,
            metrics,
            args.min_pairs,
            args.round_decimals,
        )
        comparisons = apply_holm_correction(
            comparisons,
            args.correction_scope,
            args.alpha,
        )
        report = build_report(
            paths=paths,
            metrics=metrics,
            descriptive=descriptive,
            comparisons=comparisons,
            alpha=args.alpha,
            correction_scope=args.correction_scope,
            pair_column=args.pair_column,
            round_decimals=args.round_decimals,
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (PROJECT_ROOT / args.output_dir).resolve()
        )
        save_outputs(
            output_dir,
            coverage,
            descriptive,
            comparisons,
            report,
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Erro: {exc}")
        return 1

    significant_count = int(comparisons["significativo"].sum())
    print(f"Arquivos analisados: {len(paths)}")
    print(f"Metricas analisadas: {len(metrics)}")
    print(f"Comparacoes geradas: {len(comparisons)}")
    print(f"Comparacoes significativas apos Holm: {significant_count}")
    print(f"Saidas salvas em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
