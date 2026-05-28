# Avaliacao de QA com DeepEval

Este projeto consolida execucoes de QA e avalia cada resposta com metricas do DeepEval, GEval e metricas classicas de QA.

## Arquivos principais

- `gerar_csv_deepeval_consolidado.py`: consolida os CSVs brutos em um unico arquivo.
- `avaliar_qa_deepeval.py`: avalia o CSV consolidado linha a linha.
- `config.py`: carrega configuracoes do ambiente Azure/OpenAI.
- `llm_client.py`: cliente de baixo nivel para Azure OpenAI com retry.
- `deepeval_azure_model.py`: wrapper customizado do DeepEval para usar o judge Azure do projeto.
- `pastas_contexto.csv`: controla quais pastas de `resultados/` entram na consolidacao e se recebem contexto.

## Preparacao do ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configurando o judge Azure para o DeepEval

O `avaliar_qa_deepeval.py` foi adaptado para usar um modelo customizado do DeepEval (`AzureDeepEvalModel`) que encapsula o cliente `AzureOpenAI` do projeto. Em vez de depender da configuracao padrao de providers do `deepeval`, o script le as variaveis abaixo de `.env.local` e/ou `.env` atraves de `config.py`.

Exemplo de `.env.local`:

```env
LLM_API_KEY=<sua-chave>
LLM_ENDPOINT=https://apit.petrobras.com.br/ia/openai/v1/openai-azure/openai
LLM_API_VERSION=2025-01-01-preview
LLM_MODEL=gpt-5-4-petrobras
LLM_MAX_TOKENS=1050000

TEMPERATURE=0.3
MAX_RETRIES=3
RETRY_DELAY_SECONDS=2
TIMEOUT_SECONDS=30

PROMPT_VERSION=1.0
LOG_LEVEL=DEBUG
MAX_QUESTIONS=0
```

O script procura primeiro `.env.local` e depois `.env`.

Variaveis obrigatorias:

- `LLM_API_KEY`
- `LLM_ENDPOINT`
- `LLM_MODEL`

Variaveis opcionais:

- `LLM_API_VERSION`
- `LLM_MAX_TOKENS`
- `TEMPERATURE`
- `MAX_RETRIES`
- `RETRY_DELAY_SECONDS`
- `TIMEOUT_SECONDS`
- `PROMPT_VERSION`
- `LOG_LEVEL`
- `MAX_QUESTIONS`

Observacao:

- O judge do DeepEval e criado em codigo e passado explicitamente para cada metrica via `model=AzureDeepEvalModel(...)`, seguindo a interface oficial de custom LLMs do DeepEval.

## Gerando o CSV consolidado

```bash
python3 gerar_csv_deepeval_consolidado.py
```

## Executando a avaliacao

Exemplo com os caminhos em `output/`:

```bash
python3 avaliar_qa_deepeval.py \
  --input output/todas_execucoes_deepeval.csv \
  --output output/todas_execucoes_deepeval_avaliadas.csv \
  --summary output/resumo_metricas_por_execucao.csv
```

O script tambem aceita o formato abaixo e tenta localizar o CSV de entrada automaticamente em `output/` se necessario:

```bash
python3 avaliar_qa_deepeval.py \
  --input todas_execucoes_deepeval.csv \
  --output todas_execucoes_deepeval_avaliadas.csv \
  --summary resumo_metricas_por_execucao.csv
```

## Opcoes uteis

- `--limit 20`: processa apenas as 20 primeiras linhas.
- `--save-every 10`: salva progresso incremental a cada 10 linhas.
- `--judge-model gpt-5-4-petrobras`: sobrescreve o `LLM_MODEL` do `.env` so para esta execucao.
- `--log-level DEBUG`: sobrescreve o `LOG_LEVEL` do `.env`.

## Saidas geradas

- CSV avaliado com metricas por linha.
- CSV agregado por `arquivo_fonte`, com media, mediana e desvio padrao das metricas numericas.

## Observacoes

- As metricas do DeepEval e do GEval fazem chamadas ao modelo de avaliacao configurado, entao a execucao pode levar tempo e consumir tokens.
- A integracao com Azure e feita por `openai.AzureOpenAI` no `llm_client.py`, com retry exponencial para timeout/rate limit.
- Quando `context` estiver vazio, o script nao quebra: metricas que dependem de contexto sao marcadas como `Skipped: context vazio.`.
- Se alguma metrica falhar em uma linha, a execucao continua e o erro fica registrado em `eval_error`.
