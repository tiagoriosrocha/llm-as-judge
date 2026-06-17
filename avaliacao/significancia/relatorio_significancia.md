# Analise de significancia

## Metodologia

- Arquivos analisados: 1
- Metricas analisadas: 19
- Pareamento: `question_id`
- Agrupamento: `arquivo_fonte` e `tipo_resposta`
- Comparacao: todos os pares de tipos existentes em cada `arquivo_fonte`
- Teste principal: Wilcoxon signed-rank bilateral, aproximacao normal, zeros descartados e correcao de continuidade de 0.5
- Teste complementar: teste exato bilateral dos sinais
- Diferencas pareadas arredondadas para 12 casas decimais
- Correcao de multiplas comparacoes: Holm (arquivo_fonte)
- Alpha: 0.05
- Repeticoes da mesma pergunta, fonte e tipo em arquivos diferentes foram agregadas pela media antes dos testes.
- O p-valor do Wilcoxon e assintotico; em amostras pequenas ou com muitos empates, consulte tambem o teste exato dos sinais.

## Resultados

- Grupos descritivos: 76
- Testes executados: 38
- Comparacoes significativas apos Holm: 0
- Comparacoes com pares insuficientes: 0

## Comparacoes significativas

_Nenhuma comparacao significativa apos a correcao de Holm._

## Arquivos de saida

- `cobertura_entradas.csv`
- `resumo_descritivo.csv`
- `comparacoes_significancia.csv`

