# Notas do Projeto

Estas notas explicam o projeto `cell-fuzzy-seg` do começo ao fim: o domínio, o problema que se quer resolver, os dados, o
método, as redes, as perdas, o treinamento, a avaliação e os resultados. Elas são escritas para quem chega sem contexto e
seguem a ordem em que o assunto seria apresentado num artigo. A intenção é que cada capítulo sirva de base para uma seção
do texto final.

Os registros de trabalho continuam em `docs/estudo/`: pendências (`08-pendencias.md`), correções e experimentos
(`09-correcoes-pontuais.md`) e oráculos (`11-oraculos.md`). As notas citam esses registros quando precisam de um detalhe,
mas não repetem a história de como se chegou a cada coisa.

## Como ler

Cada afirmação traz a sua fonte no próprio texto:

- **código:** link para o arquivo e a linha;
- **medida:** o que foi medido, sobre quais arquivos e em que data;
- **literatura:** autor e ano entre parênteses, com a referência completa no fim do capítulo. Cada referência está marcada
  como *conferida* (lida no original) ou *a conferir* (ainda não lida; não citar no artigo antes de conferir);
- **sem fonte:** a marca "[referência a reunir]" indica uma afirmação que precisa de uma referência que ainda não temos.

A lista central do que já foi lido e do que falta levantar está em [referencias.md](referencias.md).

O que não foi verificado começa com **Hipótese:**. Nada escrito sem essa palavra deve ser lido como suposição.

Cada capítulo termina com duas seções fixas: **Para o artigo**, que diz o que dele entra no texto e em qual seção, e
**Referências**.

## Sumário

| Capítulo | Arquivo | Situação |
|---|---|---|
| 1. Domínio: núcleos em imagens histológicas | [01-dominio.md](01-dominio.md) | rascunho para revisão |
| 2. O problema e a pergunta do trabalho | [02-problema.md](02-problema.md) | rascunho para revisão |
| 3. Dados | [03-dados.md](03-dados.md) | rascunho para revisão |
| 4. Visão geral do método | [04-metodo.md](04-metodo.md) | rascunho para revisão |
| 5. Segmentação inicial: Cellpose-SAM | [05-cellpose.md](05-cellpose.md) | rascunho para revisão |
| 6. Rede de marcadores: MarkerUNet | [06-markerunet.md](06-markerunet.md) | rascunho para revisão |
| 7. Camada de segmentação: ScribblePrompt | [07-scribbleprompt.md](07-scribbleprompt.md) | rascunho para revisão |
| 8. Funções de perda | [08-perdas.md](08-perdas.md) | rascunho para revisão |
| 9. Treinamento e protocolo de avaliação | [09-treinamento-e-avaliacao.md](09-treinamento-e-avaliacao.md) | rascunho para revisão |
| 10. Oráculos | [10-oraculos.md](10-oraculos.md) | rascunho para revisão |
| 11. Experimentos e resultados | [11-resultados.md](11-resultados.md) | rascunho para revisão |
| 12. Decisões de projeto | [12-decisoes.md](12-decisoes.md) | rascunho para revisão |
| 13. Limitações e trabalhos futuros | [13-limitacoes-e-futuro.md](13-limitacoes-e-futuro.md) | rascunho para revisão |
| Apêndice A. Implementação | [a-implementacao.md](a-implementacao.md) | rascunho para revisão |
| Referências (lista central) | [referencias.md](referencias.md) | em uso |
