# 5. Segmentação inicial: Cellpose-SAM

O primeiro bloco do método (capítulo 4) é um segmentador automático de células, o Cellpose-SAM. Ele faz o papel que, na
tese, cabia às indicações do usuário e à rede de segmentação interativa: dizer, de forma aproximada, onde estão os objetos
(capítulo 2, seção 2.4). Este capítulo explica o que o Cellpose faz, como ele é usado, que parâmetros foram escolhidos e
por quê, e o que a saída dele entrega ao resto do pipeline.

## 5.1 O que é o Cellpose

O Cellpose é um método generalista de segmentação de células por instância: foi treinado com imagens muito variadas para
funcionar em tipos de imagem novos sem ser treinado de novo (Stringer et al., 2021). Uma rede neural não consegue prever
diretamente os rótulos de instância, porque os números dados a cada célula são arbitrários. Por isso, o Cellpose treina a
rede para prever uma representação intermediária:

- **fluxos:** para cada pixel, um vetor horizontal e um vertical que apontam para o centro da célula a que ele pertence. Os
  fluxos de treino vêm de uma difusão simulada a partir do centro de cada célula anotada;
- **probabilidade de célula:** para cada pixel, a chance de ele estar dentro de alguma célula.

Na inferência, os pixels com probabilidade alta "seguem os fluxos" até um ponto de convergência, e os pixels que chegam ao
mesmo ponto formam uma instância (Stringer et al., 2021, figura 1).

O **Cellpose-SAM** é a versão 4 do pacote, cujo único modelo pré-treinado se chama `cpsam`. Ele troca a rede do Cellpose
pelo codificador do *Segment Anything Model* (SAM; Kirillov et al., 2023), um transformador ViT-L pré-treinado, e mantém a
representação por fluxos e o "seguir os fluxos" do Cellpose. A entrada do codificador foi reduzida de 1024 × 1024 para
256 × 256 pixels, com blocos (*patches*) de 8 × 8 em vez de 16 × 16 (Pachitariu et al., 2025, figura 1 e texto da seção de
resultados). Três pontos do treino importam para este trabalho (mesma fonte, seção de métodos):

- o modelo foi treinado com 18 conjuntos de dados públicos, 22.826 imagens ao todo, incluindo conjuntos de núcleos em H&E;
- as imagens de treino foram redimensionadas em torno de um diâmetro de célula de 30 pixels, com um fator aleatório entre 0,25
  e 4, o que torna o modelo robusto ao tamanho das células;
- **o MoNuSeg é um dos 18 conjuntos**, descrito com 37 imagens de treino e 14 de teste, e o artigo avalia o modelo nas 14
  imagens de teste. A leitura natural é que as 37 imagens de treino entraram no treino do Cellpose-SAM; o artigo não lista as
  imagens usadas (seção 5.4 trata da consequência).

## 5.2 Como ele é usado

O Cellpose roda uma vez, sem nenhum ajuste de pesos, sobre as 51 imagens, no notebook de preparação dos dados
(capítulo 3, seção 3.6), com GPU. O passo [CellposeStep](../../src/pipeline/steps/preprocessing/cellpose_step.py) chama o
modelo `cpsam` do Cellpose 4.1.1 e grava duas saídas:

- **`segmentation`:** a máscara de instâncias (0 = fundo, 1 a K = núcleos);
- **`cellpose_prob`:** a probabilidade de célula de cada pixel, em $[0,1]$, guardada em float16. O mapa que o Cellpose
  devolve é um logit (a rede é treinada com entropia cruzada binária sobre logits), e o passo grava a sigmoide dele.

Dessas saídas, o resto do pipeline usa:

- a **máscara binária** (instância maior que zero), como quarto canal da entrada da MarkerUNet (capítulo 6);
- a mesma máscara, como **linha de base** de todas as comparações (capítulo 1, seção 1.6);
- opcionalmente, a **probabilidade** no lugar da máscara binária, como quarto canal (capítulo 6);
- opcionalmente, o **fundo** da máscara, afastado dos núcleos por uma faixa de alguns pixels, como canal de fundo do
  ScribblePrompt (capítulo 7).

O passo exige GPU e recusa um nome de modelo desconhecido. Sem essa checagem, o Cellpose 4 trocaria silenciosamente o
modelo pedido pelo `cpsam`, com um aviso pouco claro, e a entrada do pipeline mudaria sem ninguém perceber
([cellpose_step.py](../../src/pipeline/steps/preprocessing/cellpose_step.py), `_validate_model_name`).

## 5.3 Parâmetros

| Parâmetro | Valor | O que faz |
|---|---|---|
| `pretrained_model` | `cpsam` | o modelo do Cellpose-SAM |
| `diameter` | `None` | sem reescala da imagem antes da rede |
| `flow_threshold` | 0,4 | erro máximo de fluxo aceito por máscara; valores menores descartam mais máscaras |
| `cellprob_threshold` | 0,0 | limiar do logit de probabilidade, equivalente a 0,5 em probabilidade |
| `min_size` | 15 | área mínima, em pixels, de uma instância |

São os valores padrão da biblioteca. No Cellpose 4.1.1, o diâmetro funciona assim: a imagem é reescalada por
`30 / diameter` antes da rede, e tanto `None` quanto 30 significam "não reescalar" (conferido no código da biblioteca, versão
4.1.1).

**Por que esses valores.** Quatro configurações foram comparadas nas 37 imagens de treino, sempre contra o GT atual
([cellpose_parametros_treino.csv](../estudo/resultados/cellpose_parametros_treino.csv), execução de 2026-10-02 num Colab com
GPU T4, commit `64a624f`):

| Configuração | Dice médio | Precisão | Revocação | Instâncias / núcleos do GT | Tempo por imagem |
|---|---|---|---|---|---|
| padrão: sem reescala, `flow_threshold` 0,4, `min_size` 15 | **0,845** | 0,867 | 0,826 | 0,92 | 10,7 s |
| sem reescala, `flow_threshold` 0,2, `min_size` 4 | 0,814 | 0,879 | 0,761 | 0,80 | 10,4 s |
| diâmetro 22, `flow_threshold` 0,2, `min_size` 4 | 0,815 | 0,872 | 0,768 | 0,80 | 20,6 s |
| diâmetro por imagem (12 em 20×, 22 nas demais), `flow_threshold` 0,2, `min_size` 4 | 0,816 | 0,871 | 0,771 | 0,81 | 23,7 s |

Duas leituras:

- **Os limiares importam, o diâmetro quase não.** A configuração padrão é melhor que as outras três em todas as 37 imagens.
  O ganho vem da revocação: com `flow_threshold` 0,2, o Cellpose descarta núcleos que existem. Reescalar pelo diâmetro
  medido dos núcleos muda o Dice em 0,001 a 0,002 e dobra o tempo. Hipótese: o diâmetro importa pouco porque o modelo foi
  treinado com células de tamanhos muito variados (seção 5.1).
- **O Cellpose erra mais por falta do que por excesso.** Mesmo na configuração padrão, a revocação (0,826) fica abaixo da
  precisão (0,867), e ele encontra 92% das instâncias do GT no treino. Hipótese: há espaço para a rede de marcadores
  recuperar núcleos que o Cellpose deixou de fora.

## 5.4 A linha de base

Com esses parâmetros, a máscara binária do Cellpose tem Dice médio de **0,845** no treino (desvio 0,030, pior imagem 0,768) e
**0,837** no teste (desvio 0,033, pior imagem 0,755), medidos imagem por imagem contra o GT binário (capítulo 3, seção 3.4).
Nas três imagens em 20×, o Dice do Cellpose fica um pouco abaixo da média, cerca de 0,01 a menos
([08-pendencias.md, PD-24](../estudo/08-pendencias.md#pd-24)).

Esse é o número que o pipeline precisa superar (capítulo 2, seção 2.5). Toda tabela de resultados traz o Cellpose na mesma
linha, imagem por imagem, e o teste estatístico é pareado (capítulo 9).

**O Cellpose já viu as imagens de treino.** Como o MoNuSeg entrou no treino do Cellpose-SAM (seção 5.1), a máscara do Cellpose
nas 37 imagens de treino vem de imagens que o modelo viu, e nas 14 de teste, não. Isso tem duas consequências:

- na validação cruzada, a linha de base de 0,845 pode estar otimista em relação ao que o Cellpose faria em imagens novas;
- a MarkerUNet treina com uma entrada (a máscara do Cellpose) de qualidade possivelmente maior que a que recebe no teste.

A diferença medida entre treino e teste é pequena (0,845 contra 0,837), mas os dois conjuntos também diferem em órgãos e
escala (capítulo 3), e não há como separar os efeitos. Hipótese a medir: o tamanho desse efeito. Uma forma seria comparar com
um modelo do Cellpose que não viu o MoNuSeg.

## Para o artigo

- **Métodos:** seção 5.1 em um parágrafo (o Cellpose, os fluxos e a versão SAM), seção 5.2 (usado congelado, uma vez) e a
  tabela de parâmetros da seção 5.3.
- **Resultados ou material suplementar:** a comparação de parâmetros. Mostra que os padrões da biblioteca já são a melhor
  escolha entre as testadas, e com isso a linha de base não foi enfraquecida.
- **Resultados:** a linha de base da seção 5.4, em toda tabela.
- **Limitações:** o Cellpose-SAM foi treinado com as imagens de treino do MoNuSeg (seção 5.4).

## Referências

- Kirillov, A. et al. (2023). Segment anything. *ICCV*. arXiv:2304.02643. *Conferida* (título e autores).
- Pachitariu, M.; Rariden, M.; Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation.
  *bioRxiv*, doi:10.1101/2025.04.28.651001. *Conferida* (resumo, figura 1, métodos e descrição dos conjuntos de dados).
- Stringer, C.; Wang, T.; Michaelos, M.; Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation.
  *Nature Methods*, 18, 100-106. *Conferida* a versão de pré-publicação (bioRxiv, doi:10.1101/2020.02.02.931238, em
  `docs/papers/cellpose.pdf`); autores, volume e páginas da versão publicada *a conferir*.
