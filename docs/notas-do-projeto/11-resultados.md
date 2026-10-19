# 11. Experimentos e resultados

Este capítulo reúne os experimentos de treino feitos com os dados e o protocolo atuais (capítulos 3 e 9), o diagnóstico dos
marcadores aprendidos e o que ainda está planejado. Os resultados mudam a cada rodada de experimentos; a data de cada número
está indicada. Os oráculos, que não envolvem treino, estão no capítulo 10.

**Estado em 2026-10-17.** Nenhuma configuração treinada superou o Cellpose. A melhor, com o canal de fundo do Cellpose, chega a
0,808 na validação cruzada, contra 0,845 do Cellpose.

## 11.1 Os experimentos

Todos usam a MarkerUNet do capítulo 6, o ScribblePrompt no modo `positive` com $\tau = 0{,}5$ (capítulo 7), a otimização e a
validação cruzada do capítulo 9 e as mesmas 37 imagens e *folds*. O que muda entre eles:

| Experimento | Perda | Canal de fundo | Entrada do ScribblePrompt | Estrutura |
|---|---|---|---|---|
| `base_dice_tv` | Dice + TV (0,001) | nenhum | imagem inteira em 256² | antiga |
| `dice_size` | Dice + Size (0,1) | nenhum | imagem inteira em 256² | antiga |
| `base_negativo` | Dice + TV (0,001) | fundo do Cellpose, 3 px | imagem inteira em 256² | antiga |
| `base_dice` | Dice | nenhum | imagem inteira em 256² | atual |

"Estrutura antiga" quer dizer que o notebook também treinou o modelo final e avaliou o teste logo em seguida; os três estão em
`notebooks/experiments/legado/`. Na estrutura atual, o notebook do experimento faz só a validação cruzada, e o teste fica para o
notebook final (capítulo 9, seção 9.4).

Resultados em `docs/estudo/resultados/<experimento>/` (`resumo.json`, `folds.csv`, `config.json`). Os três experimentos antigos
foram registrados em 2026-10-07 e 2026-10-15. O `base_dice` rodou em 2026-10-17 e ainda não foi registrado nem revisado pelo
autor: os números dele são preliminares.

## 11.2 Resultados

**Validação cruzada (37 imagens, cada uma prevista pelo modelo do *fold* em que ficou de fora).** Cellpose nas mesmas imagens:
Dice 0,845, precisão 0,867, revocação 0,826, massa 0,955.

| Experimento | Dice (IC 95%) | Diferença pareada para o Cellpose (IC 95%) | Imagens melhores que o Cellpose | Precisão | Revocação | Massa | Fração de pixels com marcador ≥ 0,5 |
|---|---|---|---|---|---|---|---|
| `base_dice_tv` | 0,775 (0,760 a 0,789) | −0,070 (−0,081 a −0,060) | 0 de 37 | 0,727 | 0,838 | 1,17 | 9,0% |
| `dice_size` | 0,774 (0,757 a 0,790) | −0,071 (−0,084 a −0,059) | 0 de 37 | 0,753 | 0,807 | 1,09 | 5,8% |
| `base_negativo` | **0,808** (0,796 a 0,818) | −0,038 (−0,045 a −0,030) | 2 de 37 | 0,750 | 0,879 | 1,18 | 22,8% |
| `base_dice` (preliminar) | 0,785 (0,769 a 0,801) | −0,060 (−0,072 a −0,049) | 0 de 37 | 0,758 | 0,824 | 1,10 | 11,2% |

Em todos, o teste de Wilcoxon contra o Cellpose dá $p < 10^{-10}$. Intervalos por *bootstrap* (capítulo 9, seção 9.5).

**Comparações pareadas entre experimentos** (mesmas 37 imagens e *folds*; diferença de Dice, IC 95% por *bootstrap*, Wilcoxon;
calculado em 2026-10-17 a partir dos `folds.csv`):

| Comparação | Diferença | Imagens em que o primeiro é melhor | p |
|---|---|---|---|
| `base_negativo` menos `base_dice_tv` (efeito do canal de fundo) | +0,033 (+0,024 a +0,041) | 34 de 37 | $6{,}5 \times 10^{-9}$ |
| `base_dice` menos `base_dice_tv` (efeito de tirar o TV) | +0,011 (+0,003 a +0,018) | 29 de 37 | 0,002 |
| `dice_size` menos `base_dice_tv` (efeito de trocar TV por Size) | −0,000 (−0,009 a +0,008) | 21 de 37 | 0,96 |

A comparação com o `base_dice` mistura duas mudanças de código entre as execuções (commit `f08bc1d` no `base_dice_tv` e `2fc01fa`
no `base_dice`), embora perda, dados, *folds* e sementes sejam os mesmos. Hipótese: as mudanças de código entre esses commits não
afetam o treino; conferir antes de usar a comparação no texto.

**Teste (14 imagens).** Os três experimentos antigos avaliaram o teste. Os números estão aqui para registro; as decisões foram
tomadas pela validação cruzada (capítulo 9, seção 9.7):

| Experimento | Dice no teste | Cellpose | Diferença pareada (IC 95%) | Imagens melhores |
|---|---|---|---|---|
| `base_dice_tv` | 0,777 | 0,837 | −0,060 (−0,078 a −0,043) | 1 de 14 |
| `dice_size` | 0,789 | 0,837 | −0,048 (−0,058 a −0,037) | 1 de 14 |
| `base_negativo` | 0,803 | 0,837 | −0,034 (−0,042 a −0,025) | 1 de 14 |

## 11.3 O que os resultados dizem

1. **O pipeline ainda piora a própria entrada.** Em todas as configurações, a segmentação final fica abaixo da máscara do
   Cellpose que entra na MarkerUNet, em quase todas as imagens.
2. **A perda é de precisão, não de revocação.** A revocação do pipeline é igual ou maior que a do Cellpose, e a precisão é bem
   menor (0,73 a 0,76 contra 0,87). A massa de 1,1 a 1,2 mostra que o pipeline segmenta de 10% a 20% a mais que o GT. O problema
   é o excesso ([08-pendencias.md, PD-52](../estudo/08-pendencias.md#pd-52)).
3. **O canal de fundo do Cellpose é o maior ganho até agora** (+0,033, melhor em 34 de 37 imagens), mas muda o papel do
   marcador: ele vira um campo quase uniforme de cerca de 0,5 na imagem toda, e é o canal de fundo que delimita os núcleos. O
   marcador deixa de ter forma (capítulo 8, seção 8.6).
4. **O Size reduz o marcador sem mudar o Dice.** Com o Size, o marcador cobre 5,8% da imagem em vez de 9,0%, e a segmentação troca
   revocação por precisão (massa 1,09 em vez de 1,17), com o mesmo Dice.
5. **Tirar o TV ajuda um pouco** (+0,011, preliminar). O TV, mesmo com peso pequeno, atrapalha mais do que ajuda nesta
   configuração.
6. **As imagens em 20× são as piores.** Na validação, o Dice delas fica entre 0,66 e 0,75, contra 0,78 a 0,81 nas outras. Como
   em 256² o ScribblePrompt não perde nessas imagens com marcadores ideais (capítulo 10, seção 10.2), a perda está na
   MarkerUNet ([08-pendencias.md, PD-24](../estudo/08-pendencias.md#pd-24)).

## 11.4 Diagnóstico dos marcadores aprendidos

Para entender o excesso de segmentação, o `base_dice_tv` foi retreinado nos mesmos *folds* (Dice de validação 0,773, contra
0,775 registrado), e cada imagem foi prevista pelo modelo do *fold* em que ficou de fora. Sem retreinar, foram testadas
variações na saída do pipeline ([11-oraculos.md §8](../estudo/11-oraculos.md); notebook
`notebooks/exploration/diagnostico_marcadores.ipynb`, executado em 2026-10-15; resultados só nas saídas do notebook):

| Condição | Dice | Precisão | Revocação | Massa |
|---|---|---|---|---|
| marcador aprendido (o pipeline) | 0,773 | 0,725 | 0,839 | 1,18 |
| o mesmo, com limiar de saída 0,6 | 0,772 | 0,774 | 0,779 | 1,02 |
| canal positivo binarizado (1 onde for maior que 0) | 0,711 | 0,853 | 0,620 | 0,73 |
| marcador aprendido + fundo do Cellpose | **0,802** | 0,760 | 0,852 | 1,13 |
| binarizado + fundo do Cellpose | 0,801 | 0,807 | 0,801 | 1,00 |
| miolo do GT (oráculo, mesmas imagens) | 0,754 | 0,963 | 0,622 | 0,65 |

E onde estão as coisas:

- **o marcador cai nos núcleos:** 91,5% da massa do canal positivo fica dentro do GT, 4,3% a até 3 px dele e 4,2% mais longe; 87%
  dos componentes do GT recebem marcador;
- **os falsos positivos ficam junto às bordas:** 61,5% a até 3 px do GT, 22,8% de 3 a 8 px e 15,7% mais longe;
- **a saída funde núcleos:** 28% dos componentes da saída cobrem dois ou mais componentes do GT; no Cellpose, 7%.

Leituras:

1. **Os marcadores estão no lugar certo.** O excesso vem do inchaço das bordas e da fusão de núcleos vizinhos, não de marcadores
   no fundo.
2. **O valor suave do marcador faz a rede expandir.** Binarizado, o mesmo marcador segmenta de menos, como o miolo do oráculo
   (massa 0,73). Hipótese: o canal positivo tem amplitude baixa, e o ScribblePrompt expande muito esses "toques fracos", que estão
   fora da distribuição de rabiscos com que ele foi treinado. A amplitude não foi medida em números; a métrica de contraste do
   marcador (capítulo 9, seção 9.6) foi criada depois para isso. No `base_dice`, a média do marcador é 0,48 dentro dos núcleos e
   0,28 fora (preliminar).
3. **O limiar de saída não resolve.** Ele troca precisão por revocação e mantém o Dice: a borda da saída é borrada.
4. **O canal de fundo do Cellpose é o único ganho sem retreinar** (+0,03), mas não alcança o Cellpose. Treinar com ele (o
   `base_negativo`) dá praticamente o mesmo número (0,808 contra 0,802).

## 11.5 Próximos experimentos

Em ordem, conforme o plano combinado ([08-pendencias.md, plano de ataque, bloco B](../estudo/08-pendencias.md#plano-de-ataque-combinado-com-o-autor-em-2026-10-17)):

1. **`negativo_sem_tv`:** o `base_negativo` sem o TV. Responde se é o TV que achata o marcador quando há canal de fundo.
2. **`recortes`:** treino por recortes de 256 px, ScribblePrompt em 128², avaliação em blocos (capítulo 4, seção 4.3), sem canal
   de fundo. É o modo que o oráculo de resolução apontou como equilibrado (capítulo 10, seção 10.3). Comparado com o `base_dice`.
3. **Recortes com o canal de fundo.**
4. Conforme os resultados: um marcador quase binário, ou a volta do Size e do Dmap.

Depois: as ablações da padronização do ImageNet com o congelamento do codificador e da probabilidade do Cellpose como quarto
canal (capítulo 6), e os experimentos `dice_dmap`, `tese` e `rmse`, se ainda fizerem sentido.

## 11.6 Situação em relação à pergunta do trabalho

Até aqui, a resposta para "o pipeline funciona sem o FMBS, com o ScribblePrompt?" é: **ainda não, nesta configuração**. Os
resultados e os oráculos apontam onde está o obstáculo:

- a camada de segmentação expande pouco os marcadores pequenos na resolução usada, e a informação de borda vem do canal de
  fundo, não do marcador (capítulo 10);
- os marcadores aprendidos estão no lugar certo, mas com valores suaves que a camada expande demais (seção 11.4);
- com o canal de fundo, o marcador perde a forma, e o resultado fica preso ao Cellpose.

Os oráculos mostram que há espaço: com o fundo do Cellpose e a entrada em 1024², ou em blocos, marcadores pequenos ideais passam
do Cellpose (capítulo 10, seção 10.3). Se os próximos experimentos não fecharem a diferença, a contribuição do trabalho passa a
ser a explicação de por que essa troca de camada não funciona (capítulo 2, seção 2.6).

## Para o artigo

- **Resultados:** a tabela de validação da seção 11.2, com o Cellpose na mesma tabela; as comparações pareadas entre
  experimentos; o teste do experimento escolhido, quando o notebook final rodar.
- **Resultados ou discussão:** o diagnóstico da seção 11.4 (marcadores no lugar certo, excesso por inchaço e fusão, efeito do
  valor suave).
- **Discussão:** as leituras das seções 11.3 e 11.6.
- **Limitações:** o teste já foi visto (capítulo 9, seção 9.7); o diagnóstico e o oráculo de resolução não têm os resultados
  salvos em arquivo ([08-pendencias.md, PD-47](../estudo/08-pendencias.md#pd-47)).

## Referências

Sem referências externas; os métodos estão nos capítulos 6 a 9.
