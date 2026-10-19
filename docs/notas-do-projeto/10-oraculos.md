# 10. Oráculos

Um oráculo é um experimento sem rede treinada: a MarkerUNet é substituída por marcadores conhecidos, tirados do ground truth
ou do Cellpose, e só a camada de segmentação roda. Ele responde à pergunta "se a rede de marcadores acertasse um marcador deste
tipo, até onde o pipeline chegaria?". Serve de teto para o pipeline e separa o que é limite da camada de segmentação do que é
limite do treino.

Os oráculos são uma das contribuições do trabalho (capítulo 2, seção 2.6): eles mostram que a resposta à pergunta do trabalho
depende tanto da camada de segmentação quanto da rede de marcadores. O registro completo, com todas as tabelas, está em
[11-oraculos.md](../estudo/11-oraculos.md). Este capítulo resume o método e os resultados que importam para o texto.

## 10.1 Método

**Dados.** As 37 imagens de treino, com o GT e a segmentação do Cellpose atuais. O teste não é usado, porque os oráculos
orientam escolhas de configuração (capítulo 9). Linha de base: o Cellpose nas mesmas imagens, com Dice 0,845, precisão 0,867,
revocação 0,826 e razão de massa 0,955.

**Entrada da camada.** O mesmo código do pipeline (capítulo 7), com os canais positivo e negativo montados à mão, binários (0 ou
1), numa única interação: caixa e predição anterior em zero.

**Marcadores positivos.** O mapa de distância por núcleo (capítulo 8, seção 8.3) vale 0 no ponto mais interno de cada núcleo e
1 na borda.

| Nome | Definição | Área média da imagem |
|---|---|---|
| GT inteiro | o ground truth | 24,4% |
| miolo | pixels de núcleo com mapa de distância até 0,5 | 8,4% (cerca de 34% do GT) |
| centros | disco de raio 4 px em volta do ponto mais interno de cada núcleo, cortado pelo GT | 3,8% |
| máscara do Cellpose | a máscara binária do Cellpose | 23,2% |
| miolo do Cellpose | o miolo de cada instância do Cellpose; não usa o GT | 8,0% |

O miolo e os centros representam o tipo de marcador que a tese propõe: pequeno e longe da borda.

**Canais negativos.**

| Nome | Definição | Área média da imagem |
|---|---|---|
| nenhum | canal negativo zerado, como no modo `positive` do pipeline | 0% |
| complemento | tudo o que não é positivo, como nos modos `sharpened` e `dense_soft` | 76% a 96% |
| fundo do Cellpose, 3 px | pixels a mais de 3 px de qualquer núcleo do Cellpose (capítulo 7, seção 7.4) | 64,9% |
| fundo do Cellpose, 8 px | o mesmo, a mais de 8 px | 46,4% |
| fundo do Cellpose esparso | discos de raio 3 px numa grade de 25 px, dentro do fundo a 8 px; imita cliques negativos | 2,1% |
| fundo do GT, 3 px | o negativo ideal, usado como teto | 63,0% |

**Medidas.** Dice, precisão, revocação e razão de massa por imagem (capítulo 1, seção 1.6), resumidos pela média e pelo número
de imagens, entre as 37, em que a configuração supera o Cellpose.

**Reprodução.** Notebook `notebooks/exploration/oraculo_negativo.ipynb`; resultados em
`docs/estudo/resultados/oraculo_negativo/` (CSV por imagem, resumo, configuração com commit e hash do checkpoint, figuras). A
execução salva é a do Colab com GPU T4; uma execução independente em CPU deu os mesmos números, com diferença máxima de
$2 \times 10^{-5}$ no Dice.

## 10.2 Resultado com a imagem inteira em 256²

Dice médio no treino; entre parênteses, as imagens (de 37) que superam o Cellpose. Para cada célula foi escolhido o melhor
limiar de saída (de 0,5 a 0,9) e o melhor recorte opcional da saída pela máscara dilatada do Cellpose, nas mesmas imagens; os
valores são, por isso, levemente otimistas ([11-oraculos.md §4.1](../estudo/11-oraculos.md)).

| Positivo | nenhum | fundo Cellpose 3 px | 8 px | esparso | complemento | fundo GT 3 px (teto) |
|---|---|---|---|---|---|---|
| GT inteiro | 0,915 (37) | 0,878 (34) | 0,876 (35) | 0,870 (29) | **0,944** (37) | 0,935 (37) |
| miolo | 0,754 (2) | **0,820** (2) | 0,791 (0) | 0,768 (0) | 0,561 (0) | 0,870 (29) |
| centros | 0,542 (0) | **0,786** (0) | 0,716 (0) | 0,690 (0) | 0,306 (0) | 0,830 (18) |
| máscara do Cellpose | 0,826 (0) | 0,832 (6) | 0,824 (2) | 0,822 (0) | **0,837** (5) | 0,879 (35) |
| miolo do Cellpose | 0,683 (0) | **0,797** (0) | 0,762 (0) | 0,741 (0) | 0,514 (0) | 0,838 (17) |

Precisão, revocação e massa de algumas combinações, no limiar 0,5 e sem recorte:

| Positivo e negativo | Dice | Precisão | Revocação | Massa |
|---|---|---|---|---|
| miolo, nenhum | 0,754 | 0,963 | 0,622 | 0,65 |
| miolo, fundo Cellpose 3 px | 0,820 | 0,858 | 0,792 | 0,93 |
| miolo, fundo Cellpose esparso | 0,592 | 0,555 | 0,767 | 1,95 |
| miolo, fundo GT 3 px | 0,870 | 0,899 | 0,848 | 0,95 |
| GT inteiro, nenhum | 0,839 | 0,725 | 0,997 | 1,38 |
| Cellpose sozinho | 0,845 | 0,867 | 0,826 | 0,96 |

O que esses números dizem:

1. **Em 256², nenhum marcador pequeno alcança o Cellpose, nem perfeito.** O melhor é o miolo com o fundo do Cellpose, 0,820, que
   supera o Cellpose em 2 de 37 imagens.
2. **Sem negativo, um marcador pequeno bem colocado segmenta de menos.** O miolo dá precisão 0,963 e massa 0,65: o ScribblePrompt
   expande pouco e deixa a borda dos núcleos de fora.
3. **O complemento como negativo é o pior para marcadores pequenos.** Dizer à rede que todo pixel não marcado é fundo derruba o
   miolo para 0,561 e os centros para 0,306. É a confirmação experimental do problema descrito no capítulo 7, seção 7.3.
4. **O fundo do Cellpose ajuda muito os marcadores pequenos, mas prende o resultado ao Cellpose.** O miolo sobe de 0,754 para
   0,820 e os centros de 0,542 para 0,786. Nenhuma configuração que só usa informação do Cellpose passa dele em 256² (a melhor
   dá 0,837), e com o GT inteiro o fundo do Cellpose piora o resultado (0,915 para 0,878), porque corta núcleos que o Cellpose
   não achou.
5. **A qualidade do negativo vale mais que a do positivo.** Com o fundo do GT, o miolo passa do Cellpose (0,870, em 29 de 37
   imagens). O negativo carrega a informação de borda que um marcador pequeno não tem.
6. **Negativos esparsos pioram.** Cliques negativos espalhados fazem a rede segmentar o dobro do GT (massa 1,95). Hipótese: é uma
   entrada fora da distribuição de treino do ScribblePrompt.

**Resolução e imagens em 20×.** Em 128², todas as configurações ficam abaixo de 256²; o miolo com o fundo do Cellpose cai de
0,820 para 0,761. As três imagens em 20× perdem muito em 128² (miolo sem negativo: 0,542, contra 0,677 nas outras), mas não em
256² (0,788 contra 0,751). O problema das imagens em 20× é de resolução, não da camada em si.

## 10.3 Resultado com outras resoluções e com blocos

O ScribblePrompt expande pouco em 256². A segunda pergunta foi se outra forma de apresentar a imagem muda isso. Foram comparados
seis modos ([11-oraculos.md §7](../estudo/11-oraculos.md); notebook `notebooks/exploration/oraculo_resolucao.ipynb`; resultados
só nas saídas do notebook):

| Modo | O que entra na rede | Núcleo mediano na entrada (40×) |
|---|---|---|
| imagem 256² | a imagem inteira reduzida para 256² | cerca de 5 px |
| imagem 512² | a imagem inteira reduzida para 512² | cerca de 11 px |
| imagem 1024² | a imagem inteira, levemente ampliada | cerca de 22 px |
| blocos 128 em 128² | blocos de 128 px da imagem, sem redução | cerca de 21 px |
| blocos 256 em 256² | blocos de 256 px, sem redução | cerca de 21 px |
| blocos 256 em 128² | blocos de 256 px reduzidos para 128² | cerca de 11 px |

Os blocos se sobrepõem em 1/4 do lado e são colados por média ponderada com janela de Hann, como no modo por recortes do
pipeline (capítulo 4, seção 4.3).

Dice médio no limiar 0,5 (entre parênteses, imagens que superam o Cellpose no melhor limiar) e, abaixo, a massa:

| Marcador e negativo | 256² | 512² | 1024² | blocos 128 em 128² | blocos 256 em 256² | blocos 256 em 128² |
|---|---|---|---|---|---|---|
| miolo, nenhum | 0,754 (2) | 0,794 (7) | 0,691 (5) | 0,586 (0) | 0,580 (0) | **0,808** (14) |
| miolo, fundo Cellpose | 0,820 (2) | 0,823 (3) | **0,852** (27) | 0,745 (0) | 0,737 (0) | 0,836 (9) |
| centros, nenhum | 0,542 (0) | 0,672 (1) | 0,732 (1) | 0,315 (0) | 0,311 (0) | **0,751** (2) |
| centros, fundo Cellpose | 0,786 (0) | 0,769 (0) | **0,825** (3) | 0,684 (0) | 0,669 (0) | 0,795 (0) |
| GT inteiro, nenhum | 0,839 (37) | 0,727 (25) | 0,444 (0) | 0,940 (37) | 0,942 (37) | 0,698 (22) |
| GT inteiro, fundo Cellpose | 0,841 (34) | 0,860 (37) | 0,820 (26) | 0,833 (34) | 0,832 (34) | 0,864 (37) |

| Massa | 256² | 512² | 1024² | blocos 128 em 128² | blocos 256 em 256² | blocos 256 em 128² |
|---|---|---|---|---|---|---|
| miolo, nenhum | 0,65 | 0,91 | 1,88 | 0,42 | 0,41 | 1,02 |
| miolo, fundo Cellpose | 0,93 | 0,93 | 1,06 | 1,10 | 1,07 | 0,94 |
| GT inteiro, nenhum | 1,38 | 1,79 | 3,66 | 1,13 | 1,13 | 1,93 |

1. **Um marcador pequeno passa do Cellpose.** O miolo com o fundo do Cellpose, com a imagem em 1024², dá 0,852 e supera o
   Cellpose em 27 de 37 imagens. Isso revê a leitura 4 da seção 10.2: o teto com o fundo do Cellpose não é o próprio Cellpose.
   Sem negativo, porém, a entrada em 1024² inunda (GT inteiro: massa 3,66).
2. **Sem negativo, os blocos de 256 px reduzidos para 128² são o modo equilibrado.** O miolo vai de 0,754 para 0,808, com massa
   1,02; os centros, de 0,542 para 0,751. Com o negativo, 0,836. É o único modo que vai bem com e sem negativo. Por isso o modo
   por recortes do pipeline usa esse formato (capítulo 4).
3. **Blocos sem redução quase não expandem.** O miolo fica com massa 0,42, e o GT inteiro dá 0,94: a rede devolve o marcador que
   recebeu.
4. **Hipótese sobre o mecanismo.** A expansão tem um alcance mais ou menos fixo em pixels da entrada da rede. Em 128², o tamanho
   de treino do ScribblePrompt, com o núcleo em cerca de 11 px, esse alcance coincide com a distância entre o miolo e a borda. Com
   o núcleo em 21 px, não alcança; com a entrada em 1024², muito acima do tamanho de treino, a rede inunda. Não foi testado
   diretamente.

## 10.4 Limites dos oráculos

- Os marcadores e negativos são binários. A MarkerUNet produz valores contínuos, que passam pelo corte em $\tau$ (capítulo 7,
  seção 7.3); um marcador aprendido pode se comportar diferente de um binário de mesma área. O capítulo 11 mostra que é o caso.
- Uma única interação, sem a predição anterior como entrada. O ScribblePrompt foi treinado com interações sucessivas.
- Na seção 10.2, limiar e recorte foram escolhidos nas mesmas imagens em que são medidos.
- Só as 37 imagens de treino; as três em 20× são uma amostra pequena.
- O oráculo de resolução não tem os resultados salvos em arquivo, só nas saídas do notebook
  ([08-pendencias.md, PD-47](../estudo/08-pendencias.md#pd-47)).
- Existe um oráculo anterior, feito com a rasterização antiga do GT e 30 imagens. Os números dele não devem ser usados.

## Para o artigo

- **Métodos:** a definição de oráculo e os marcadores e negativos da seção 10.1.
- **Resultados:** a tabela principal da seção 10.2 (ou uma versão reduzida: GT inteiro, miolo e centros, com nenhum negativo,
  fundo do Cellpose e fundo do GT) e as linhas do miolo e do GT inteiro da seção 10.3.
- **Figuras:** as três figuras de `docs/estudo/resultados/oraculo_negativo/` (entradas, resolução e saídas); a de saídas mostra o
  anel de borda perdida sem negativo.
- **Discussão:** a camada expande pouco em 256²; a informação de borda vem do negativo; a expansão depende da resolução de
  entrada; o complemento como negativo é prejudicial.
- **Limitações:** seção 10.4.

## Referências

- Wong, H. E.; Rakic, M.; Guttag, J.; Dalca, A. V. (2024). ScribblePrompt: fast and flexible interactive segmentation for
  any biomedical image. arXiv:2312.07381v3. *Conferida.*
