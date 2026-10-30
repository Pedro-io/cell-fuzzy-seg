# 4. Visão geral do método

Este capítulo mostra o método inteiro em blocos, antes de entrar em cada peça. Ele responde a quatro perguntas: o que entra e
o que sai de cada bloco, o que é treinado e o que fica fixo, em que resolução cada bloco trabalha e qual é a configuração final.
Os capítulos 5 a 9 detalham cada bloco.

## 4.1 Os blocos

```
                         +----------------------+
  imagem H&E  ---------->|  Cellpose-SAM        |  calculado uma vez, antes do treino
  1000 x 1000 RGB        |  (congelado)         |
       |                 +----------+-----------+
       |                            |
       |                  máscara binária de núcleos ------------------+
       |                            |                                  |
       |                 +----------v-----------+                      |
       +---------------->|  MarkerUNet          |  única rede treinada |
       |   RGB + máscara |  (pesos treinados)   |                      |
       |   = 4 canais    +----------+-----------+                      |
       |                            |                                  |
       |                  marcador difuso, valores em [0, 1]           |
       |                            |                                  |
       |                 +----------v-----------+                      |
       |                 |  Fechamento          |  operação fixa,      |
       |                 |  morfológico (3 px)  |  sem pesos           |
       |                 +----------+-----------+                      |
       |                            |                                  |
       |                  marcador fechado                  canal de fundo:
       |                            |                       pixels a mais de
       |                 +----------v-----------+           2 px de qualquer
       +---------------->|  ScribblePrompt-UNet |<----------núcleo do Cellpose
           imagem        |  (congelado)         |
                         +----------+-----------+
                                    |
                         segmentação final, em [0, 1]
                                    |
                         +----------v-----------+
                         |  perda: Dice suave   |  contra o ground truth
                         +----------------------+
```

1. **Segmentação inicial (Cellpose-SAM, capítulo 5).** Um segmentador generalista de células, usado sem nenhum ajuste,
   produz uma máscara de núcleos para cada imagem. Ela é calculada uma vez e salva em disco (capítulo 3, seção 3.6).
2. **Rede de marcadores (MarkerUNet, capítulo 6).** Uma U-Net recebe a imagem RGB e a máscara do Cellpose como quarto canal e
   devolve um marcador difuso do tamanho da imagem, com valores em $[0,1]$. É a única parte do método com pesos treinados.
3. **Fechamento morfológico (seção 4.2).** Uma operação clássica de processamento de imagens, sem pesos, que tapa buracos e
   frestas do marcador. É opcional; na configuração final, usa um disco de 3 px no treino e na avaliação.
4. **Camada de segmentação (ScribblePrompt-UNet, capítulo 7).** Uma rede de segmentação interativa pré-treinada recebe a
   imagem, o marcador, convertido em rabiscos positivos, e um **canal de fundo** tirado da máscara do Cellpose, e devolve a
   segmentação final. Os pesos dela não mudam. O canal de fundo não passa pela MarkerUNet e não é treinado (capítulo 7, seção 7.4).
5. **Perda (capítulo 8).** O Dice suave compara a segmentação final com o ground truth. Regularizações sobre o marcador, como
   as da tese, podem ser somadas, mas a configuração final usa só o Dice.

O método produz duas saídas: o **marcador** e a **segmentação**. A segmentação é avaliada contra o ground truth (capítulo 1,
seção 1.6). O marcador é avaliado pela **editabilidade**: o quanto ele se parece com "um traço por núcleo", que uma pessoa
poderia aceitar ou corrigir (capítulo 9, seção 9.7).

## 4.2 O fechamento morfológico

**O que faz.** O marcador aprendido tende a sair esburacado e partido em vários pedaços dentro de um mesmo núcleo (capítulo 11).
O fechamento é uma dilatação seguida de uma erosão pelo mesmo elemento estruturante, aqui um disco de raio $r$ px
([marker_closing_step.py](../../src/pipeline/steps/inference/marker_closing_step.py)):

- a **dilatação** troca o valor de cada pixel pelo **máximo** dos valores no disco em volta dele. O marcador "engorda" e os
  buracos e frestas menores que o disco se fecham;
- a **erosão** troca cada valor pelo **mínimo** no disco. O marcador volta ao contorno de antes, mas os buracos que a dilatação
  fechou continuam fechados.

Num marcador difuso, máximo e mínimo são a dilatação e a erosão em tons de cinza com elemento estruturante plano (Haralick,
Sternberg e Zhuang, 1987; Maragos e Schafer, 1987). Pixels fora da imagem não contam, então a borda da imagem não cria nem apaga marcador.

**Três propriedades usadas no projeto**, todas conferidas em testes
([test_marker_closing_step.py](../../tests/test_marker_closing_step.py)):

- o fechamento é **extensivo**: nunca diminui o marcador, só tapa (Maragos e Schafer, 1987);
- ele **comuta com o corte em $\tau$**: fechar o marcador difuso e depois cortar em $\tau$ dá o mesmo que cortar e depois fechar,
  porque os operadores planos comutam com a limiarização (Maragos e Schafer, 1987) e o corte é crescente e contínuo (dedução
  conferida num teste). Então dá no mesmo pensar no fechamento sobre o marcador ou sobre os rabiscos que o ScribblePrompt recebe;
- o **gradiente passa** por ele: máximo e mínimo deixam o gradiente seguir para o pixel que vence em cada vizinhança. Por isso o
  fechamento pode ficar dentro do treino, e a MarkerUNet aprende sabendo que o marcador será fechado.

**Por que foi incluído.** O coordenador do trabalho sugeriu "tapar os buraquinhos" dos marcadores com uma operação clássica,
para facilitar a edição humana, o que seria uma contribuição do trabalho. A tese propõe operadores morfológicos como o
fechamento como trabalho futuro, para marcadores mais fáceis de editar, lá como uma perda (Fonseca, 2026, seção 7.1.4); aqui ele
é uma operação sobre o marcador. Nos experimentos, o fechamento de 3 px não mudou o Dice e levou a fração das regiões do GT com
um único componente de marcador de 18% para 90% (capítulo 11). O raio de 3 px foi escolhido entre 1, 2, 3 e 5 px: com 5 px o
Dice cai e os marcadores de núcleos vizinhos se unem.

## 4.3 O que é treinado e por onde passa o gradiente

Só os pesos da MarkerUNet são otimizados. O gradiente da perda vai da segmentação final para trás, atravessa o ScribblePrompt
(cujos pesos ficam fixos, mas cujas operações são diferenciáveis), atravessa o corte em $\tau$ e o fechamento e chega ao marcador
e, dele, à MarkerUNet. O Cellpose fica fora do grafo de gradiente: a máscara dele e o canal de fundo são entradas fixas, lidas do
disco.

É a mesma ideia da tese (capítulo 2, seção 2.3): a rede de marcadores aprende sem nunca ver um "marcador correto", só pelo
efeito que o marcador tem na segmentação.

Na implementação, cada bloco é um passo (*Step*) de um pipeline que lê e escreve num dicionário compartilhado: a
[MarkerStep](../../src/pipeline/steps/inference/marker_step.py) grava o marcador em `markers`, a
[MarkerClosingStep](../../src/pipeline/steps/inference/marker_closing_step.py) grava o marcador fechado em `markers_closed`, e a
[FrozenSegmentationStep](../../src/pipeline/steps/inference/frozen_segmentation_step.py) lê o marcador indicado e grava a
segmentação final em `segmentation`. As perdas sobre o marcador olham o marcador cru, `markers` (capítulo 8, seção 8.1). O
apêndice A descreve essa estrutura.

## 4.4 Resoluções

As imagens têm 1000 × 1000 pixels, mas nenhuma das duas redes trabalha nessa resolução. O método teve dois modos de operação;
o atual é o modo por recortes.

**Modo imagem inteira (experimentos iniciais).** A imagem é reduzida para a entrada de cada rede e a saída é ampliada de volta:
1000² para 256² na MarkerUNet e no ScribblePrompt, com o núcleo mediano de 21 px (capítulo 3) reduzido a cerca de 5 px. Reduções
e ampliações são bilineares.

**Modo por recortes (atual).** No treino, cada imagem fornece 4 recortes de 256 × 256 em posições sorteadas a cada época. A
MarkerUNet vê o recorte sem redução, e o ScribblePrompt o recebe reduzido para 128², com o núcleo mediano em cerca de 10 px. Na
validação e no teste, a imagem inteira é dividida em blocos de 256 × 256 que se sobrepõem; cada bloco passa pelo pipeline sozinho,
e as saídas são coladas de volta por média ponderada com uma janela de Hann, que dá menos peso à borda de cada bloco, onde a rede
vê menos contexto ([tiled_pipeline.py](../../src/pipeline/tiled_pipeline.py)).

A ideia é a da segmentação em blocos sobrepostos da U-Net (Ronneberger et al., 2015) e da inferência por janela deslizante do
nnU-Net, que sobrepõe os blocos em metade do lado e dá mais peso ao centro de cada um (Isensee et al., 2018, seção 2.4); o MONAI
oferece o mesmo recurso, com sobreposição e combinação configuráveis (Cardoso et al., 2022). A janela de Hann com peso mínimo 0,05
é escolha deste trabalho.

A motivação do modo por recortes vem dos oráculos: o ScribblePrompt expande o marcador de forma diferente conforme o tamanho dos
objetos na entrada dele, e os blocos de 256 px reduzidos para 128² foram o modo equilibrado (capítulo 10). No treino, os recortes
subiram o Dice de 0,785 para 0,831, em todas as 37 imagens (capítulo 11).

## 4.5 Configuração final

A configuração escolhida pela validação cruzada e avaliada no teste é a `recortes_negativo_r2_fechamento`
(`notebooks/experiments/recortes_negativo_r2_fechamento.ipynb`; capítulo 11, seção 11.7):

| Parte | Escolha |
|---|---|
| Entrada da MarkerUNet | RGB e máscara binária do Cellpose; recortes de 256² sem redução |
| Fechamento | disco de 3 px, no treino e na avaliação |
| Conversão do marcador | corte em $\tau = 0{,}5$ (modo `positive`) |
| Canal de fundo | pixels a mais de 2 px de qualquer núcleo do Cellpose |
| ScribblePrompt | entrada de 128², uma interação |
| Perda | só o Dice suave, sem nenhuma regularização sobre o marcador |
| Avaliação durante o treino | blocos de 256² com sobreposição de 1/4 |
| Inferência final | blocos com sobreposição de 1/2 e aumento no teste (TTA): a média das saídas para as 8 combinações de rotação de 90° e espelhamento da imagem |

| Treino final | as 37 imagens, por 200 épocas (capítulo 9, seção 9.4) |

O aumento no teste e a sobreposição maior só mudam a inferência, não o treino: subiram o Dice de 0,8383 para 0,8402 na validação
cruzada e de 0,8300 para 0,8333 no teste (capítulo 11). Hoje, as duas coisas estão implementadas nos notebooks, não em `src/`.

A forma editável do marcador vem do fechamento, que é uma operação fixa, e não de uma perda: a configuração final não usa nenhuma
das regularizações da tese, porque elas pioraram o Dice e a editabilidade (capítulo 8, seção 8.6). O canal de fundo do Cellpose
também não é uma perda: é uma entrada fixa do ScribblePrompt.

## 4.6 Treino e uso

- **Treino.** A MarkerUNet começa com o codificador pré-treinado no ImageNet e o decodificador aleatório. A cada passo, um
  lote de recortes atravessa o pipeline, a perda é calculada e só a MarkerUNet é atualizada. A configuração é escolhida por
  validação cruzada nas 37 imagens de treino (capítulo 9).
- **Uso.** O mesmo pipeline roda em modo de avaliação, sem gradiente. A segmentação final é binarizada com o limiar 0,5 e
  comparada com o ground truth e com a máscara do Cellpose, imagem por imagem (capítulo 1, seção 1.6). O marcador fechado é o
  que se entregaria a uma pessoa para edição.

## Para o artigo

- **Métodos:** a figura da seção 4.1, redesenhada como diagrama de blocos (Cellpose, MarkerUNet, fechamento, ScribblePrompt com
  o canal de fundo, perda), indicando o que é congelado e o caminho do gradiente; as duas saídas, segmentação e marcador.
- **Métodos:** o fechamento (seção 4.2), com as três propriedades e o motivo; é parte da contribuição de editabilidade.
- **Detalhes de implementação:** o modo por recortes com blocos (seção 4.4) e a tabela da configuração final (seção 4.5).

## Referências

As referências de cada bloco estão nos capítulos 5 a 8. Para o fechamento e a inferência em blocos:

- Cardoso, M. J. et al. (2022). MONAI: an open-source framework for deep learning in healthcare. arXiv:2211.02701.
  *Conferida* (seção 2.8, inferência por janela deslizante).
- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seção 7.1.4, operadores morfológicos para marcadores editáveis).
- Haralick, R. M.; Sternberg, S. R.; Zhuang, X. (1987). Image analysis using mathematical morphology. *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, 9(4), 532-550. doi:10.1109/TPAMI.1987.4767941. *Conferida* (dilatação e erosão em
  tons de cinza calculadas por máximo e mínimo).
- Isensee, F. et al. (2018). nnU-Net: self-adapting framework for U-Net-based medical image segmentation. arXiv:1809.10486.
  *Conferida* (seção 2.4).
- Maragos, P.; Schafer, R. W. (1987). Morphological filters, Part I: their set-theoretic analysis and relations to linear
  shift-invariant filters. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(8), 1153-1169.
  doi:10.1109/TASSP.1987.1165259. *Conferida* (seção II-D: erosão, dilatação, abertura e fechamento de funções por
  conjuntos comutam com a limiarização; seção III-A: o fechamento de um conjunto o contém).
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
  arXiv:1505.04597. *Conferida* (seção 2 e figura 2).
