# 4. Visão geral do método

Este capítulo mostra o método inteiro em blocos, antes de entrar em cada peça. Ele responde a três perguntas: o que entra e o
que sai de cada bloco, o que é treinado e o que fica fixo, e em que resolução cada bloco trabalha. Os capítulos 5 a 9
detalham cada bloco.

## 4.1 Os blocos

```
                         +----------------------+
  imagem H&E  ---------->|  Cellpose-SAM        |  calculado uma vez, antes do treino
  1000 x 1000 RGB        |  (congelado)         |
       |                 +----------+-----------+
       |                            |
       |                  máscara binária de núcleos
       |                            |
       |                 +----------v-----------+
       +---------------->|  MarkerUNet          |  única rede treinada
       |   RGB + máscara |  (pesos treinados)   |
       |   = 4 canais    +----------+-----------+
       |                            |
       |                  marcador difuso, valores em [0, 1]
       |                            |
       |                 +----------v-----------+
       +---------------->|  ScribblePrompt-UNet |  pesos fixos; o gradiente passa por ela
           imagem        |  (congelado)         |
                         +----------+-----------+
                                    |
                         segmentação final, em [0, 1]
                                    |
                         +----------v-----------+
                         |  perda: Dice suave   |  contra o ground truth
                         |  + regularizações    |  sobre o marcador (opcionais)
                         +----------------------+
```

1. **Segmentação inicial (Cellpose-SAM, capítulo 5).** Um segmentador generalista de células, usado sem nenhum ajuste,
   produz uma máscara de núcleos para cada imagem. Ela é calculada uma vez e salva em disco (capítulo 3, seção 3.6).
2. **Rede de marcadores (MarkerUNet, capítulo 6).** Uma U-Net recebe a imagem RGB e a máscara do Cellpose como quarto canal e
   devolve um marcador difuso do tamanho da imagem, com valores em $[0,1]$. É a única parte do método com pesos treinados.
3. **Camada de segmentação (ScribblePrompt-UNet, capítulo 7).** Uma rede de segmentação interativa pré-treinada recebe a
   imagem e o marcador, convertido em rabiscos positivos, e devolve a segmentação final. Os pesos dela não mudam.
4. **Perda (capítulo 8).** O Dice suave compara a segmentação final com o ground truth. Regularizações sobre o marcador,
   como as da tese, podem ser somadas, mas a configuração base usa só o Dice.

Opcionalmente, um **canal de fundo** derivado da máscara do Cellpose é entregue ao ScribblePrompt junto com o marcador. Ele
não passa pela MarkerUNet e não é treinado (capítulo 7).

## 4.2 O que é treinado e por onde passa o gradiente

Só os pesos da MarkerUNet são otimizados. O gradiente da perda vai da segmentação final para trás, atravessa o ScribblePrompt
(cujos pesos ficam fixos, mas cujas operações são diferenciáveis) e chega ao marcador e, dele, à MarkerUNet. O Cellpose fica
fora do grafo de gradiente: a máscara dele é uma entrada fixa, lida do disco.

É a mesma ideia da tese (capítulo 2, seção 2.3): a rede de marcadores aprende sem nunca ver um "marcador correto", só pelo
efeito que o marcador tem na segmentação.

Na implementação, cada bloco é um passo (*Step*) de um pipeline que lê e escreve num dicionário compartilhado: a
[MarkerStep](../../src/pipeline/steps/inference/marker_step.py) grava o marcador em `markers`, e a
[FrozenSegmentationStep](../../src/pipeline/steps/inference/frozen_segmentation_step.py) grava a segmentação final em
`segmentation`. O apêndice A descreve essa estrutura.

## 4.3 Resoluções

As imagens têm 1000 × 1000 pixels, mas nenhuma das duas redes trabalha nessa resolução. O método tem dois modos de operação,
que diferem na forma de lidar com isso.

**Modo imagem inteira.** A imagem é reduzida para a entrada de cada rede e a saída é ampliada de volta:

| Etapa | Resolução |
|---|---|
| Entrada da MarkerUNet | 1000² reduzida para 256² |
| Marcador | 256², ampliado para 1000² |
| Entrada do ScribblePrompt | 1000² reduzida para 256² (128² é o padrão da rede) |
| Segmentação final | ampliada para 1000² |

Nesse modo, um núcleo mediano de 21 px de diâmetro (capítulo 3) fica com cerca de 5 px em 256². Reduções e ampliações são
bilineares ([marker_step.py](../../src/pipeline/steps/inference/marker_step.py),
[scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py)).

**Modo por recortes.** No treino, cada imagem fornece recortes de 256 × 256 em posições sorteadas, 4 por imagem a cada época.
A MarkerUNet vê o recorte sem redução, e o ScribblePrompt o recebe reduzido para 128². Na validação e no teste, a imagem
inteira é dividida em 25 blocos de 256 × 256 com sobreposição de 1/4; cada bloco passa pelo pipeline sozinho, e as saídas são
coladas de volta por média ponderada com uma janela de Hann, que dá menos peso à borda de cada bloco
([tiled_pipeline.py](../../src/pipeline/tiled_pipeline.py), notebook `notebooks/experiments/recortes.ipynb`). A ideia é a da
segmentação em blocos sobrepostos da U-Net (Ronneberger et al., 2015) e da inferência por janela deslizante do nnU-Net, que
sobrepõe os blocos em metade do lado e dá mais peso ao centro de cada um, onde a rede é mais precisa (Isensee et al., 2018,
seção 2.4); o MONAI oferece o mesmo recurso, com sobreposição e combinação configuráveis (Cardoso et al., 2022). A
sobreposição de 1/4 e a janela de Hann com peso mínimo 0,05 são escolhas deste trabalho. Nesse modo, o
núcleo mediano chega ao ScribblePrompt com cerca de 10 px de diâmetro, o dobro do modo imagem inteira.

A escolha entre os dois modos é experimental. A motivação do modo por recortes vem dos oráculos: o ScribblePrompt responde de
forma diferente conforme o tamanho dos objetos na entrada dele (capítulo 10). A comparação entre os modos está no capítulo 11.

## 4.4 Treino e uso

- **Treino.** A MarkerUNet começa com o codificador pré-treinado no ImageNet e o decodificador aleatório. A cada passo, um
  lote de imagens (ou de recortes) atravessa os dois blocos, a perda é calculada e só a MarkerUNet é atualizada. A
  configuração é escolhida por validação cruzada nas 37 imagens de treino (capítulo 9).
- **Uso.** O mesmo pipeline roda em modo de avaliação, sem gradiente. A segmentação final é binarizada com o limiar 0,5 e
  comparada com o ground truth e com a máscara do Cellpose, imagem por imagem (capítulo 1, seção 1.6).

O método produz, portanto, duas saídas: o marcador e a segmentação. Só a segmentação é avaliada contra o ground truth. O
marcador é examinado à parte, para entender o que a rede aprendeu (capítulo 11).

## Para o artigo

- **Métodos:** a figura da seção 4.1, redesenhada como diagrama de blocos com as quatro caixas (Cellpose, MarkerUNet,
  ScribblePrompt, perda), indicando o que é congelado e o caminho do gradiente; o texto das seções 4.1 e 4.2 em um parágrafo.
- **Métodos, detalhes de implementação:** a tabela de resoluções e o modo por recortes com blocos (seção 4.3).

## Referências

As referências de cada bloco estão nos capítulos 5 a 8. Para a inferência em blocos:

- Cardoso, M. J. et al. (2022). MONAI: an open-source framework for deep learning in healthcare. arXiv:2211.02701.
  *Conferida* (seção 2.8, inferência por janela deslizante).
- Isensee, F. et al. (2018). nnU-Net: self-adapting framework for U-Net-based medical image segmentation. arXiv:1809.10486.
  *Conferida* (seção 2.4).
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
  arXiv:1505.04597. *Conferida* (seção 2 e figura 2).
