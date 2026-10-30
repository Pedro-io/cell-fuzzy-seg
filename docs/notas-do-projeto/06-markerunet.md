# 6. Rede de marcadores: MarkerUNet

A MarkerUNet é a única parte treinada do método (capítulo 4). Ela recebe a imagem e a máscara do Cellpose e devolve um
marcador difuso, um mapa do tamanho da imagem com valores em $[0,1]$. Este capítulo descreve a arquitetura, a entrada, a
inicialização e a forma como a rede é executada no treino e na avaliação.

## 6.1 Arquitetura

A MarkerUNet é uma U-Net (Ronneberger et al., 2015) com codificador ResNet34 (He et al., 2016), montada com a biblioteca
*segmentation_models_pytorch* (smp), versão 0.5.0 ([marker_unet.py](../../src/models/networks/marker_unet.py),
[marker_unet_config.py](../../src/models/configs/marker_unet_config.py)):

| Configuração | Valor |
|---|---|
| Arquitetura | `smp.Unet` |
| Codificador | ResNet34, 5 estágios, pesos pré-treinados no ImageNet |
| Canais de entrada | 4 (RGB e a máscara do Cellpose) |
| Saída | 1 canal, sem ativação (logits) |
| Parâmetros | 24.439.505, todos treinados |

Numa U-Net, o codificador reduz a resolução em estágios, extraindo características cada vez mais abstratas, e o
decodificador volta à resolução de entrada. As conexões de atalho (*skip connections*) levam os detalhes finos de cada
estágio do codificador ao estágio correspondente do decodificador, o que ajuda a localizar bordas com precisão. Só o
codificador vem pré-treinado; o decodificador começa com pesos aleatórios.

A sigmoide que transforma os logits em marcador é aplicada fora da rede, no passo
[MarkerStep](../../src/pipeline/steps/inference/marker_step.py).

**Relação com a tese.** A tese usa duas variantes de U-Net com codificadores ResNet: na prova de conceito, um único
codificador ResNet18 com a imagem e a indicação do objeto concatenadas numa entrada de 4 canais; no modelo de marcador de
objeto, três codificadores ResNet34, um para a imagem, um para as segmentações de objeto e fundo e um para o mapa de
saliência da hierarquia (Fonseca, 2026, seções 5.4.1 e 5.4.2). A MarkerUNet segue a ideia da prova de conceito, com fusão
precoce num único codificador, e usa o ResNet34 do modelo de objeto. Ela não recebe a hierarquia, porque a camada de
segmentação deste trabalho não usa hierarquias (capítulo 7).

## 6.2 Entrada

A entrada é o tensor `rgba` montado na preparação dos dados (capítulo 3, seção 3.6), com 4 canais em $[0,1]$:

1. a imagem RGB dividida por 255;
2. a máscara binária do Cellpose como quarto canal.

Uma variante, escolhida na leitura dos dados, troca a máscara binária pela probabilidade de célula do Cellpose
([load_preprocessed](../../src/data/load/preprocessed_dataset.py), opção `alpha="prob"`). A probabilidade carrega mais
informação que a máscara, por exemplo a incerteza nas bordas e nos núcleos que o Cellpose quase encontrou. Essa variante
ainda não foi avaliada ([08-pendencias.md, PD-34](../estudo/08-pendencias.md#pd-34)).

**Sem normalização do ImageNet.** Os pesos pré-treinados no ImageNet esperam entradas padronizadas pela média e pelo desvio
de cada canal; a smp oferece uma função para isso. A MarkerUNet recebe os valores em $[0,1]$ sem essa padronização. As
camadas de normalização em lote (*BatchNorm*) do codificador compensam parte da diferença durante o treino. Hipótese: o
ponto de partida pré-treinado fica menos aproveitado; o efeito só se sabe medindo
([08-pendencias.md, PD-36](../estudo/08-pendencias.md#pd-36)).

## 6.3 Como os pesos do ImageNet viram 4 canais

Os pesos do ImageNet têm 3 canais de entrada. Com `in_channels=4`, a smp 0.5.0 copia os pesos da primeira convolução em
ciclo e multiplica todos por 3/4 (conferido no código da biblioteca, função `patch_first_conv`):

| Canal de entrada | Pesos iniciais da primeira convolução |
|---|---|
| R | pesos de R × 0,75 |
| G | pesos de G × 0,75 |
| B | pesos de B × 0,75 |
| máscara do Cellpose | cópia dos pesos de R × 0,75 |

No início do treino, portanto, a máscara do Cellpose é vista como se fosse mais vermelho na imagem. O fator 3/4 mantém a
escala da ativação parecida com a original. A documentação da função diz que os canais além do terceiro recebem
inicialização aleatória, mas o código faz a cópia descrita acima.

## 6.4 Execução no treino e na avaliação

A [MarkerStep](../../src/pipeline/steps/inference/marker_step.py) executa a rede de forma diferenciável:

1. reduz a entrada para 256 × 256 por interpolação bilinear, se ela for maior (no modo por recortes, o recorte já tem 256 ×
   256 e passa sem redução);
2. aplica a MarkerUNet e a sigmoide;
3. amplia o marcador de volta ao tamanho da entrada, também por interpolação bilinear.

O marcador nunca é binarizado: ele segue contínuo, passando pelo fechamento morfológico na configuração final (capítulo 4, seção
4.2), até a camada de segmentação, e é isso que permite o gradiente passar.

**Modos de treino e de avaliação.** A rede tem camadas de normalização em lote no codificador e no decodificador. Durante o
treino, elas usam as estatísticas do lote corrente e atualizam as médias acumuladas; na avaliação, usam as médias
acumuladas. O passo expõe `set_training`, chamado pelo laço de treino e pela função de avaliação, para alternar entre os dois
modos ([metrics.py](../../src/evaluation/metrics.py), `evaluate`). Na avaliação, o mesmo caminho diferenciável roda sem
gradiente.

O passo também tem um modo de inferência que binariza o marcador com limiar 0,5. Os experimentos não usam esse modo.

## 6.5 Tamanho do modelo e quantidade de dados

São 24,4 milhões de parâmetros treinados com 37 imagens, sem congelar nenhuma parte. Duas alternativas que reduzem o número
de parâmetros livres, congelar o codificador e usar a padronização do ImageNet, serão testadas juntas numa ablação depois de
escolhida a configuração principal ([08-pendencias.md, PD-38](../estudo/08-pendencias.md#pd-38)). A aumentação por rotações e
espelhamentos e o treino por recortes também ajudam a compensar o conjunto pequeno (capítulos 3 e 9).

## Para o artigo

- **Métodos:** seção 6.1 (arquitetura, em uma frase com a tabela) e seção 6.2 (entrada de 4 canais, fusão precoce); a
  relação com a tese em uma frase.
- **Detalhes de implementação:** a inicialização do quarto canal (seção 6.3) e a resolução de 256² (seção 6.4).
- **Ablações:** padronização do ImageNet e congelamento do codificador (seção 6.5), probabilidade do Cellpose no lugar da
  máscara (seção 6.2), quando forem rodadas.

## Referências

- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seções 5.4.1 e 5.4.2).
- He, K.; Zhang, X.; Ren, S.; Sun, J. (2016). Deep residual learning for image recognition. *CVPR*. arXiv:1512.03385.
  *Conferida* (seção 4.1, rede de 34 camadas).
- Iakubovskii, P. Segmentation Models Pytorch. <https://github.com/qubvel/segmentation_models.pytorch>. *A conferir* (forma
  de citação recomendada pelo projeto).
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
  arXiv:1505.04597. *Conferida* (seção 2).
