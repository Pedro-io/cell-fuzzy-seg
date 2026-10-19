# 1. Domínio: núcleos em imagens histológicas

Este capítulo apresenta o ambiente em que o projeto trabalha: o que é uma imagem histológica, por que os núcleos
importam, o que torna difícil segmentá-los e como uma segmentação é avaliada. Os detalhes do conjunto de dados ficam no
capítulo 3, e o método, a partir do capítulo 4.

## 1.1 Da lâmina à imagem digital

A histopatologia estuda tecidos ao microscópio para diagnosticar doenças, em especial o câncer. O tecido retirado do
paciente é fixado, incluído em parafina e cortado em fatias de poucos micrômetros de espessura. As fatias são montadas em
lâminas de vidro e coradas, porque o tecido sem coloração é quase transparente [referência a reunir].

Na patologia digital, a lâmina inteira é digitalizada por um scanner, e o resultado é uma imagem de lâmina inteira
(*whole slide image*, WSI) (Kumar et al., 2020), com dezenas de milhares de pixels de lado [referência a reunir]. Imagens
desse tamanho não cabem inteiras na memória de uma placa de vídeo junto com uma rede neural. Por isso os trabalhos de
segmentação processam recortes da imagem: a U-Net, por exemplo, segmenta imagens grandes bloco a bloco, com blocos
sobrepostos (Ronneberger et al., 2015). O próprio MoNuSeg, conjunto usado neste projeto, foi montado assim: como processar
imagens de lâmina inteira tem custo computacional alto, os autores tiraram de cada uma um recorte de 1000 × 1000 pixels de
uma região densa em núcleos, das imagens do arquivo TCGA (*The Cancer Genome Atlas*) (Kumar et al., 2017).

**Do pixel ao tecido.** Uma imagem digital é uma grade de pixels, e a rede neural só enxerga essa grade. Para saber quanto
tecido real cada pixel representa, duas grandezas importam:

- **aumento:** o quanto a lente da objetiva amplia o tecido na hora de digitalizar, por exemplo 20 ou 40 vezes (20× ou 40×).
  Quanto maior o aumento, menor o pedaço de tecido que cabe em cada pixel, e mais detalhe se vê;
- **micrômetros por pixel (µm/px):** a largura de tecido que um pixel representa. Um micrômetro (µm) é a milésima parte de
  um milímetro. No MoNuSeg, cada arquivo de anotação informa esse valor no atributo `MicronsPerPixel`.

Um exemplo com os números do MoNuSeg deixa a relação concreta. Uma imagem tem 1000 pixels de lado. Em 40×, cada pixel
representa cerca de 0,25 µm, e a imagem inteira cobre 1000 × 0,25 = 250 µm de tecido, um quarto de milímetro. Um núcleo
típico mede cerca de 5 a 6 µm e ocupa, nessa escala, cerca de 21 pixels de diâmetro. Três imagens de treino do MoNuSeg estão
em 20×, com 0,5 µm por pixel: os mesmos 1000 pixels cobrem 500 µm de tecido, uma área quatro vezes maior, e o mesmo núcleo
ocupa só cerca de 12 pixels. É como fotografar a mesma cena de mais longe: cabe mais coisa na foto, e cada coisa aparece
menor.

| Aumento | Tecido por pixel | Tecido coberto pela imagem de 1000 px | Diâmetro mediano do núcleo |
|---|---|---|---|
| 40× | cerca de 0,25 µm | cerca de 250 µm | 21,3 px (cerca de 5,3 µm) |
| 20× | 0,5005 µm | cerca de 500 µm | 12,2 px (cerca de 6,1 µm) |

O diâmetro é o do círculo de mesma área que o contorno anotado; valores medidos nos 51 XMLs (ver
[11-oraculos.md](../estudo/11-oraculos.md), tabela de termos). Em micrômetros, os núcleos têm praticamente o mesmo tamanho
nos dois aumentos. Para a rede neural, que só conta pixels, os núcleos das imagens em 20× parecem ter metade do tamanho.

## 1.2 A coloração H&E

A coloração mais usada na rotina é a **hematoxilina e eosina** (H&E), um esquema de coloração geral e barato. A hematoxilina
deixa os núcleos em azul-arroxeado escuro, e a eosina deixa o estroma (o tecido conjuntivo) em rosa (Kumar et al., 2017;
Kumar et al., 2020). A cor dos núcleos vem da ligação da hematoxilina a estruturas ácidas, como o DNA (Fischer et al.,
2008). Numa imagem H&E, portanto, os núcleos aparecem como manchas escuras sobre um fundo rosado, e é essa diferença de cor
que torna a segmentação de núcleos possível só com a imagem RGB. Todas as imagens do MoNuSeg são coradas com H&E e foram
digitalizadas em 40× (Kumar et al., 2017).

A cor de uma mesma estrutura varia de um laboratório para outro, por diferenças nas práticas de coloração e no equipamento
de digitalização. Por isso um método treinado com imagens de uma instituição pode piorar nas de outra. O MoNuSeg foi montado
com imagens de 18 hospitais e de vários órgãos justamente para que os métodos treinados nele generalizem (Kumar et al., 2017).

## 1.3 Por que segmentar núcleos

A avaliação do patologista se apoia na aparência, no tamanho, na forma, na cor e na aglomeração dos núcleos (Kumar et al.,
2020). Características como a densidade de núcleos, a razão entre núcleo e citoplasma, o tamanho médio e o pleomorfismo
(variação de forma e tamanho) ajudam a graduar tumores e a prever a resposta ao tratamento (Kumar et al., 2017); na graduação
de Nottingham para o câncer de mama, por exemplo, o pleomorfismo nuclear é um dos três critérios (Elston e Ellis, 1991).
Medir essas características de forma automática e em grande escala exige, antes, saber quais pixels pertencem a cada núcleo.
Por isso a segmentação de núcleos é um módulo central de vários fluxos de patologia computacional (Kumar et al., 2017).

Fazer isso à mão é caro. Uma única imagem do MoNuSeg, que cobre cerca de 250 × 250 µm de tecido em 40×, tem em média 652
núcleos anotados no treino (24.140 em 37 imagens) e 478 no teste (6.697 em 14 imagens), cada um com o contorno desenhado
(contagem das regiões nos XMLs, [data_source/README.md](../../data_source/README.md)). Esse custo de anotação é o que motiva
métodos automáticos e, no caso deste projeto, métodos que reduzem o trabalho humano na segmentação interativa (capítulo 2).

## 1.4 O que torna a tarefa difícil

- **Muitos objetos pequenos.** Centenas de núcleos por imagem, cada um com cerca de 20 px de diâmetro em 40×. Erros de um
  ou dois pixels na borda já pesam na métrica de um objeto tão pequeno.
- **Núcleos que se tocam ou se sobrepõem.** Em tecidos densos, os núcleos encostam uns nos outros, e a fronteira entre dois
  deles pode não ter contraste nenhum. No MoNuSeg, os contornos anotados chegam a se sobrepor (seção 1.5).
- **Variedade de aparência.** Tipos celulares diferentes têm núcleos de tamanhos e texturas diferentes; linfócitos, por
  exemplo, têm núcleos pequenos e muito escuros [referência a reunir].
- **Variação de cor e de escala.** A coloração muda entre instituições (seção 1.2), e o aumento muda o tamanho do núcleo em
  pixels (seção 1.1).
- **Desequilíbrio entre classes, moderado.** Os pixels de núcleo são, em média, 24,4% da imagem no treino (de 10,6% a 40,0%
  conforme a imagem) e 20,0% no teste (de 9,4% a 28,0%). Medido em 2026-10-17 sobre as máscaras binárias de
  `data_source/MoNuSegPreprocessed/{train,test}/ground_truth/`.

## 1.5 Segmentação semântica e segmentação por instância

Há dois jeitos de definir a tarefa:

- **segmentação semântica:** cada pixel recebe uma classe, aqui "núcleo" ou "fundo". O resultado é uma máscara binária, e
  ela não diz onde termina um núcleo e começa o vizinho;
- **segmentação por instância:** cada núcleo recebe um rótulo próprio. O resultado separa núcleos que se tocam e permite
  contá-los e medi-los um a um.

As anotações do MoNuSeg são por instância: cada núcleo é um polígono no XML (Kumar et al., 2017). Este projeto, porém, trabalha
com segmentação **semântica**. Os polígonos do XML são rasterizados numa máscara por instância, e o ground truth usado no
treino e na avaliação é a versão binária dela: todo pixel com algum núcleo vale 1
([monuseg_dataset.py:124-125](../../src/data/load/monuseg_dataset.py#L124-L125)). A saída do método também é uma máscara
binária. A escolha segue uma sugestão do coordenador do trabalho e está registrada no capítulo 12.

A consequência é que núcleos que se tocam viram uma região só. No GT binário atual, as 24.140 regiões anotadas do treino
formam 19.326 componentes conexos, 20% a menos; no teste, 6.697 regiões formam 6.221 componentes, 7% a menos (componentes com
vizinhança de 8 pixels, medido em 2026-10-17 sobre as mesmas máscaras da seção 1.4). A perda se concentra nas imagens
densas do treino: só as três imagens em 20×, com núcleos pequenos e muito próximos, respondem por 27% dos componentes perdidos
(mesma medida; ver também [08-pendencias.md, PD-24](../estudo/08-pendencias.md#pd-24)). A máscara por instância não vai para
o treino nem para a avaliação; ela só é usada para construir o mapa de distância de uma das perdas
([distance_map_step.py](../../src/pipeline/steps/preprocessing/distance_map_step.py); capítulo 8).

Para segmentação por instância, a literatura usa métricas próprias, como o *Aggregated Jaccard Index* (AJI), proposto junto
com o MoNuSeg (Kumar et al., 2017), e a *Panoptic Quality* (PQ) (Kirillov et al., 2019). Elas não se aplicam diretamente a uma
saída binária e não são usadas aqui. O desafio MoNuSeg classificou os participantes pelo AJI médio no conjunto de teste,
justamente para privilegiar a segmentação por instância em vez da semântica (Kumar et al., 2020). Os números deste trabalho,
portanto, não são comparáveis aos da classificação do desafio.

## 1.6 Como uma segmentação semântica é medida

A avaliação compara, pixel a pixel, a máscara prevista com o ground truth. Antes da comparação, a predição contínua (por
exemplo, a saída de uma sigmoide) é binarizada com o limiar 0,5, e o ground truth é tomado como `> 0`
([metrics.py](../../src/evaluation/metrics.py)). Cada pixel cai então em um de quatro casos: verdadeiro positivo (TP),
falso positivo (FP), falso negativo (FN) ou verdadeiro negativo (TN). As métricas usadas no projeto são:

| Métrica | Fórmula | O que diz |
|---|---|---|
| Dice | $\dfrac{2\,TP}{2\,TP + FP + FN}$ | sobreposição entre predição e GT; é o F1 dos pixels (Dice, 1945) |
| IoU (Jaccard) | $\dfrac{TP}{TP + FP + FN}$ | interseção sobre união (Jaccard, 1912) |
| Precisão | $\dfrac{TP}{TP + FP}$ | dos pixels marcados como núcleo, quantos são núcleo |
| Revocação | $\dfrac{TP}{TP + FN}$ | dos pixels de núcleo, quantos foram encontrados |
| Razão de massa | $\dfrac{TP + FP}{TP + FN}$ | área prevista sobre a área do GT; acima de 1, o método segmenta a mais |

O verdadeiro negativo não entra em nenhuma delas. Isso importa porque o fundo é a maior parte da imagem (seção 1.4): uma
acurácia por pixel seria alta mesmo para um método que não encontra núcleo nenhum.

Algumas relações úteis para ler os resultados:

- numa mesma imagem, IoU e Dice dão a mesma ordem entre métodos, porque $IoU = \dfrac{Dice}{2 - Dice}$. A média do IoU sobre
  várias imagens, porém, não é a conversão da média do Dice, então os dois são reportados;
- precisão e revocação separam os dois tipos de erro que o Dice mistura. Um método que "incha" os núcleos tem revocação alta
  e precisão baixa, e a razão de massa acima de 1 mostra isso de forma direta. Esse diagnóstico é usado no capítulo 11;
- quando a imagem (ou um recorte dela) não tem núcleo nem predição, Dice e IoU valem 1 por convenção; quando não há pixel
  previsto, a precisão vale 1; quando o GT está vazio, a revocação vale 1 ([metrics.py](../../src/evaluation/metrics.py)).

As métricas são calculadas **por imagem**, e o resultado de um método é a média sobre as imagens. Isso é diferente de somar
TP, FP e FN de todas as imagens e calcular um Dice só: na soma, as imagens com mais núcleos pesariam mais. O cálculo por
imagem também permite a comparação pareada com a linha de base (o mesmo par de números para cada imagem), que é a forma de
comparação usada em todo o trabalho (capítulo 9).

A perda usada no treino é uma versão contínua do Dice, calculada sem binarizar a predição. Ela é apresentada no capítulo 8.

## Para o artigo

- **Introdução:** seções 1.1 a 1.3, em versão curta: patologia digital, H&E, a importância dos núcleos e o custo da anotação.
  Precisam das referências marcadas.
- **Dados (Métodos):** os números das seções 1.1, 1.4 e 1.5 (escala, fração de pixels de núcleo, fusão dos núcleos no GT
  binário) complementam o capítulo 3.
- **Métricas (Métodos):** a seção 1.6, com as fórmulas, o limiar, a média por imagem e a comparação pareada.
- **Limitações:** a avaliação semântica não separa núcleos que se tocam e não é comparável ao AJI do desafio (seção 1.5).

## Referências

- Dice, L. R. (1945). Measures of the amount of ecologic association between species. *Ecology*, 26(3), 297-302. *A conferir.*
- Elston, C. W.; Ellis, I. O. (1991). Pathological prognostic factors in breast cancer. I. The value of histological grade in
  breast cancer: experience from a large study with long-term follow-up. *Histopathology*, 19(5), 403-410. *A conferir.*
- Fischer, A. H.; Jacobson, K. A.; Rose, J.; Zeller, R. (2008). Hematoxylin and eosin staining of tissue and cell sections.
  *Cold Spring Harbor Protocols*. *A conferir.*
- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37-50. *A conferir.*
- Kirillov, A.; He, K.; Girshick, R.; Rother, C.; Dollár, P. (2019). Panoptic segmentation. *CVPR*. *A conferir.*
- Kumar, N.; Verma, R.; Sharma, S.; Bhargava, S.; Vahadane, A.; Sethi, A. (2017). A dataset and a technique for generalized
  nuclear segmentation for computational pathology. *IEEE Transactions on Medical Imaging*, 36(7), 1550-1560.
  doi:10.1109/TMI.2017.2677499. Citação exigida pelos autores do MoNuSeg. *Conferida* (seções I, II-A e III).
- Kumar, N. et al. (2020). A multi-organ nucleus segmentation challenge. *IEEE Transactions on Medical Imaging*, 39(5),
  1380-1391. doi:10.1109/TMI.2019.2947628. Citação exigida pelos autores do MoNuSeg. *Conferida* no manuscrito aceito
  (resumo, seções II-A e III); volume e páginas vêm de repositórios institucionais.
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
  arXiv:1505.04597. *Conferida* (seção 2 e figura 2, estratégia de blocos sobrepostos).
