# Referências

Lista central das referências das Notas do Projeto: as que já foram lidas, as que são citadas mas ainda não foram lidas e as
afirmações que ainda não têm referência. Cada capítulo mantém a sua seção "Referências" com a referência completa; este
documento é o índice para saber o que falta levantar.

Os PDFs ficam em `docs/papers/`, que está no `.gitignore` por decisão do autor: eles existem só na cópia local do repositório.

**Situação:** *conferida* quer dizer que o trecho citado foi lido no PDF; *a conferir* quer dizer que a referência é citada,
mas o texto ainda não foi lido, e ela não deve ir para o artigo antes disso.

## 1. Disponíveis e conferidas

| Referência | Arquivo em `docs/papers/` | Usada nos capítulos | Observação |
|---|---|---|---|
| Fonseca (2026), *Hierarchy-Based Fuzzy Segmentation and Marker Learning*, tese | `Hierarchy-Based Fuzzy Segmentation and Marker Learning.pdf` | 2, 4, 6, 7, 8, 13 | a seção 7.1.4 propõe operadores morfológicos para marcadores editáveis |
| Kumar et al. (2017), *A dataset and a technique for generalized nuclear segmentation for computational pathology*, IEEE TMI 36(7) | `IEEE_TMI_NuceliSegmentation.pdf` | 1, 3 | citação exigida pelo MoNuSeg |
| Kumar et al. (2020), *A multi-organ nucleus segmentation challenge*, IEEE TMI 39(5) | `A multi-organ nucleus segmentation challenge.pdf` | 1, 3 | o PDF é o manuscrito aceito; citar a versão publicada (doi:10.1109/TMI.2019.2947628); citação exigida pelo MoNuSeg |
| Pachitariu, Rariden e Stringer (2025), *Cellpose-SAM*, bioRxiv | `cellpose_sam.pdf` | 3, 5 | pré-publicação, sem revisão por pares |
| Stringer et al. (2021), *Cellpose: a generalist algorithm for cellular segmentation* | `cellpose.pdf` | 5 | o PDF é a pré-publicação de 2020; autores, volume e páginas da versão publicada (Nature Methods) a conferir |
| Wong et al. (2024), *ScribblePrompt*, arXiv:2312.07381v3 | `scribble_prompt.pdf` | 2, 7, 10 | local de publicação definitivo a conferir |
| Kirillov et al. (2023), *Segment anything*, arXiv:2304.02643 | `sam.pdf` | 5, 7 | |
| Ronneberger, Fischer e Brox (2015), *U-Net*, arXiv:1505.04597 | `U-Net: Convolutional Networks for Biomedical.pdf` | 1, 4, 6 | |
| He et al. (2016), *Deep residual learning for image recognition*, arXiv:1512.03385 | `resnet.pdf` | 6 | |
| Milletari, Navab e Ahmadi (2016), *V-Net*, arXiv:1606.04797 | `V-Net: Fully Convolutional Neural Networks for.pdf` | 8 | |
| Isensee et al. (2018), *nnU-Net: self-adapting framework for U-Net-based medical image segmentation*, arXiv:1809.10486 | `nnunet.pdf` | 4, 9 | é a versão de 2018 do arXiv, não o artigo da Nature Methods (2021); descreve só o Dice como perda |
| Haralick, Sternberg e Zhuang (1987), *Image analysis using mathematical morphology*, IEEE TPAMI 9(4) | `Haralick_Sternberg_Zhuang_1987_Image_Analysis_Using_Mathematical_Morphology.pdf` | 4 | dilatação e erosão em tons de cinza por máximo e mínimo |
| Maragos e Schafer (1987), *Morphological filters, Part I*, IEEE TASSP 35(8) | `Maragos_Schafer_1987_Morphological_Filters_Part_I.pdf` | 4 | seção II-D: comutação com a limiarização; seção III-A: o fechamento contém o conjunto |
| Cardoso et al. (2022), *MONAI: an open-source framework for deep learning in healthcare*, arXiv:2211.02701 | `monai.pdf` | 4 | |

## 2. Citadas e ainda não lidas

| Referência | Capítulos | Para quê |
|---|---|---|
| Costa et al. (2023), SIBGRAPI, anais estendidos | 2 | exemplo de imagens de células com centenas de casos por imagem; citada pela tese |
| Dice (1945), *Ecology* | 1 | origem do coeficiente de Dice |
| Efron e Tibshirani (1993), *An Introduction to the Bootstrap* | 9 | intervalo de confiança por *bootstrap* |
| Elston e Ellis (1991), *Histopathology* | 1 | graduação de Nottingham e pleomorfismo nuclear |
| Fischer et al. (2008), *Cold Spring Harbor Protocols* | 1 | química da coloração H&E |
| Fonseca et al. (2025), *Journal of Mathematical Imaging and Vision* | 2 | artigo que publica o FMBS; pode ser a referência preferível no artigo |
| Iakubovskii, *Segmentation Models Pytorch* | 6 | biblioteca da MarkerUNet; conferir a forma de citação recomendada |
| Isensee et al. (2021), *Nature Methods* 18 | 8, 9 | versão publicada do nnU-Net; conferir se usa Dice somado à entropia cruzada |
| Bloch e Maître (1995), *Pattern Recognition* 28(9), doi:10.1016/0031-3203(94)00312-A | 13 | morfologia de conjuntos difusos |
| Franchi, Fehri e Yao (2020), *Pattern Recognition* 102:107246, doi:10.1016/j.patcog.2020.107246 | 13 | camadas morfológicas treináveis |
| Jaccard (1912), *New Phytologist* | 1 | origem do índice de Jaccard |
| Kingma e Ba (2015), *Adam*, ICLR | 9 | otimizador |
| Kirillov et al. (2019), *Panoptic segmentation*, CVPR | 1 | métrica PQ |
| Loshchilov e Hutter (2017), *SGDR*, ICLR | 9 | agenda de taxa em cosseno |
| Salembier e Garrido (2000), IEEE TIP 9(4) | 2 | segmentação por marcadores sobre árvores de partição; citada pela tese |
| Wilcoxon (1945), *Biometrics Bulletin* | 9 | teste de postos sinalizados |

## 3. Afirmações que ainda não têm referência

| Capítulo e seção | Afirmação |
|---|---|
| 1.1 | O preparo do tecido: fixação, inclusão em parafina, corte em fatias de poucos micrômetros e coloração, porque o tecido sem coloração é quase transparente. |
| 1.1 | Imagens de lâmina inteira têm dezenas de milhares de pixels de lado. |
| 1.4 | Linfócitos têm núcleos pequenos e muito escuros. |

## 4. Referências que podem ser úteis

Ainda não citadas; entram se o texto precisar delas.

| Referência | Para quê |
|---|---|
| Gamper et al., PanNuke (2019 e 2020) | conferir a origem das imagens do PanNuke, usado no treino do ScribblePrompt (capítulo 7, ameaças à validade) |
| Bengio, Léonard e Courville (2013), arXiv:1308.3432 | estimador *straight-through*, para o marcador quase binário (capítulo 13); não confundir com arXiv:1305.2982 |
| Graham et al. (2019), HoVer-Net, arXiv:1812.06499 | segmentação de núcleos por instância em H&E, para trabalhos relacionados e para a extensão por instância (capítulo 13) |
| Macenko et al. (2009), ISBI | normalização de cor em histologia, se o texto discutir a variação de coloração (capítulo 1) |
| Sternberg (1986), CVGIP 35(3):333-355, doi:10.1016/0734-189X(86)90004-6 | morfologia em tons de cinza, referência clássica para o fechamento (capítulo 4) |
| Soille, *Morphological Image Analysis*, 2ª ed., Springer, doi:10.1007/978-3-662-05088-0 | livro-texto de morfologia matemática, para definir o fechamento no texto (capítulo 4) |
| Xu et al. (2016), *Deep interactive object selection*, CVPR, doi:10.1109/CVPR.2016.47; Sofiiuk et al. (2021), arXiv:2102.06583 | protocolo de cliques simulados citado pelo ScribblePrompt, para o segundo artigo (capítulo 13) |
