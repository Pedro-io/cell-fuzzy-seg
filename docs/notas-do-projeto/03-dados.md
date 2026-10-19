# 3. Dados

Este capítulo descreve o conjunto de dados usado em todo o trabalho, o MoNuSeg: de onde vem, como as anotações viram
máscaras, o que foi medido nele, como treino e teste são separados e como os dados são preparados uma única vez antes dos
experimentos. Os conceitos de domínio (H&E, aumento, segmentação semântica) estão no capítulo 1.

## 3.1 O MoNuSeg

O MoNuSeg (*Multi-Organ Nucleus Segmentation*) foi o desafio de segmentação de núcleos do MICCAI 2018, com 32 equipes
participantes (Kumar et al., 2020). O conjunto de treino do desafio é o conjunto publicado por Kumar et al. (2017):

- **origem:** imagens de lâmina inteira do arquivo TCGA (*The Cancer Genome Atlas*), digitalizadas em 40× e coradas com H&E,
  de 18 hospitais diferentes, o que traz variação de coloração entre laboratórios;
- **recortes:** um recorte de 1000 × 1000 pixels por imagem de lâmina inteira e por paciente, tirado de uma região densa em
  núcleos;
- **anotação:** contornos desenhados no software Aperio ImageScope, com a imagem ampliada para que cada pixel ocupasse 5 × 5
  pixels da tela. Os anotadores eram estudantes de engenharia treinados pelos autores, e um patologista revisou as anotações.
  Foram anotados núcleos epiteliais e do estroma, distribuídos em arquivos XML;
- **órgãos:** tecido benigno e doente de sete órgãos no treino; o teste tem dois órgãos que não aparecem no treino, pulmão e
  cérebro (Kumar et al., 2020, tabela II):

| | Mama | Fígado | Rim | Próstata | Bexiga | Cólon | Estômago | Pulmão | Cérebro | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Treino do desafio | 6 | 6 | 6 | 6 | 2 | 2 | 2 | | | 30 |
| Teste | 2 | | 3 | 2 | 2 | 1 | | 2 | 2 | 14 |

A licença é a CC BY-NC-SA 4.0 (atribuição, uso não comercial, mesma licença) (página de dados do desafio, consultada em
2026-10-17).

**A versão usada.** Os artigos descrevem 30 imagens de treino, com 21.623 núcleos, e 14 de teste, com cerca de 7.223 núcleos
(Kumar et al., 2020). O pacote de treino baixado da página do desafio em 2026-09-29 traz **37 imagens** e 24.140 contornos;
as 30 imagens que coincidem com uma cópia anterior são idênticas a ela. O teste traz as 14 imagens e 6.697 contornos
([data_source/README.md](../../data_source/README.md); [08-pendencias.md, PD-23](../estudo/08-pendencias.md#pd-23)). A
descrição do MoNuSeg no artigo do Cellpose-SAM também fala em 37 imagens de treino e 14 de teste (Pachitariu et al., 2025).
O trabalho usa as 37 + 14 imagens. Os arquivos estão sem alteração de conteúdo (conferido por SHA-256); só a organização das
pastas foi ajustada.

Duas diferenças entre os artigos e os arquivos ficam sem explicação: os órgãos das 7 imagens de treino a mais não estão nos
artigos, e o teste tem 6.697 contornos nos XMLs, contra cerca de 7.223 núcleos no artigo do desafio.

Cada imagem tem 1000 × 1000 pixels RGB e vem de um paciente diferente. Os 51 identificadores de paciente do TCGA (os doze
primeiros caracteres do nome do arquivo) são todos distintos, e nenhum paciente aparece no treino e no teste ao mesmo tempo
(conferido em 2026-10-17 nos nomes dos arquivos).

**Citações exigidas.** Quem usa os dados deve citar Kumar et al. (2017) e Kumar et al. (2020).

## 3.2 Arquivos e anotações

```
data_source/
  MoNuSegTrainingData/
    Tissue_Images/*.tif     37 imagens RGB 1000 × 1000 (TIFF com compressão LZW)
    Annotations/*.xml       37 arquivos de contornos
  MoNuSegTestData/
    Tissue_Images/*.tif     14 imagens
    Annotations/*.xml       14 arquivos de contornos
  MoNuSegPreprocessed/      dados derivados (seção 3.6)
```

O identificador de cada imagem é o nome do arquivo sem extensão (por exemplo, `TCGA-18-5592-01Z-00-DX1`). Ele é o mesmo no
`.tif`, no `.xml` e nos arquivos derivados.

Cada XML segue a hierarquia `Annotations > Annotation > Regions > Region > Vertices > Vertex(X, Y)`, em que cada `Region` é o
contorno de um núcleo, com coordenadas fracionárias em pixels. O cabeçalho traz `MicronsPerPixel`, a escala física da
imagem. É o formato do Aperio ImageScope, o software usado na anotação (Kumar et al., 2017).

## 3.3 Do contorno à máscara

As anotações são polígonos, e a rede trabalha com pixels. A conversão é feita por
[xml_to_instance_mask](../../src/data/load/monuseg_dataset.py) com quatro regras:

1. **Um pixel pertence ao núcleo se o seu centro cai dentro do polígono.** As coordenadas do XML seguem a convenção de canto:
   `X = 0` é a borda esquerda do pixel 0, e o centro do pixel `(linha r, coluna c)` fica em `(c + 0,5, r + 0,5)`.
2. **Regiões degeneradas são descartadas:** contornos com menos de 3 vértices ou área zero, que correspondem a cliques soltos
   na anotação.
3. **Na sobreposição, o menor núcleo vence.** Os núcleos são pintados do maior para o menor, de modo que nenhum núcleo pequeno
   desaparece sob um vizinho maior.
4. **Rótulos 1 a N** na ordem do XML, um por núcleo; 0 é o fundo.

O resultado é uma máscara por instância. O ground truth binário usado no treino e na avaliação é a versão binária dela
(rótulo maior que zero), como explicado no capítulo 1, seção 1.5.

**Diferença em relação aos autores do MoNuSeg.** Na sobreposição, os autores atribuíram cada pixel ao **maior** núcleo que o
contém (Kumar et al., 2017, seção III); a regra 3 acima faz o contrário. A escolha só afeta a máscara por instância, que é
usada no mapa de distância (capítulo 8); o GT binário é o mesmo nas duas regras. Com a regra do menor, nenhum núcleo some por
inteiro ([08-pendencias.md, PD-07](../estudo/08-pendencias.md#pd-07)). A regra do maior segue a convenção dos autores, mas pode
apagar núcleos pequenos cobertos por um maior; isso não foi medido. A escolha está em aberto (capítulo 12).

**Por que o centro do pixel.** Pintar todo pixel tocado pela borda do polígono, o que faz o preenchimento de polígonos do
OpenCV (`cv2.fillPoly`), deixa a máscara cerca de 10% maior que a área anotada. Com a regra do centro, a área rasterizada fica
a 0,995 (treino) e 1,011 (teste) da área anotada, e a máscara fica mais alinhada com a segmentação do Cellpose
([09-correcoes-pontuais.md §13](../estudo/09-correcoes-pontuais.md)). Hipótese: o fato de essa convenção acertar a área
indica que é a convenção usada na anotação; nenhum documento do desafio confirma isso.

Com essas regras, das 24.140 regiões do treino, 24.135 são válidas e 24.133 ocupam pelo menos um pixel; no teste, as 6.697
são válidas e ocupam pixels (mesma fonte). Os polígonos se sobrepõem pouco: 1,00% dos pixels de núcleo do treino e 0,25% do
teste são cobertos por mais de um polígono ([08-pendencias.md, PD-07](../estudo/08-pendencias.md#pd-07)).

## 3.4 O que foi medido

Medido em 2026-10-17 sobre os XMLs e as máscaras de `data_source/MoNuSegPreprocessed/`. "Diâmetro" é o diâmetro do círculo
de mesma área que o contorno anotado.

| | Treino | Teste |
|---|---|---|
| Imagens | 37 | 14 |
| Núcleos (regiões válidas) | 24.135 | 6.697 |
| Núcleos por imagem: mínimo, mediana, máximo | 294, 440, 1.862 | 249, 466, 818 |
| Fração de pixels de núcleo: média (mínimo a máximo) | 24,4% (10,6% a 40,0%) | 20,0% (9,4% a 28,0%) |
| Componentes conexos no GT binário (vizinhança 8) | 19.326 (20% a menos que os núcleos) | 6.221 (7% a menos) |
| Instâncias encontradas pelo Cellpose | 22.447 (93% dos núcleos) | 7.327 (109%) |
| Fração de pixels marcada pelo Cellpose (média) | 23,2% | 20,7% |
| Dice do Cellpose contra o GT (média por imagem; desvio; pior imagem) | 0,845; 0,030; 0,768 | 0,837; 0,033; 0,755 |

**Escala.** Os núcleos têm tamanhos em pixels muito diferentes conforme o aumento da imagem:

| Grupo (pelo `MicronsPerPixel` do XML) | Imagens | Núcleos | Área mediana | Diâmetro mediano | Diâmetro, 5% a 95% |
|---|---|---|---|---|---|
| Treino, 40× (cerca de 0,25 µm/px) | 30 | 16.332 | 349 px | 21,1 px | 13,0 a 36,2 px |
| Treino, 20× (0,5005 µm/px) | 3 | 4.521 | 117 px | 12,2 px | 7,8 a 19,5 px |
| Treino, sem escala no XML | 4 | 3.282 | 294 px | 19,4 px | 11,2 a 36,6 px |
| Teste, 40× | 14 | 6.697 | 369 px | 21,7 px | 13,3 a 32,4 px |

As três imagens em 20× são `TCGA-HE-7128`, `-7129` e `-7130`. As quatro sem escala são `TCGA-AY-A8YK`, `TCGA-KB-A93J`,
`TCGA-NH-A8F7` e `TCGA-RD-A8N9`; pelo tamanho dos núcleos, parecem estar em 40× (hipótese). O teste não tem nenhuma imagem em
20×.

O que essas medidas dizem sobre a tarefa:

- **Treino e teste têm perfis diferentes.** O treino tem imagens mais densas (até 1.862 núcleos), as três imagens em 20× e mais
  fusão de núcleos no GT binário. As três imagens em 20× respondem sozinhas por 27% dos componentes perdidos na fusão
  (capítulo 1, seção 1.5). O teste tem dois órgãos que o treino não tem, pulmão e cérebro (seção 3.1).
- **O Cellpose é uma linha de base forte.** Ele já acerta Dice 0,845 no treino e 0,837 no teste, com pouca variação entre
  imagens. Qualquer ganho do pipeline terá de ser medido sobre esse patamar (capítulo 2, seção 2.5). O Cellpose-SAM foi
  treinado com as imagens de treino do MoNuSeg, e não com as de teste; o número do treino é, portanto, de imagens que ele já
  viu (capítulo 5, seção 5.4).
- **O fundo é a maior parte da imagem,** mas o desequilíbrio é moderado: cerca de um quarto dos pixels é núcleo.

## 3.5 Treino, validação e teste

A divisão oficial é mantida: as 37 imagens de treino são usadas para treinar e escolher configurações, e as 14 de teste só
para a avaliação final. Como o conjunto de treino é pequeno, a escolha de configurações é feita por validação cruzada em
*folds* dentro das 37 imagens, sempre dividindo por imagem, e o teste é usado uma única vez, com a configuração escolhida. O
protocolo completo está no capítulo 9.

## 3.6 Preparação dos dados

O que não depende do treino é calculado uma vez e salvo em disco. O notebook
`notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb`, que roda no Colab com GPU, lê as 51 imagens com o
`MonusegDataset`, confere que todas têm 1000 × 1000 pixels (nada é redimensionado) e aplica, em ordem, quatro etapas
(detalhes no apêndice A):

1. **Cellpose-SAM:** segmenta os núcleos de cada imagem (capítulo 5);
2. **RGBA:** junta a imagem RGB, em $[0,1]$, e a máscara binária do Cellpose como quarto canal, que é a entrada da rede de
   marcadores (capítulo 6);
3. **Mapa de distância:** calculado por núcleo, a partir da máscara por instância, para uma das perdas (capítulo 8);
4. **Gravação:** um arquivo `.npy` por chave e por imagem, em `data_source/MoNuSegPreprocessed/{train,test}/<chave>/<id>.npy`,
   mais um `meta.json`.

| Chave | Formato | Conteúdo |
|---|---|---|
| `image` | 1000 × 1000 × 3, uint8 | a imagem original |
| `ground_truth` | 1000 × 1000, uint8 | GT binário (0 ou 1) |
| `segmentation` | 1000 × 1000 | instâncias do Cellpose (0 = fundo, 1 a K = núcleos) |
| `cellpose_prob` | 1000 × 1000, float16 | probabilidade de célula do Cellpose |
| `rgba` | 1000 × 1000 × 4, float32 | RGB em $[0,1]$ e a máscara do Cellpose |
| `distance_map` | 1000 × 1000, float32 | mapa de distância por núcleo, em $[0,1]$ |

A máscara por instância do GT não é salva; ela só é usada para calcular o mapa de distância. A versão atual dos dados
derivados foi gerada em 2026-10-07 (`meta.json`). Se o Cellpose não carregar, o notebook para com erro em vez de seguir com
outra entrada.

**Leitura nos experimentos.** A função [load_preprocessed](../../src/data/load/preprocessed_dataset.py) lê os `.npy` de um split
uma vez. Ela renomeia a segmentação do Cellpose para `cellpose_segmentation`, porque a chave `segmentation` é usada depois
pela saída da camada de segmentação (apêndice A). Duas variantes da entrada são montadas na leitura, sem rodar o Cellpose de
novo: o quarto canal do RGBA pode ser a probabilidade do Cellpose em vez da máscara, e um canal de fundo pode ser derivado da
segmentação do Cellpose (capítulo 7).

**Aumentação.** A cada época, cada imagem recebe uma rotação de 0°, 90°, 180° ou 270° e espelhamentos horizontal e vertical
(probabilidade 0,5 cada), aplicados juntos a todas as chaves
([PreprocessedDataset](../../src/data/load/preprocessed_dataset.py)). Como são só rotações de 90° e espelhamentos, o mapa de
distância continua válido sem ser recalculado. Opcionalmente, o treino usa recortes em posição sorteada em vez da imagem
inteira (capítulo 9).

## Para o artigo

- **Dados (Métodos):** seção 3.1 (origem, licença, citações exigidas), a versão usada (37 + 14, com a diferença em relação
  às 30 imagens da página) e a separação por paciente; seção 3.3 (a regra de rasterização, em uma frase); a primeira tabela
  da seção 3.4, em versão resumida.
- **Detalhe que vale uma nota:** rasterizar os contornos com `cv2.fillPoly` aumenta a área dos núcleos em cerca de 10%
  (seção 3.3). Hipótese: outros trabalhos que convertem os XMLs dessa forma avaliam contra um GT inflado; conferir antes de
  afirmar.
- **Discussão:** treino e teste com perfis diferentes, inclusive na escala (seção 3.4).
- **Limitações:** conjunto pequeno (37 imagens de treino) e um único conjunto de dados.

## Referências

- Kumar, N.; Verma, R.; Sharma, S.; Bhargava, S.; Vahadane, A.; Sethi, A. (2017). A dataset and a technique for generalized
  nuclear segmentation for computational pathology. *IEEE Transactions on Medical Imaging*, 36(7), 1550-1560.
  doi:10.1109/TMI.2017.2677499. Citação exigida pelo desafio. *Conferida* (seção III).
- Kumar, N. et al. (2020). A multi-organ nucleus segmentation challenge. *IEEE Transactions on Medical Imaging*, 39(5),
  1380-1391. doi:10.1109/TMI.2019.2947628. Citação exigida pelo desafio. *Conferida* no manuscrito aceito (resumo, seção III,
  tabela II).
- Pachitariu, M.; Rariden, M.; Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation.
  *bioRxiv*, doi:10.1101/2025.04.28.651001. *Conferida* (descrição dos conjuntos de dados).
- MoNuSeg Grand Challenge, página de dados. <https://monuseg.grand-challenge.org/Data/>. *Conferida* em 2026-10-17.
