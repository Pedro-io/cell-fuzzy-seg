# 2. O problema e a pergunta do trabalho

Este capítulo explica de onde o trabalho parte: a segmentação interativa por marcadores, a dificuldade de aprender
marcadores, a solução proposta na tese que serve de base ao projeto e o que este trabalho muda nela. Termina com a pergunta
que o trabalho quer responder, o critério para dizer que ela foi respondida e as contribuições pretendidas.

A base teórica é a tese de doutorado de Gabriel Barbosa da Fonseca, orientador deste trabalho (Fonseca, 2026). As seções da
tese citadas abaixo foram conferidas no PDF publicado no HAL.

## 2.1 Segmentação interativa por marcadores

A segmentação é, por definição, um problema mal posto: numa imagem com vários objetos, nada na imagem diz qual deles
interessa a um usuário específico (Fonseca, 2026, seção 1.1). Por isso, em muitos casos a segmentação ainda depende de
alguma indicação humana.

Uma forma comum de dar essa indicação são os **marcadores** (ou sementes, *seeds*): rabiscos (*scribbles*), cliques,
caixas ou regiões que dizem onde está o objeto e onde está o fundo. Um algoritmo propaga essa informação para o resto da
imagem e devolve a segmentação. Há métodos clássicos, como o watershed por marcadores, os cortes em grafos (*graph cuts*) e
o *random walker*, e métodos baseados em redes neurais que recebem os marcadores como entrada (Fonseca, 2026, seções 1.1 e
2.1). Num processo interativo, o usuário olha o resultado, corrige os marcadores e repete até ficar satisfeito.

O custo desse processo cresce com o número de objetos. A tese dá como exemplo imagens de células com centenas de casos
positivos por imagem, em que anotar tudo à mão é muito trabalhoso (Fonseca, 2026, seção 1.1 e figura 1.1; Costa et al.,
2023). As imagens do MoNuSeg estão nessa situação, com centenas de núcleos cada (capítulo 1, seção 1.3). Reduzir o esforço
do usuário na segmentação interativa é, portanto, um objetivo com valor prático.

## 2.2 Por que não se aprende um marcador de forma supervisionada

A ideia natural seria treinar uma rede para propor os marcadores. O obstáculo é que **não existe o marcador correto** de uma
imagem. Pessoas diferentes marcam de jeitos diferentes, e muitos marcadores distintos levam à mesma boa segmentação. Por
isso não há conjuntos de dados com gabarito de marcador, e o aprendizado de marcadores não pode ser formulado como um
problema supervisionado comum (Fonseca, 2026, seção 1.2, questão 2).

A tese contorna o problema definindo o bom marcador pelo que ele faz, e não pela sua forma exata. Um bom marcador
(Fonseca, 2026, seção 1.2, questão 1):

1. **recupera a segmentação correta** quando é entregue a um método de segmentação por marcadores;
2. **é fácil de editar** por um usuário, que pode aceitá-lo ou corrigi-lo.

## 2.3 A proposta da tese: aprender marcadores através de uma camada de segmentação

A solução da tese é não supervisionar o marcador diretamente, e sim o resultado que ele produz. Uma rede
$f(X;\theta)$ recebe a imagem e propõe um marcador difuso $\mu = f(X;\theta)$, com valores em $[0,1]$. Uma camada de
segmentação $S$ transforma o marcador numa segmentação, e a perda compara essa segmentação com o ground truth
$\hat{S}$. A perda total é (Fonseca, 2026, equação 5.1):

$$
L_{marker} = L_{seg}\big(S(f(X;\theta)),\ \hat{S}\big) + \sum_i \lambda_i\, L_i\big(f(X;\theta)\big)
$$

Cada parte cuida de uma das duas propriedades da seção 2.2:

- $L_{seg}$, uma perda de segmentação (a tese usa o Dice suave), garante que o marcador recupera a segmentação correta;
- os termos $L_i$ regularizam a forma do marcador para que ele se pareça com o que um usuário desenharia: pequeno
  ($L_{size}$), longe da borda do objeto ($L_{Dmap}$), com poucos componentes conexos ($L_{comp}$) e suave ($L_{TV}$)
  (Fonseca, 2026, seção 5.3). As fórmulas e o que cada termo faz estão no capítulo 8.

Para que o gradiente de $L_{seg}$ chegue à rede de marcadores, a camada $S$ precisa ser diferenciável em relação ao
marcador. A contribuição central da tese é essa camada: o **FMBS** (*Fuzzy Marker-Based Segmentation*). Ele estende um método
clássico de segmentação por marcadores sobre hierarquias de partições (Salembier e Garrido, 2000), em que toda região da
hierarquia que toca o marcador de objeto e não toca o marcador de fundo é classificada como objeto. O FMBS aceita marcadores
difusos em vez de binários e é subdiferenciável em relação a eles (Fonseca, 2026, capítulo 4 e seção 5.2).

O nome do repositório vem daqui: o "fuzzy" se refere aos **marcadores difusos**, com valores contínuos em $[0,1]$. Não há
outra lógica difusa no projeto.

A tese apresenta três arquiteturas (Fonseca, 2026, seção 5.4):

- uma **prova de conceito**, em que uma U-Net recebe a imagem RGB e o próprio ground truth como indicação do objeto, numa
  entrada de 4 canais, e propõe só o marcador de objeto;
- um **modelo de marcador de objeto**, em duas etapas: uma rede de segmentação interativa transforma indicações mínimas do
  usuário (as *cues*) numa segmentação intermediária, que não precisa ser precisa, e a rede de marcadores transforma essa
  segmentação, junto com a imagem e uma representação da hierarquia, no marcador;
- um **modelo de marcador de fundo**, que usa a mesma rede com o complemento da segmentação intermediária para propor
  também o marcador de fundo.

Os experimentos da tese usam imagens naturais (MSRA10K, MS COCO e conjuntos não vistos no treino), sempre com **um objeto de
interesse por imagem**, escolhido pelas *cues* (Fonseca, 2026, seções 3.3 e 6.1).

## 2.4 O cenário deste trabalho: uma classe de objeto conhecida

Nos trabalhos futuros, a tese propõe adaptar o método a casos em que o objeto de interesse é conhecido de antemão, como a
delimitação de tipos celulares em microscopia. Nesses casos, a indicação do usuário pode ser desnecessária, porque o modelo
pode ser otimizado diretamente para a classe de interesse. A tese observa ainda que isso permitiria comparar o método com
modelos de segmentação totalmente supervisionados (Fonseca, 2026, seção 7.1.2).

Este trabalho se encaixa nesse cenário. O objeto de interesse é sempre o mesmo, os núcleos, e não há indicação do usuário.
O papel da segmentação intermediária do modelo de marcador de objeto da tese é feito por um segmentador automático de
células, o Cellpose-SAM, rodado uma vez sobre cada imagem (capítulo 5). A partir dela, uma rede de marcadores (a MarkerUNet,
capítulo 6) propõe o marcador, e uma camada de segmentação diferenciável produz a segmentação final.

Duas diferenças de cenário pesam no desenho do método:

- **Muitos objetos por imagem.** Na tese o objeto é um só. Aqui o "objeto" é a união de centenas de núcleos (capítulo 1), e
  a segmentação é semântica, núcleo contra fundo. Um bom marcador tem, então, centenas de componentes conexos, um por núcleo.
  O $L_{comp}$ da tese foi usado com o número alvo de máximos igual a 1, ou seja, empurrando o marcador para um único
  componente (Fonseca, 2026, seção 5.3). Nesse cenário ele puxaria na direção errada, o que se soma às razões para não
  usá-lo (capítulo 12).
- **A editabilidade não é o objetivo.** O marcador vai para uma rede, não para um usuário. As regularizações de forma deixam
  de ser um requisito e passam a ser uma escolha a testar: elas só ficam se ajudarem a segmentação (capítulo 8).

A tabela resume o que muda em relação ao modelo de marcador de objeto da tese. Os detalhes de cada linha estão nos
capítulos indicados.

| | Tese (modelo de marcador de objeto) | Este trabalho |
|---|---|---|
| Objeto de interesse | um por imagem, indicado pelo usuário | todos os núcleos da imagem; classe conhecida, sem indicação |
| Segmentação intermediária | rede de segmentação interativa treinada, a partir das *cues* | Cellpose-SAM, congelado e calculado uma vez (cap. 5) |
| Rede de marcadores | U-Net com três codificadores ResNet34 (imagem; segmentação de objeto e fundo; hierarquia) | U-Net com um codificador ResNet34 e entrada de 4 canais, RGB mais a máscara do Cellpose, como na prova de conceito (cap. 6) |
| Camada de segmentação | FMBS sobre uma hierarquia, sem pesos | ScribblePrompt-UNet, rede neural pré-treinada e congelada (cap. 7) |
| Marcador de fundo | fixo ou proposto pela mesma rede, por inversão | nenhum na configuração base; como opção, um canal de fundo fixo tirado da segmentação do Cellpose (cap. 7) |
| Regularizações | $L_{size}$, $L_{Dmap}$, $L_{comp}$, $L_{TV}$ | $L_{size}$, $L_{Dmap}$ e $L_{TV}$ disponíveis; $L_{comp}$ não implementado (cap. 8 e 12) |
| Dados | imagens naturais (MSRA10K, MS COCO e outros) | MoNuSeg, 37 imagens de treino e 14 de teste (cap. 3) |
| Avaliação | IoU e métricas sobre os marcadores | Dice e IoU por imagem, comparados de forma pareada com o Cellpose (cap. 1 e 9) |

Fontes da coluna da tese: Fonseca (2026), seções 5.3, 5.4.2, 5.4.3, 6.1 e 6.2. Fontes da coluna deste trabalho:
[marker_unet_config.py](../../src/models/configs/marker_unet_config.py),
[scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py),
[preprocessed_dataset.py](../../src/data/load/preprocessed_dataset.py) (canal de fundo do Cellpose) e
[terms.py](../../src/losses/terms.py).

## 2.5 A pergunta do trabalho

> O mesmo tipo de pipeline da tese, com uma segmentação intermediária, uma rede de marcadores e uma camada de segmentação
> diferenciável treinados de ponta a ponta, funciona **sem o FMBS**, usando outra camada de segmentação no lugar? E, sem
> ele, pode funcionar melhor?

A camada escolhida é o **ScribblePrompt-UNet** (Wong et al., 2024), uma rede de segmentação interativa feita para imagens
biomédicas. Ela recebe a imagem e indicações do usuário (rabiscos, cliques ou caixas) e devolve a segmentação. Os pesos
dela ficam congelados, mas, como é uma rede neural, o gradiente passa por ela até a rede de marcadores. Dois pontos a
favor da escolha estão no artigo. Primeiro, o treino usou 65 conjuntos de dados biomédicos, entre eles dois de segmentação
de núcleos: o BBBC038, do *Data Science Bowl* de 2018, e o PanNuke, de histologia (Wong et al., 2024, seção 4 e tabela de
conjuntos no apêndice). Segundo, o MoNuSeg não está nessa lista, então os dados deste trabalho não foram vistos no treino
dela. Por que esta rede e não outras camadas está no capítulo 12.

Trocar o FMBS por uma rede muda a natureza do marcador. No FMBS, a segmentação é inteiramente determinada pelos marcadores e
pela hierarquia, e é isso que torna o marcador uma explicação da segmentação e permite editá-lo de forma previsível
(Fonseca, 2026, seção 1.1, *thesis statement*). O ScribblePrompt aprendeu com dados o que fazer com os marcadores: ele pode
segmentar regiões que o marcador não cobre. Além disso, foi treinado com interações simuladas e esparsas: cliques
codificados como máscaras binárias e rabiscos como máscaras com valores em $[0,1]$, não com mapas densos sobre a imagem
inteira (Wong et al., 2024, seção 3.2 e apêndice B.2). O marcador deixa, então, de ser uma explicação exata da segmentação. Essa é uma limitação conceitual da
troca, discutida no capítulo 13.

**Critério de sucesso.** A rede de marcadores recebe a segmentação do Cellpose como entrada. O pipeline só se justifica se a
segmentação final for **melhor que a do próprio Cellpose**. A linha de base de todo resultado é, portanto, o Cellpose
sozinho, comparado imagem a imagem com o pipeline nas mesmas imagens (capítulo 9). Até 2026-10-17, nenhuma configuração
treinada superou essa linha de base (capítulo 11).

## 2.6 Contribuições pretendidas

As contribuições se dividem entre as que valem qualquer que seja o resultado final e as que dependem dele.

Valem em qualquer caso:

1. **A adaptação do aprendizado de marcadores a uma classe conhecida**, sem indicação do usuário, usando a segmentação de
   um modelo generalista como entrada da rede de marcadores. É o cenário que a tese deixou como trabalho futuro (seção 7.1.2).
2. **A troca do FMBS por uma rede interativa pré-treinada e congelada** como camada de segmentação diferenciável, com a
   análise do que isso muda no papel do marcador (seção 2.5).
3. **Os oráculos:** medidas de até onde a camada de segmentação consegue chegar quando recebe marcadores ideais, tirados do
   ground truth, sem nenhum treino. Eles separam o que é limite da camada do que é limite do treino da rede de marcadores
   (capítulo 10).
4. **Um protocolo de avaliação contra a própria entrada:** validação cruzada nas imagens de treino para escolher
   configurações, conjunto de teste usado uma vez e comparação pareada com o Cellpose em todas as imagens (capítulo 9).

Dependem do resultado:

5. **Um ganho sobre o Cellpose.** Hipótese: um marcador aprendido, entregue a uma camada de segmentação interativa, corrige
   erros da segmentação inicial. Se o ganho não aparecer, a contribuição passa a ser a explicação de por que ele não aparece,
   com base nos oráculos e no diagnóstico dos marcadores aprendidos (capítulos 10 e 11).

## Para o artigo

- **Introdução:** seções 2.1 e 2.2 (o custo da segmentação interativa e por que os marcadores não se aprendem de forma
  supervisionada), a pergunta da seção 2.5 e as contribuições da seção 2.6.
- **Trabalhos relacionados:** seção 2.3 (a tese e o FMBS), o ScribblePrompt e os métodos clássicos de segmentação por
  marcadores. Precisa de referências próprias, além das que a tese cita.
- **Métodos:** a tabela da seção 2.4, convertida num parágrafo de posicionamento em relação à tese.
- **Discussão e limitações:** a perda da editabilidade e da explicação exata ao trocar o FMBS por uma rede (seção 2.5) e o
  $L_{comp}$ inadequado para muitos objetos (seção 2.4).
- **Enquadramento:** a seção 7.1.2 da tese como motivação do trabalho. Ainda não foi discutida com o orientador.

## Referências

- Costa, G.; Rodrigues, A.; Fonseca, G.; Patrocínio Jr., Z.; Souto, G.; Guimarães, S. (2023). Single-shot object detection
  and supervised image segmentation for analysing cell images obtained by immunohistochemistry. *Anais Estendidos da XXXVI
  Conference on Graphics, Patterns and Images (SIBGRAPI)*, 121-125. *A conferir* (citado pela tese como referência 42).
- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. <https://theses.hal.science/tel-05743320>. *Conferida* (seções 1.1, 1.2, 1.3, 3.3, 5.1 a 5.4,
  6.1, 6.2 e 7.1.2).
- Fonseca, G. B. da; Negrel, R.; Perret, B.; Cousty, J.; Guimarães, S. J. F. (2025). Hierarchy-based fuzzy segmentation and
  marker learning layer: theory and algorithms. *Journal of Mathematical Imaging and Vision*. *A conferir* (artigo que publica
  parte da tese; pode ser a referência preferível para o FMBS no artigo).
- Salembier, P.; Garrido, L. (2000). Binary partition tree as an efficient representation for image processing,
  segmentation, and information retrieval. *IEEE Transactions on Image Processing*, 9(4), 561-576. *A conferir* (citado pela
  tese como referência 118).
- Wong, H. E.; Rakic, M.; Guttag, J.; Dalca, A. V. (2024). ScribblePrompt: fast and flexible interactive segmentation for
  any biomedical image. arXiv:2312.07381v3. *Conferida* (resumo, seção 4 e tabela de conjuntos de dados; o PDF está em
  `docs/papers/scribble_prompt.pdf`). Falta conferir o local de publicação definitivo.
