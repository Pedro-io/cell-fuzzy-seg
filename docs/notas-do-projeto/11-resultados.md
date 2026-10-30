# 11. Experimentos e resultados

Este capítulo reúne os experimentos de treino feitos com os dados e o protocolo atuais (capítulos 3 e 9), os testes feitos sem
retreinar com os modelos gravados, a editabilidade dos marcadores e o diagnóstico dos marcadores aprendidos. Os oráculos, que não
envolvem treino, estão no capítulo 10. Os números são da validação cruzada nas 37 imagens de treino, salvo quando indicado.

**Resultado final (experimentos encerrados em 2026-10-27).** A configuração escolhida pela validação cruzada,
`recortes_negativo_r2_fechamento`, chega a Dice 0,8402 na validação cruzada com a inferência final (sobreposição de 1/2 e TTA),
contra 0,8450 do Cellpose, e a **0,8333 no teste, contra 0,8370 do Cellpose** (−0,0037; Wilcoxon p = 0,12). O marcador fechado dá
um único componente de marcador a 89% das regiões do GT na validação cruzada e a 96% no teste. **O pipeline sem o FMBS chega ao
nível do Cellpose, com marcadores editáveis, sem superá-lo** (seção 11.7).

## 11.1 Os experimentos

Todos usam a MarkerUNet do capítulo 6, o ScribblePrompt no modo `positive` com $\tau = 0{,}5$ (capítulo 7), a otimização e a
validação cruzada do capítulo 9 e as mesmas 37 imagens e *folds*. As comparações entre experimentos são pareadas por imagem. Os
experimentos seguiram uma ordem, uma mudança de cada vez:

| Etapa | Experimento | O que muda em relação ao anterior |
|---|---|---|
| imagem inteira, 256² | `base_dice_tv` | ponto de partida: Dice + TV (0,001) |
| | `dice_size` | Size (0,1) no lugar do TV |
| | `base_dice` | só Dice |
| | `base_negativo` | `base_dice_tv` + canal de fundo do Cellpose (3 px) |
| | `negativo_sem_tv` | `base_negativo` sem o TV |
| recortes | `recortes` | `base_dice` com recortes de 256², ScribblePrompt em 128² e avaliação em blocos |
| | `recortes_negativo` | `recortes` + canal de fundo (3 px) |
| perdas sobre os recortes | `recortes_negativo_dmap` | + Dmap (0,1) |
| | `recortes_negativo_bce` | + BCE (1) |
| configuração final | `recortes_negativo_r2_fechamento` | `recortes_negativo` com faixa do canal de fundo de 2 px e fechamento de 3 px no treino e na avaliação |
| perdas da tese sobre a configuração final | `recortes_negativo_r2_fechamento_tese` | o anterior + Size (0,1) + Dmap (0,1) |

Não rodaram: `recortes_negativo_tese` e `recortes_negativo_tese_bce` (substituídos pela rodada única da etapa atual) e
`recortes_negativo_bce_dmap` (interrompido por falta de memória na GPU); também `dice_dmap`, `tese` e `rmse`, da etapa da imagem
inteira. Os três primeiros experimentos da tabela estão em `notebooks/experiments/legado/`, porque avaliaram também o teste
(capítulo 9, seção 9.9).

Resultados em `docs/estudo/resultados/<experimento>/`. A pasta do `recortes_negativo_r2_fechamento` ainda não está no repositório;
os números dele vêm das saídas do notebook, registradas em [09-correcoes-pontuais.md §18.26](../estudo/09-correcoes-pontuais.md).
O `recortes_negativo_r2_fechamento_tese` rodou em 2026-10-26 (commit `5fabf43`).

## 11.2 Resultados

Cellpose nas mesmas imagens: Dice 0,845, precisão 0,867, revocação 0,826, massa 0,96.

| Experimento | Dice | Diferença para o Cellpose (IC 95%) | Imagens acima do Cellpose | Precisão | Revocação | Massa | Dice nas 3 imagens em 20× |
|---|---|---|---|---|---|---|---|
| `base_dice_tv` | 0,775 | −0,070 (−0,081 a −0,060) | 0 | 0,727 | 0,838 | 1,17 | 0,709 |
| `dice_size` | 0,774 | −0,071 (−0,084 a −0,059) | 0 | 0,753 | 0,807 | 1,09 | 0,663 |
| `base_dice` | 0,785 | −0,060 (−0,072 a −0,049) | 0 | 0,758 | 0,824 | 1,10 | 0,691 |
| `base_negativo` | 0,808 | −0,038 (−0,045 a −0,030) | 2 | 0,750 | 0,879 | 1,18 | 0,752 |
| `negativo_sem_tv` | 0,819 | −0,026 (−0,032 a −0,020) | 2 | 0,777 | 0,869 | 1,12 | 0,771 |
| `recortes` | 0,831 | −0,014 (−0,018 a −0,010) | 1 | 0,822 | 0,844 | 1,03 | 0,805 |
| `recortes_negativo` | 0,835 | −0,010 (−0,013 a −0,007) | 7 | 0,812 | 0,864 | 1,07 | 0,810 |
| `recortes_negativo_dmap` | 0,830 | −0,015 (−0,018 a −0,012) | 2 | 0,807 | 0,858 | 1,07 | 0,805 |
| `recortes_negativo_bce` | 0,834 | −0,012 (−0,015 a −0,009) | 6 | 0,817 | 0,854 | 1,05 | 0,809 |
| `recortes_negativo_r2_fechamento` | **0,838** | −0,007 (−0,010 a −0,004) | 8 | 0,816 | 0,865 | 1,07 | 0,821 |
| o mesmo, inferência final (sobreposição 1/2 e TTA) | **0,840** | −0,005 | 10 | 0,818 | 0,867 | 1,06 | 0,822 |
| `recortes_negativo_r2_fechamento_tese` | 0,836 | −0,009 (−0,011 a −0,007) | 4 | 0,828 | 0,846 | 1,02 | 0,818 |
| o mesmo, inferência final | 0,838 | −0,007 | 8 | 0,829 | 0,851 | 1,03 | 0,820 |

Em todas as linhas, o teste de Wilcoxon contra o Cellpose dá $p \le 0{,}002$: a diferença é pequena nas últimas linhas, mas
consistente. Intervalos por *bootstrap* (capítulo 9, seção 9.5).

**Comparações pareadas entre experimentos** (diferença de Dice; imagens em que o primeiro é melhor e pior):

| Comparação | Efeito de | Diferença | Melhor / pior | p |
|---|---|---|---|---|
| `base_dice` menos `base_dice_tv` | tirar o TV | +0,011 | 29 / 8 | 0,002 |
| `dice_size` menos `base_dice_tv` | trocar TV por Size | −0,000 | 21 / 16 | 0,96 |
| `base_negativo` menos `base_dice_tv` | canal de fundo | +0,033 | 34 / 3 | $7 \times 10^{-9}$ |
| `negativo_sem_tv` menos `base_negativo` | tirar o TV com canal de fundo | +0,012 | 36 / 1 | $3 \times 10^{-11}$ |
| `recortes` menos `base_dice` | recortes | +0,046 | 37 / 0 | $2 \times 10^{-11}$ |
| `recortes_negativo` menos `recortes` | canal de fundo, com recortes | +0,004 | 24 / 13 | 0,017 |
| `recortes_negativo_dmap` menos `recortes_negativo` | Dmap | −0,005 | 2 / 35 | $< 0{,}001$ |
| `recortes_negativo_bce` menos `recortes_negativo` | BCE | −0,002 | 11 / 26 | 0,008 |

Fontes: os `folds.csv` de cada pasta e [08-pendencias.md, PD-52](../estudo/08-pendencias.md#pd-52). As comparações pareadas do
`recortes_negativo_r2_fechamento` com o `recortes_negativo` e com o `recortes_negativo_r2_fechamento_tese` dependem da pasta de
resultados dele, que ainda falta; sem ela, a diferença entre os dois últimos (−0,0025 no registro, −0,0022 na inferência final) é
só entre médias.

**Teste dos experimentos antigos.** Os três experimentos da estrutura antiga também avaliaram o teste: `base_dice_tv` 0,777,
`dice_size` 0,789 e `base_negativo` 0,803, contra 0,837 do Cellpose. O teste da configuração final está na seção 11.7.

## 11.3 Testes sem retreinar

Com os modelos dos *folds* gravados, mudanças fora da MarkerUNet foram testadas sem retreinar, cada imagem prevista pelo modelo do
*fold* em que ficou de fora. Base: os modelos do `recortes_negativo` (0,8350)
([09-correcoes-pontuais.md §18.22 e §18.23](../estudo/09-correcoes-pontuais.md)):

| Mudança | Dice | Diferença | Melhor / pior |
|---|---|---|---|
| faixa do canal de fundo 2 px (treinado com 3) | 0,8394 | +0,004 | 28 / 9 |
| sobreposição dos blocos de 1/2 | 0,8355 | +0,0005 | 32 / 5 |
| TTA, 8 rotações e espelhamentos | 0,8383 | +0,003 | 36 / 1 |
| as três juntas | **0,8423** | +0,007 | 33 / 4 |
| faixa 1 px / 5 px | 0,8370 / 0,8273 | +0,002 / −0,008 | 21 / 16; 0 / 37 |
| $\tau$ 0,4 / 0,6 | 0,8228 / 0,8340 | −0,012 / −0,001 | 5 / 32; 13 / 24 |
| entrada do ScribblePrompt 160² / 192² | 0,8310 / 0,8336 | −0,004 / −0,001 | 9 / 28; 17 / 20 |
| 2 passadas do ScribblePrompt | 0,8330 | −0,002 | 1 / 36 |
| fundo também entre núcleos vizinhos (1 px) | 0,8323 | −0,003 | 0 / 37 |

As três mudanças vencedoras foram escolhidas e medidas nas mesmas 37 imagens, então o +0,007 é otimista (capítulo 9, seção 9.9).
Retreinado com a faixa de 2 px (a configuração final), o ganho diminuiu: a melhor inferência ficou em 0,8402, e não em 0,8423.
Hipótese: parte do ganho sem retreinar vinha do descompasso entre o treino com 3 px e a inferência com 2 px, que faz o modelo
segmentar menos; retreinado, ele se reequilibra. O capítulo 7 (seções 7.4 e 7.7) discute cada variação.

## 11.4 Editabilidade do marcador

Medida com as definições do capítulo 9, seção 9.7, sobre o marcador depois do corte em $\tau$; médias por imagem, cada imagem com
cerca de 650 regiões do GT ([09-correcoes-pontuais.md §18.24 a §18.26](../estudo/09-correcoes-pontuais.md)).

**O fechamento, sem retreinar** (modelos do `recortes_negativo`, com a melhor inferência):

| Raio do fechamento | Dice | Componentes | Fragmentos | Buracos | Regiões com um componente | Regiões partidas | Componentes unindo regiões |
|---|---|---|---|---|---|---|---|
| 0 (sem fechamento) | 0,8423 | 2.746 | 2.028 | 1.052 | 18% | 80% | 21 |
| 1 px | 0,8425 | 1.544 | 878 | 377 | 46% | 52% | 23 |
| 2 px | 0,8421 | 938 | 313 | 161 | 81% | 17% | 29 |
| 3 px | 0,8416 | 800 | 231 | 18 | **90%** | 9% | 47 |
| 5 px | 0,8381 | 510 | 142 | 13 | 96% | 3% | 88 |

Até 3 px, o Dice não muda (diferenças de até 0,0006, sem significância), e o marcador passa de "partido em 80% das regiões" para
"um componente em 90%". O custo aparece nos componentes que unem regiões vizinhas (de 21 para 47 por imagem, cerca de 7% das
regiões). Com 5 px, o Dice cai (−0,004, $p = 0{,}001$) e as uniões dobram de novo; 3 px é o raio escolhido. A figura mostra que o
marcador cru tem uma textura em grade, de pontos claros e escuros alternados, que gera a maior parte dos buracos; o fechamento a
apaga. A origem dessa textura não foi verificada.

**As perdas da tese julgadas pela editabilidade** (melhor inferência, com fechamento de 3 px):

| Experimento | Dice | Área marcada | Regiões sem marcador | Regiões com um componente | Regiões partidas | Componentes unindo regiões |
|---|---|---|---|---|---|---|
| `recortes_negativo` (só Dice) | **0,8416** | 19,1% | 1,9% | **90%** | 9% | 47 |
| `recortes_negativo_dmap` | 0,8384 | 11,8% | 8,0% | 66% | 26% | 23 |
| `recortes_negativo_bce` | 0,8413 | 15,2% | 1,6% | 65% | 33% | 34 |

O Dmap deixa o marcador menor e une menos núcleos vizinhos, como a tese pede, mas o marcador fica picotado e 8% das regiões ficam
sem marcador. Hipótese: na tese, o $L_{comp}$ segura os componentes enquanto o Size e o Dmap encolhem o marcador; sem ele, encolher
quebra o marcador (capítulo 8, seção 8.3). O fechamento dentro do treino não cumpriu esse papel (abaixo).

**O fechamento dentro do treino, com e sem as perdas da tese.** As duas rodadas usam a mesma configuração (faixa de 2 px,
fechamento de 3 px no treino e na avaliação) e diferem só nas perdas; editabilidade com a inferência final
(`editabilidade.csv` de cada experimento; a do primeiro, das saídas do notebook):

| | Só Dice (`r2_fechamento`) | Dice + Size + Dmap (`r2_fechamento_tese`) |
|---|---|---|
| Dice (registro / inferência final) | **0,8383 / 0,8402** | 0,8358 / 0,8380 |
| Precisão / revocação / massa (registro) | 0,816 / 0,865 / 1,07 | 0,828 / 0,846 / 1,02 |
| Área marcada, marcador fechado | 17,9% | 9,6% |
| Regiões com um componente: cru / fechado | 54% / **89%** | 11% / 55% |
| Regiões partidas, fechado | 9% | 35% |
| Regiões sem marcador | 2,0% | 9,7% |
| Fragmentos / buracos por imagem, fechado | 259 / 32 | 495 / 32 |
| Componentes unindo regiões, fechado | 39 | 18 |
| Componentes fora do GT, fechado | 316 | 90 |

Sem as perdas da tese, o fechamento dentro do treino já limpa o marcador cru (54% das regiões com um componente, contra 18% sem o
fechamento no treino), e o marcador fechado chega a 89%. O valor médio do marcador é 0,51 dentro dos núcleos e 0,38 fora.

Com o Size e o Dmap, o marcador faz parte do que a tese pede: fica com metade da área, une menos núcleos vizinhos (18 contra 39) e
cai muito menos fora dos núcleos (90 contra 316 componentes). Mas sai muito picotado (cru: 11% das regiões com um componente) e o
fechamento de 3 px não emenda os pedaços: só 55% das regiões ficam com um componente, 35% continuam partidas e quase 10% ficam sem
marcador. Pela medida principal de editabilidade, é bem pior; o Dice também cai um pouco, trocando revocação por precisão, como o
Size já fazia (seção 11.2). Os *folds* pararam cedo, com a melhor época entre 15 e 42 (mediana 19), contra 23 a 99 (mediana 67) sem
as perdas da tese.

**Leitura.** A hipótese era que o fechamento cumprisse parte do papel do $L_{comp}$ da tese, segurando os componentes enquanto o Size
e o Dmap encolhem o marcador (capítulo 8, seção 8.3). Do jeito testado, **não cumpre**: as perdas da tese quebram o marcador em
pedaços menores do que o fechamento de 3 px consegue emendar e apagam o marcador de parte dos núcleos. Uma ressalva: o Size e o
Dmap foram calculados sobre o marcador **cru**, antes do fechamento e do corte em $\tau$ (capítulo 8, seção 8.1), então a perda não
"sabe" que os pedaços serão emendados; e o Dmap teve peso 0,1, cerca de 300 vezes o da tese em proporção. Regularizar o marcador
fechado não foi testado.

## 11.5 Diagnóstico dos marcadores aprendidos

Para entender por que o pipeline segmentava a mais, o `base_dice_tv` foi retreinado nos mesmos *folds* (Dice de validação 0,773),
e a saída foi analisada imagem por imagem ([11-oraculos.md §8](../estudo/11-oraculos.md); notebook
`notebooks/exploration/diagnostico_marcadores.ipynb`, executado em 2026-10-15; resultados só nas saídas do notebook):

- **o marcador cai nos núcleos:** 91,5% da massa do canal positivo fica dentro do GT, e 87% das regiões do GT recebem marcador;
- **o excesso é inchaço e fusão:** 61,5% dos falsos positivos ficam a até 3 px do GT, e 28% dos componentes da saída cobrem duas
  ou mais regiões do GT, contra 7% no Cellpose;
- **o valor suave do marcador faz a camada expandir:** o mesmo canal positivo binarizado segmenta de menos (massa 0,73, contra 1,18),
  como o miolo do oráculo. A hipótese é que valores fracos estão fora da distribuição de rabiscos com que o ScribblePrompt foi
  treinado;
- **o limiar de saída não resolve:** em 0,6, a massa cai para 1,02, mas o Dice fica igual (0,772); a borda da saída é borrada;
- **o canal de fundo do Cellpose ajuda mesmo sem retreinar:** 0,773 para 0,802.

Esse diagnóstico motivou os passos seguintes: o canal de fundo no treino e, depois, os recortes, que no oráculo de resolução
equilibram a expansão (capítulo 10, seção 10.3).

## 11.6 O que os resultados dizem

1. **Os recortes foram o maior ganho isolado** (+0,046, melhor em todas as 37 imagens), só mudando a resolução em que as duas redes
   veem os núcleos. A massa caiu de 1,10 para 1,03.
2. **O canal de fundo do Cellpose ajuda, mas aproxima a saída do Cellpose.** Ele foi o maior ganho na imagem inteira (+0,033) e
   ainda ajuda com recortes (+0,004). Quanto mais estreita a faixa, mais a saída copia a máscara do Cellpose (capítulo 7, seção 7.4).
3. **O TV atrapalha** em todas as configurações em que foi testado; com o canal de fundo, ele achatava o marcador (capítulo 8).
4. **As perdas da tese não melhoram o Dice neste pipeline,** e julgadas pela editabilidade também não ganharam do Dice sozinho com
   fechamento, nem sem retreinar (Dmap, BCE) nem com o fechamento dentro do treino (Size e Dmap juntos): o marcador fica menor, mas
   picotado e ausente em parte dos núcleos (seção 11.4).
5. **O fechamento entrega marcadores editáveis sem custo de segmentação.** É a contribuição mais direta para a ideia da tese de
   marcadores fáceis de editar.
6. **A perda que sobra é de precisão.** A revocação do pipeline passa a do Cellpose (0,865 contra 0,826), mas a precisão fica abaixo
   (0,816 contra 0,867): o pipeline segmenta cerca de 6% a mais que o GT, sobretudo inchando bordas e unindo núcleos densos.
7. **As imagens em 20× melhoraram, mas continuam as piores** (de 0,66 a 0,82 ao longo dos experimentos, contra 0,82 a 0,86 do
   Cellpose nelas). Como o ScribblePrompt não perde nelas com marcadores ideais em 256² (capítulo 10), a perda está na MarkerUNet ou
   na escala dos recortes.

## 11.7 Configuração final e resultado no teste

**Escolha.** Pelos dois critérios, Dice e editabilidade, a validação cruzada escolheu o `recortes_negativo_r2_fechamento`: recortes
de 256², ScribblePrompt em 128², canal de fundo do Cellpose com faixa de 2 px, fechamento de 3 px no treino e na avaliação, **só o
Dice suave como perda, sem nenhuma regularização sobre o marcador** (capítulo 4, seção 4.5). A forma editável do marcador vem do
fechamento, que é uma operação fixa, e não de uma perda.

**Protocolo do teste** ([09-correcoes-pontuais.md §18.28](../estudo/09-correcoes-pontuais.md); notebook
`notebooks/experiments/experimento_final.ipynb`):

- o modelo final foi treinado com as 37 imagens por **200 épocas**, decisão do autor tomada antes de ver o teste (capítulo 9, seção
  9.4);
- dois números foram declarados antes de rodar: o **registro** (sobreposição de 1/4, sem TTA), a mesma avaliação usada para comparar
  experimentos, e a **inferência final** (sobreposição de 1/2 e TTA), escolhida só nas imagens de treino. **O número principal do
  trabalho é o da inferência final**;
- as 14 imagens de teste foram avaliadas uma única vez. Uma execução anterior, de 67 épocas, chegou a gravar um modelo, mas o
  resultado dela no teste não foi visto (seção 11.8).

**Resultado** (`docs/estudo/resultados/final_recortes_negativo_r2_fechamento/`, commit `60e89c1`; 14 imagens, todas em 40×):

| Condição | Dice | Precisão | Revocação | Massa | Diferença para o Cellpose (IC 95%) | Imagens acima | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| registro | 0,8300 | 0,782 | 0,890 | 1,15 | −0,0070 (−0,0103 a −0,0036) | 2 de 14 | 0,004 |
| **inferência final (número principal)** | **0,8333** | 0,786 | 0,892 | 1,15 | **−0,0037 (−0,0071 a −0,0002)** | 3 de 14 | **0,12** |
| Cellpose | 0,8370 | 0,819 | 0,864 | 1,07 | | | |

O intervalo do registro vem do `resumo.json`; o da inferência final foi calculado em 2026-10-29 a partir de
`teste_melhor_inferencia.csv`, com o mesmo *bootstrap* do projeto (10.000 reamostragens das imagens, semente 42; capítulo 9, seção
9.5).

- **O número principal fica 0,0037 abaixo do Cellpose.** O teste de Wilcoxon não acusa diferença (p = 0,12), mas o intervalo de
  confiança por *bootstrap* da diferença média fica todo abaixo de zero, por pouco. Com 14 imagens, as duas leituras convivem: a
  diferença é pequena e não se pode afirmar que o pipeline seja igual ao Cellpose nem que seja claramente pior. Nove das 11 imagens
  abaixo perdem menos de 0,01, e a inferência final melhora o registro nas 14 imagens.
- **O padrão do erro é o da validação cruzada, mais forte:** revocação acima do Cellpose em 13 de 14 imagens (+0,029) e precisão
  abaixo em todas (−0,033). A segmentação sai maior que a do Cellpose e que o GT (massa 1,15, contra 1,06 na validação cruzada).
  Hipótese: a massa maior vem das 200 épocas, já que os *folds* pararam entre as épocas 23 e 99; não dá para separar esse efeito do
  efeito das imagens de teste (capítulo 13).
- **A editabilidade se confirma fora das imagens de treino.** No teste, o marcador fechado dá um único componente a **96%** das
  regiões do GT (cru: 86%), com 0,3% das regiões sem marcador, 48 buracos, 263 fragmentos e 22 componentes unindo regiões por
  imagem; o marcador cobre 17,5% da imagem. O marcador cru do modelo final é mais limpo que o da validação cruzada (86% contra 54%);
  hipótese: efeito das épocas a mais.

## 11.8 Situação em relação à pergunta do trabalho

A pergunta era se o pipeline da tese funciona sem o FMBS, com outra camada de segmentação (capítulo 2, seção 2.5). Com o
ScribblePrompt, a resposta é: **funciona no nível da entrada, sem superá-la, e entrega um marcador editável.** No teste, a
segmentação fica 0,0037 abaixo do Cellpose, e o marcador fechado tem um componente em 96% das regiões dos núcleos.

O que limita, segundo os resultados e os oráculos:

- a camada expande o marcador de um jeito que depende da resolução, e a informação de borda vem mais do canal de fundo do que do
  marcador (capítulo 10);
- com o canal de fundo, o marcador pesa pouco na segmentação, e o resultado fica preso ao Cellpose;
- o erro que sobra é de precisão: bordas infladas e núcleos densos fundidos. Parte parece estar na camada de segmentação, em que o
  espaço de 1 a 2 px entre núcleos vizinhos some em 128² (hipótese; [08-pendencias.md, PD-40](../estudo/08-pendencias.md#pd-40),
  leitura transversal).

**Plano de publicação** (decisão do autor, 2026-10-26): o que já existe sustenta o TCC e um artigo, com o pipeline sem FMBS no
nível do Cellpose, os marcadores editáveis pelo fechamento, os oráculos e a análise do tamanho do marcador. A edição com cliques
sobre o marcador fica para um segundo artigo (capítulo 13).

**Pendências registradas** (nenhuma muda o número principal):

1. regerar os CSVs por imagem e as curvas da validação cruzada do `recortes_negativo_r2_fechamento` a partir dos modelos dos *folds*
   gravados, sem treinar. Hoje a pasta dele só tem o `config.json`, reconstruído das saídas do notebook, e as comparações pareadas
   com os outros experimentos dependem desses CSVs;
2. o modelo de 67 épocas (`final_67_epocas.pt`, no Google Drive do autor) não foi avaliado. Avaliá-lo agora seria olhar o teste uma
   segunda vez: serviria só como análise registrada do efeito das épocas, nunca para trocar o número principal.

## Para o artigo

- **Resultados:** a tabela da seção 11.2 (com o Cellpose na mesma tabela), as comparações pareadas que contam a sequência de
  mudanças e a tabela do fechamento da seção 11.4.
- **Resultados ou discussão:** a editabilidade comparada entre perdas (seção 11.4) e o diagnóstico (seção 11.5).
- **Discussão:** as leituras das seções 11.6 a 11.8.
- **Resultados, número principal:** a tabela do teste da seção 11.7, com os dois números declarados antes, o intervalo por
  *bootstrap* e o Wilcoxon, e a editabilidade no teste.
- **Limitações:** escolhas de inferência feitas nas mesmas imagens; 200 épocas no treino final sem validação; o diagnóstico, o
  oráculo de resolução e a validação cruzada da configuração final sem resultados por imagem salvos em arquivo
  ([08-pendencias.md, PD-47](../estudo/08-pendencias.md#pd-47)).

## Referências

Sem referências externas; os métodos estão nos capítulos 4 a 9.
