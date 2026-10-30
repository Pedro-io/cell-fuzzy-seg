# 8. Funções de perda

A perda é o único sinal que a rede de marcadores recebe durante o treino. Este capítulo explica, com exemplos numéricos, o que
cada perda do projeto mede, por que ela foi usada, o que ela fez nos experimentos e como acompanhar as perdas no log de treino.

Notação: $\hat{y}$ é a segmentação final, com valores em $[0,1]$ (a saída do ScribblePrompt depois da sigmoide); $y$ é o ground
truth binário (1 no núcleo, 0 no fundo); $m$ é o marcador produzido pela MarkerUNet; as somas correm sobre os pixels. Todos os
termos estão em [src/losses/](../../src/losses/).

## 8.1 O que é uma perda, em poucas palavras

Uma perda é um número que mede o quanto a saída está errada: zero é perfeito, e quanto maior, pior. A cada passo do treino, o
otimizador calcula em que direção mexer cada peso da MarkerUNet para a perda diminuir (o gradiente) e dá um pequeno passo
nessa direção. Três consequências práticas:

- **uma perda só ensina o que ela mede.** Se nada na perda olha um aspecto da saída, a rede não tem motivo para melhorá-lo;
- **onde a perda não muda quando a saída muda, o gradiente é zero, e a rede não aprende nada ali.** Esse ponto explica vários
  resultados deste capítulo;
- **quando há vários termos, cada um tem um peso** $\lambda$, e a perda total é a soma ponderada. O peso decide quanto cada termo
  puxa. Se dois termos querem coisas opostas, vence o que tiver mais força na soma.

A perda total do projeto é:

$$
L = \underbrace{\lambda_{seg}\, L_{seg}(\hat{y}, y)}_{\text{sobre a segmentação}} + \underbrace{\sum_i \lambda_i\, L_i(m, \dots)}_{\text{sobre o marcador}}
$$

Os termos de segmentação dizem "a saída final tem de parecer com o ground truth". Os termos sobre o marcador dizem "o marcador
tem de ter tal forma". Cada termo é um objeto com o seu peso, e o experimento monta a lista no notebook
([LossComposer](../../src/losses/loss_composer.py), [terms.py](../../src/losses/terms.py)).

**Onde cada termo olha.** O caminho do marcador até a segmentação tem três etapas depois da MarkerUNet: o fechamento morfológico
(quando usado), o corte em $\tau$ que transforma o marcador no canal positivo e o ScribblePrompt (capítulos 4 e 7). Os termos de
segmentação olham a saída do ScribblePrompt; os termos do marcador olham o marcador **cru**, antes do fechamento e do corte
([trainer.py](../../src/training/trainer.py), `_compute_loss`, que entrega `markers` aos termos). Isso tem uma consequência: um
termo do marcador "vê" valores que o ScribblePrompt nunca recebe, porque tudo abaixo de $\tau = 0{,}5$ vira zero no corte.

## 8.2 Perdas sobre a segmentação

### Dice suave

O Dice da avaliação (capítulo 1, seção 1.6) conta pixels de uma máscara binária, e contar pixels não tem gradiente. O Dice
suave usa as probabilidades no lugar das contagens, com os quadrados no denominador, como proposto na V-Net
(Milletari et al., 2016, seção 3) ([soft_dice_loss.py](../../src/losses/soft_dice_loss.py)):

$$
L_{Dice} = 1 - \frac{2 \sum \hat{y}\, y + \epsilon}{\sum \hat{y}^2 + \sum y^2 + \epsilon}
$$

O Dice é calculado por imagem e depois tirada a média do lote.

**Exemplo.** Uma imagem com 4 pixels, dois de núcleo e dois de fundo: $y = (1, 1, 0, 0)$.

| Predição $\hat{y}$ | Numerador $2\sum \hat{y} y$ | Denominador $\sum \hat{y}^2 + \sum y^2$ | $L_{Dice}$ | Dice binarizado em 0,5 |
|---|---|---|---|---|
| $(1;\ 1;\ 0;\ 0)$, perfeita | 4,0 | 4,0 | 0,000 | 1,00 |
| $(0{,}9;\ 0{,}8;\ 0{,}2;\ 0{,}1)$, confiante e certa | 3,4 | 3,5 | 0,029 | 1,00 |
| $(0{,}6;\ 0{,}6;\ 0{,}4;\ 0{,}4)$, hesitante e certa | 2,4 | 3,04 | 0,211 | 1,00 |
| $(1;\ 1;\ 1;\ 0)$, um falso positivo | 4,0 | 5,0 | 0,200 | 0,80 |

O que o exemplo mostra:

- o Dice suave mede a **sobreposição** entre as regiões prevista e real. O fundo que está certo ($\hat{y} = 0$ onde $y = 0$) não
  entra na conta, então a perda não é dominada pelo fundo, que é cerca de 3/4 da imagem (capítulo 1, seção 1.4);
- ele pune a **hesitação**: a terceira linha erra zero pixels depois de binarizada, mas custa quase tanto quanto um falso positivo.
  Isso empurra a rede para respostas confiantes.

É a perda de segmentação de todos os experimentos e a da tese (Fonseca, 2026, seção 6.1.1).

### Entropia cruzada binária (BCE)

A BCE mede, pixel a pixel, a "surpresa" de cada predição diante da resposta certa, e tira a média sobre todos os pixels
([bce_loss.py](../../src/losses/bce_loss.py)):

$$
L_{BCE} = -\frac{1}{N} \sum \big[\, y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \,\big]
$$

As probabilidades são cortadas em $[10^{-6}, 1 - 10^{-6}]$ para o logaritmo não ir ao infinito.

**Exemplo.** O custo de um único pixel de fundo ($y = 0$) com probabilidade $\hat{y}$:

| $\hat{y}$ num pixel de fundo | Custo $-\log(1 - \hat{y})$ |
|---|---|
| 0,05 (certo, confiante) | 0,05 |
| 0,4 (certo depois de binarizar, mas hesitante) | 0,51 |
| 0,6 (errado, hesitante) | 0,92 |
| 0,9 (errado, confiante) | 2,30 |

O custo cresce muito rápido para um erro confiante. A BCE olha **cada pixel isoladamente**, inclusive o fundo certo, ao contrário
do Dice suave, que olha a sobreposição das regiões.

**Por que foi testada.** Nos recortes com o canal de fundo do Cellpose, o pipeline já passava o Cellpose em revocação, e faltava
precisão: a borda prevista ficava "borrada" e inchada (capítulo 11). A ideia era que a BCE, punindo cada pixel de fundo com
probabilidade alta, afinasse essa borda ([09-correcoes-pontuais.md §18.21](../estudo/09-correcoes-pontuais.md)). Hipótese a
conferir: o Dice somado à entropia cruzada é a perda usada pelo nnU-Net na versão publicada (Isensee et al., 2021); a versão de
2018 em `docs/papers/` descreve só o Dice.

**O que aconteceu.** `recortes_negativo_bce` (Dice + BCE com peso 1): 0,8335 na validação cruzada, contra 0,8350 sem a BCE
(−0,002, pior em 26 de 37 imagens, $p = 0{,}008$). O marcador ficou mais picotado: sem fechamento, 91% dos núcleos têm o marcador
partido em mais de um pedaço (capítulo 11). A BCE não melhorou a precisão.

### RMSE

A raiz do erro quadrático médio entre $\hat{y}$ e $y$ ([rmse_loss.py](../../src/losses/rmse_loss.py)), sugerida pelo orientador.
Como a BCE, olha cada pixel. A regra adotada é usar o Dice ou o RMSE, nunca os dois (capítulo 12). O experimento só com RMSE não
foi rodado.

## 8.3 Perdas sobre o marcador

Estas perdas vêm da tese (Fonseca, 2026, seção 5.3). Lá, o objetivo delas é deixar o marcador **fácil de editar por uma
pessoa**: pequeno, longe da borda do objeto, com poucos componentes e suave, parecido com um traço que alguém desenharia. Elas não
existem para melhorar a segmentação; quem expande o marcador pequeno até o objeto inteiro é o FMBS. Por isso, neste trabalho elas
são julgadas pela editabilidade do marcador, com o Dice como restrição (não pode piorar) (capítulo 11; 09 §18.25).

As três dividem por uma medida do ground truth, para que o valor não dependa do tamanho dos objetos. A tabela abaixo dá a escala
de cada termo, com peso 1, para dois marcadores binários medidos nos dados (lotes de 4 imagens inteiras;
[09-correcoes-pontuais.md §18.13](../estudo/09-correcoes-pontuais.md)):

| Termo | Marcador do tamanho do núcleo | Marcador só no miolo (cerca de 1/3 da área) |
|---|---|---|
| $L_{size}$ | 0,95 a 1,00 | 0,33 |
| $L_{Dmap}$ | 0,59 | 0,10 |
| $L_{TV}$ | cerca de 200 | cerca de 140 |

### Tamanho ($L_{size}$)

$$
L_{size} = \frac{\sum m}{\sum y}
$$

([object_size_loss.py](../../src/losses/object_size_loss.py); tese, equação 5.5.) É a "massa" do marcador dividida pela área dos
núcleos: vale cerca de 1 para um marcador que cobre os núcleos inteiros e cai à medida que ele encolhe. Sozinho, seria mínimo com
o marcador vazio; quem impede isso é a perda de segmentação, porque sem marcador não há segmentação.

Detalhe que importa: como o termo olha o marcador cru, ele também cobra os valores abaixo de $\tau$, que o ScribblePrompt nunca
vê. A rede pode baixar esses valores "de graça", sem nenhum efeito na segmentação.

**O que aconteceu.** No `dice_size` (Dice + Size 0,1, imagem inteira em 256²), o marcador encolheu de 9,0% para 5,8% da imagem e
a segmentação trocou revocação por precisão (massa 1,17 para 1,09), com o mesmo Dice (0,774 contra 0,775). Encolher o marcador não
melhorou a segmentação (seção 8.6).

### Mapa de distância ($L_{Dmap}$)

$$
L_{Dmap} = \frac{\sum m \cdot D}{\sum y}, \qquad D = 1 - \frac{EDT}{\max EDT}
$$

([distance_map_loss.py](../../src/losses/distance_map_loss.py); tese, equações 5.6 e 5.7.) $EDT$ é a distância de cada pixel ao
fundo mais próximo. O mapa $D$ funciona como uma "tabela de preços" para o marcador: vale 0 no ponto mais interno de cada
núcleo, cresce até perto de 1 na borda e vale 1 no fundo. Marcar o centro sai quase de graça; marcar a borda ou o fundo custa
caro. Numa célula redonda, $D$ cresce em linha reta do centro até a borda.

O mapa é calculado **por núcleo**, a partir da máscara por instância, uma vez na preparação dos dados
([compute_instance_distance_map](../../src/pipeline/steps/preprocessing/distance_map_step.py)):

- o máximo da normalização é o de cada núcleo, não o da imagem. Com o máximo da imagem, o centro de um núcleo pequeno teria
  preço alto, como se fosse borda: medido assim, 43,5% dos núcleos do treino tinham o centro acima de 0,5; por núcleo, nenhum
  ([08-pendencias.md, PD-06](../estudo/08-pendencias.md#pd-06));
- núcleos que se tocam são tratados separadamente: a fronteira entre dois núcleos conta como borda dos dois, o que o GT binário
  não permitiria;
- a borda da imagem não conta como fundo: num núcleo cortado pela borda, o centro verdadeiro provavelmente está do lado de fora.

**O que aconteceu.** No `recortes_negativo_dmap` (Dice + Dmap 0,1), o marcador fez o que o termo pede: ficou menor (8,2% da imagem,
contra 14,2% sem o Dmap) e mais limpo fora dos núcleos (valor médio no fundo de 0,36 para 0,09). Mas ficou **picotado**, com traços
finos e muitos fragmentos, e deixou 8% dos núcleos sem nenhum marcador. A segmentação piorou um pouco: 0,830 contra 0,835 (pior em
35 de 37 imagens, $p < 0{,}001$) (capítulo 11).

### Variação total ($L_{TV}$)

$$
L_{TV} = \frac{\sum_{j,k} \lvert m_{j+1,k} - m_{j,k} \rvert + \lvert m_{j,k+1} - m_{j,k} \rvert}{\sqrt{\sum y}}
$$

([total_variation_loss.py](../../src/losses/total_variation_loss.py); tese, equação 5.8.) Soma o quanto o marcador muda de um
pixel para o vizinho. Um marcador binário paga, em essência, o comprimento do seu contorno; um marcador com ruído paga muito; e um
**campo constante paga zero**, qualquer que seja o valor.

Na tese, com um objeto por imagem, dividir pela raiz da área torna o termo independente do tamanho do objeto. Aqui as somas
cobrem todos os núcleos do lote: o numerador cresce com o número de núcleos e o denominador com a raiz desse número, então o valor
cresce com a raiz do número de núcleos do lote (dedução a partir da fórmula; não medido).

**O que aconteceu.** Com o canal de fundo do Cellpose, o TV achatou o marcador num campo quase uniforme de cerca de 0,5. O
mecanismo junta três pontos deste capítulo: com o canal de fundo, o Dice quase não depende do marcador; abaixo de $\tau$, o
gradiente do Dice sobre o marcador é zero; e o TV, com peso 0,001, vale cerca de 0,14 a 0,20, a mesma ordem da perda de Dice, e é
zero num campo constante. Tirar o TV confirmou isso: `negativo_sem_tv` 0,819 contra 0,808 do `base_negativo` (+0,012, melhor em 36
de 37 imagens), e o marcador voltou a distinguir núcleo de fundo (valor médio 0,55 dentro e 0,30 fora). Sem o canal de fundo, tirar
o TV também ajudou: `base_dice` 0,785 contra 0,775 do `base_dice_tv` (+0,011, 29 de 37) (capítulo 11). O TV saiu de todos os
experimentos seguintes.

### Componentes ($L_{comp}$, não implementado)

A tese tem uma quarta regularização, que favorece marcadores com um único componente conexo, calculada sobre a árvore de
componentes do marcador. Ela não foi implementada: é complexa, foi pensada junto com o FMBS e, com o alvo de um componente, puxaria
na direção errada num marcador que deve ter centenas de componentes, um por núcleo (capítulo 2, seção 2.4; capítulo 12).

Hipótese: a falta dela explica por que o Dmap picota o marcador. Na tese, o $L_{comp}$ segura os componentes enquanto o Size e o
Dmap encolhem o marcador; sem ele, encolher quebra o marcador em pedaços. O experimento `recortes_negativo_r2_fechamento_tese`
testou se o fechamento morfológico (capítulo 4) cumpre esse papel, emendando os pedaços antes do ScribblePrompt. Do jeito testado,
não cumpre: com Size e Dmap, só 55% das regiões do GT ficam com um único componente de marcador fechado, contra 89% só com o Dice,
e quase 10% ficam sem marcador (capítulo 11, seção 11.4). Ressalva: o Size e o Dmap olham o marcador cru (seção 8.1), então a perda
não sabe que o fechamento vai emendar os pedaços; regularizar o marcador fechado não foi testado.

## 8.4 Termos que não estão na tese

Sugeridos pelo orientador, implementados e fora dos experimentos:

- **Borda da imagem** ([border_loss.py](../../src/losses/border_loss.py)): cobra a massa do marcador numa faixa de 50 px junto às
  bordas da imagem, dividida pelo número total de pixels. A faixa cobre 19% da área da imagem e, com a rasterização anterior do GT,
  16 a 17% dos pixels de núcleo, que o termo empurraria para ficar sem marcador
  ([08-pendencias.md, PD-41](../estudo/08-pendencias.md#pd-41)). O motivo original do termo não é conhecido.
- **Não muito fino** ([not_too_thin_loss.py](../../src/losses/not_too_thin_loss.py)): compara o marcador com uma abertura
  morfológica dele e cobra as partes finas que a abertura removeria. Nunca foi usado.

## 8.5 Pesos usados

Os pesos são relativos ao Dice, que tem peso 1. O peso da tese "em proporção" é o peso dela dividido pelo peso da perda de
segmentação (30).

| Experimento | Dice | BCE | Size | Dmap | TV | Rodou? |
|---|---|---|---|---|---|---|
| Tese, modelo de marcador de objeto (pesos originais) | 30 | | 3 | 0,01 | 0,01 | |
| `base_dice_tv`, `base_negativo` | 1 | | | | 0,001 | sim |
| `dice_size` | 1 | | 0,1 | | | sim |
| `base_dice`, `negativo_sem_tv`, `recortes`, `recortes_negativo`, `recortes_negativo_r2_fechamento` | 1 | | | | | sim |
| `recortes_negativo_dmap` | 1 | | | 0,1 | | sim |
| `recortes_negativo_bce` | 1 | 1 | | | | sim |
| `recortes_negativo_bce_dmap` | 1 | 1 | | 0,1 | | interrompido (sem memória na GPU) |
| `recortes_negativo_tese` (tese em proporção, sem TV) | 1 | | 0,1 | 1/3000 | | não |
| `recortes_negativo_tese_bce` | 1 | 1 | 0,1 | 1/3000 | | não |
| `recortes_negativo_r2_fechamento_tese` | 1 | | 0,1 | 0,1 | | sim |

Fontes: Fonseca (2026), seção 6.1.1; notebooks em `notebooks/experiments/` e `notebooks/experiments/legado/`. Com o peso da tese
em proporção, o Dmap vale cerca de 0,0002 e praticamente não age, por isso os experimentos mais recentes usam 0,1.

Para ter uma ideia do quanto cada termo pesa na soma: com os pesos acima, o Size de um marcador do tamanho dos núcleos vale cerca de
0,1 × 1 = 0,1; o Dmap, cerca de 0,1 × 0,6 = 0,06; o TV com peso 0,001, de 0,14 a 0,20; e a perda de Dice de um modelo treinado
fica em torno de 0,2. Um termo de 0,06 contra um Dice de 0,2 já tem força para mudar a forma do marcador, como o Dmap mudou.

## 8.6 O que os experimentos ensinaram

**Na configuração final, a única perda é o Dice suave.** Nenhuma regularização sobre o marcador é usada: a forma editável do
marcador vem do fechamento morfológico, uma operação fixa (capítulo 4, seção 4.2), e o canal de fundo do Cellpose é uma entrada
fixa do ScribblePrompt, não uma perda. As lições abaixo explicam por que as regularizações ficaram de fora.

1. **Com o ScribblePrompt, encolher o marcador não melhora a segmentação.** O tamanho certo do marcador é o que a camada consegue
   expandir até a borda do núcleo, nem menos nem mais. Encolher além disso (Size, Dmap) troca revocação por precisão ou deixa
   núcleos sem marcador; aumentar além disso faz a camada vazar. O ponto de equilíbrio depende da resolução da entrada do
   ScribblePrompt (capítulo 10). Na tese é diferente: o FMBS expande qualquer marcador até o objeto, e marcadores pequenos não
   custam nada à segmentação ([08-pendencias.md, PD-40](../estudo/08-pendencias.md#pd-40), leitura transversal).
2. **Com o canal de fundo, o marcador pesa menos.** No `recortes_negativo_dmap`, o marcador mudou muito e a segmentação quase não
   mudou: quem desenha a borda e o fundo é o canal de fundo com o ScribblePrompt.
3. **Termos que valem zero num marcador sem informação podem apagá-lo.** O TV achatou o marcador quando o Dice parou de depender
   dele. Antes de somar um termo sobre o marcador, vale perguntar o que o minimiza e se a perda de segmentação ainda segura o
   marcador nessa situação.
4. **Só o Dice deu o melhor marcador nos dois critérios.** Entre os experimentos com recortes e canal de fundo, o treinado só com
   Dice teve o maior Dice e, com o fechamento, o marcador mais editável: cerca de 90% das regiões com um único componente de
   marcador, contra 66% com o Dmap, 65% com a BCE e 55% com Size e Dmap juntos e o fechamento dentro do treino (capítulo 11,
   seção 11.4). Sem um termo que segure os componentes, as perdas que encolhem o marcador o quebram, e o fechamento não basta para
   emendá-lo.

## 8.7 Como ler as perdas no log de treino

Durante o treino, o notebook imprime uma linha a cada 10 épocas
([training_loop.py](../../src/training/training_loop.py), `_log_epoch`):

```
[TrainingLoop] Epoch 20/200 | train_loss=0.3445 | val_loss=0.3730 | val_dice=0.8189 | best_dice=0.8189 | best_epoch=20
```

| Campo | O que é |
|---|---|
| `train_loss` | a perda total (a soma ponderada de todos os termos), média dos lotes de treino da época; no modo por recortes, calculada sobre os recortes |
| `val_loss` | a mesma perda total nas imagens de validação do *fold* |
| `val_dice` | o Dice binarizado em 0,5, média por imagem da validação; é o critério que escolhe a melhor época ([best_model_callback.py](../../src/training/callbacks/best_model_callback.py)) |
| `best_dice`, `best_epoch` | o melhor `val_dice` até aqui e a época em que ele ocorreu |

Quando o treino para, aparece também `[BestModelCallback] 20 validações sem melhora; melhor Dice ... na época ...`.

Como ler:

- **Acompanhe o `val_dice`, não a perda.** O Dice binarizado é o que se avalia. A perda pode cair sem que o Dice suba, por
  exemplo quando a rede só fica mais confiante (seção 8.2).
- **Não compare `train_loss` entre experimentos com perdas diferentes.** Ela é uma soma de termos; com a BCE somada, por exemplo, a
  perda total da primeira época do `recortes_negativo_bce` foi 0,87, e a do `recortes_negativo`, só com o Dice, 0,27, sem que um
treino fosse pior que o outro por isso (`curvas.csv` de cada experimento).
- **Diferença entre treino e validação.** Uma `val_loss` que sobe enquanto a `train_loss` cai indica que a rede está decorando as
  imagens de treino. A parada antecipada pela `val_dice` protege contra isso (capítulo 9).
- **O valor de cada termo não aparece no log.** O laço de treino guarda a média de cada termo por época na memória (`train_terms`
  e `val_terms` no histórico), mas não as imprime, e o arquivo `curvas.csv` dos resultados guarda só a perda total e o Dice de
  validação ([kfold.py](../../src/training/kfold.py), [report.py](../../src/evaluation/report.py)). Para saber quanto o Size ou o
  Dmap pesaram num experimento, hoje é preciso refazer a conta com os modelos gravados.

## 8.8 Detalhes de implementação

- **Normalização pelo lote.** $L_{size}$, $L_{Dmap}$ e $L_{TV}$ somam o marcador e o GT do lote inteiro antes de dividir. Na
  tese, a normalização é por objeto. A diferença dá mais peso às imagens ou recortes com mais núcleos no lote; o efeito não foi
  medido ([08-pendencias.md, PD-12](../estudo/08-pendencias.md#pd-12)).
- **Lote sem núcleo.** Com recortes, um lote pode não ter nenhum pixel de núcleo. O denominador de $L_{Dmap}$ e de $L_{TV}$ é
  limitado a no mínimo 1 para não dividir por zero; $L_{size}$ passa a valer a massa do marcador.
- **Valores no histórico.** Cada termo já sai multiplicado pelo seu peso; a perda total é a soma deles.

## Para o artigo

- **Métodos:** a equação da seção 8.1, as fórmulas do Dice suave (seção 8.2) e das regularizações (seção 8.3), com o mapa de
  distância por núcleo; o motivo para não usar o $L_{comp}$; a tabela de pesos dos experimentos apresentados.
- **Discussão:** a seção 8.6. O ponto central para a pergunta do trabalho é o 1: sem o FMBS, as regularizações da tese deixam de
  servir à segmentação, e o tamanho certo do marcador passa a depender da camada. O ponto 4 sustenta a escolha do Dice com
  fechamento.
- **Detalhes de implementação:** normalização pelo lote e lotes sem núcleo (seção 8.8).

## Referências

- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seções 5.3 e 6.1.1).
- Isensee, F. et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature
  Methods*, 18, 203-211. *A conferir* (combinação de Dice e entropia cruzada).
- Milletari, F.; Navab, N.; Ahmadi, S.-A. (2016). V-Net: fully convolutional neural networks for volumetric medical image
  segmentation. *International Conference on 3D Vision (3DV)*. arXiv:1606.04797. *Conferida* (seção 3).
