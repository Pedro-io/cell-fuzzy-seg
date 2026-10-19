# 8. Funções de perda

A perda é o único sinal que a rede de marcadores recebe. Ela tem duas partes, como na tese (capítulo 2, seção 2.3): uma perda
de segmentação, calculada sobre a saída do ScribblePrompt, e regularizações opcionais, calculadas diretamente sobre o
marcador. Este capítulo apresenta cada termo, a forma como eles se combinam e o que já se sabe sobre a interação entre eles
neste pipeline.

Notação: $\hat{y}$ é a segmentação final, com valores em $[0,1]$; $y$ é o ground truth binário; $m$ é o marcador; as somas
correm sobre os pixels. Todos os termos estão em [src/losses/](../../src/losses/).

## 8.1 Como os termos se combinam

A perda total é uma soma ponderada:

$$
L = \lambda_{seg}\, L_{seg}(\hat{y}, y) + \sum_i \lambda_i\, L_i(m, \dots)
$$

Cada termo é um objeto independente com o seu peso, e a lista de termos de um experimento é montada no notebook, sem mudar
o código ([LossComposer](../../src/losses/loss_composer.py), [terms.py](../../src/losses/terms.py)). O compositor entrega aos
termos dois tensores distintos: a segmentação final, para os termos de segmentação, e o marcador, para as regularizações.

## 8.2 Perda de segmentação

**Dice suave** ([soft_dice_loss.py](../../src/losses/soft_dice_loss.py)). Versão contínua do Dice da seção 1.6, calculada sem
binarizar a predição e com os quadrados no denominador, como proposto na V-Net (Milletari et al., 2016, seção 3):

$$
L_{Dice} = 1 - \frac{2 \sum \hat{y}\, y + \epsilon}{\sum \hat{y}^2 + \sum y^2 + \epsilon}
$$

O Dice é calculado por imagem e depois tirada a média do lote. É a perda de segmentação de todos os experimentos principais,
como na tese, que também usa um Dice suave (Fonseca, 2026, seção 6.1.1).

**RMSE** ([rmse_loss.py](../../src/losses/rmse_loss.py)). A raiz do erro quadrático médio entre $\hat{y}$ e $y$, sugerida
pelo orientador. A regra adotada é usar o Dice ou o RMSE, nunca os dois juntos (capítulo 12). Existe um experimento só com
RMSE, ainda não rodado.

## 8.3 Regularizações da tese

As três regularizações abaixo seguem as equações da tese (Fonseca, 2026, seção 5.3). As três dividem por uma medida do ground
truth para que o valor não dependa do tamanho do objeto.

**Tamanho** ([object_size_loss.py](../../src/losses/object_size_loss.py)). Quer um marcador pequeno (tese, equação 5.5):

$$
L_{size} = \frac{\sum m}{\sum y}
$$

É mínima com o marcador vazio. O papel de impedir esse extremo cabe à perda de segmentação: sem marcador, não há
segmentação. Na tese, o termo é calculado por objeto; aqui, as somas cobrem o lote inteiro (seção 8.5).

**Mapa de distância** ([distance_map_loss.py](../../src/losses/distance_map_loss.py)). Quer o marcador longe da borda dos
objetos (tese, equações 5.6 e 5.7):

$$
L_{Dmap} = \frac{\sum m \cdot D}{\sum y}, \qquad D = 1 - \frac{EDT}{\max EDT}
$$

$EDT$ é a distância euclidiana de cada pixel ao fundo. O mapa $D$ vale 0 no centro de cada núcleo, cresce até perto de 1 na
borda e vale 1 no fundo. Ele é calculado **por núcleo**, a partir da máscara por instância, uma vez na preparação dos dados
([compute_instance_distance_map](../../src/pipeline/steps/preprocessing/distance_map_step.py)):

- o máximo da normalização é o de cada núcleo, não o da imagem. Com o máximo da imagem, o centro de um núcleo pequeno teria
  valor alto e seria punido como se fosse borda: medido com o máximo da imagem, 43,5% dos núcleos do treino tinham o centro
  acima de 0,5; com o máximo por núcleo, nenhum ([08-pendencias.md, PD-06](../estudo/08-pendencias.md#pd-06));
- núcleos que se tocam são tratados separadamente: a fronteira entre dois núcleos conta como borda dos dois, o que o GT
  binário não permitiria;
- a borda da imagem não conta como fundo: num núcleo cortado pela borda, o centro verdadeiro provavelmente está do lado de
  fora, e o ponto mais barato fica junto dela.

**Variação total** ([total_variation_loss.py](../../src/losses/total_variation_loss.py)). Quer um marcador suave, sem
variações bruscas entre pixels vizinhos (tese, equação 5.8):

$$
L_{TV} = \frac{\sum_{j,k} |m_{j+1,k} - m_{j,k}| + |m_{j,k+1} - m_{j,k}|}{\sqrt{\sum y}}
$$

**Componentes ($L_{comp}$).** A tese tem uma quarta regularização, que favorece marcadores com um único componente conexo,
calculada sobre a árvore de componentes do marcador. Ela não foi implementada: neste trabalho o marcador deve ter centenas de
componentes, um por núcleo, e o termo puxaria na direção errada (capítulo 2, seção 2.4; capítulo 12).

## 8.4 Termos que não estão na tese

Sugeridos pelo orientador, implementados e fora dos experimentos atuais:

- **Borda da imagem** ([border_loss.py](../../src/losses/border_loss.py)): penaliza a massa do marcador numa faixa de 50 px
  junto às bordas da imagem, dividida pelo número total de pixels. A faixa cobre 19% da área da imagem e, com o GT da
  rasterização anterior, 16 a 17% dos pixels de núcleo, que o termo empurraria para não serem marcados
  ([08-pendencias.md, PD-41](../estudo/08-pendencias.md#pd-41)). O motivo original do termo não é conhecido.
- **Não muito fino** ([not_too_thin_loss.py](../../src/losses/not_too_thin_loss.py)): compara o marcador com uma abertura
  morfológica dele e penaliza as partes finas que a abertura removeria. Nunca foi usado.

## 8.5 Detalhes de implementação

- **Normalização pelo lote.** $L_{size}$, $L_{Dmap}$ e $L_{TV}$ somam o marcador e o GT do lote inteiro antes de dividir. Na
  tese, a normalização é por objeto. A diferença dá mais peso às imagens com mais núcleos no lote; o efeito não foi medido
  ([08-pendencias.md, PD-12](../estudo/08-pendencias.md#pd-12)).
- **Lote sem núcleo.** Com recortes, um lote pode não ter nenhum pixel de núcleo. O denominador de $L_{Dmap}$ e de $L_{TV}$ é
  limitado a no mínimo 1 para não dividir por zero; $L_{size}$ passa a valer a massa do marcador.
- **Pesos.** Os pesos são relativos ao Dice, que fica com peso 1. A tabela mostra os pesos da tese e os das configurações
  definidas nos notebooks de experimento:

| Configuração | Dice | Size | Dmap | TV |
|---|---|---|---|---|
| Tese, modelo de marcador de objeto (pesos originais) | 30 | 3 | 0,01 | 0,01 |
| Tese, na mesma proporção com Dice = 1 (notebook `tese`) | 1 | 0,1 | 1/3000 | 1/3000 |
| `base_dice`, `recortes`, `negativo_sem_tv` | 1 | | | |
| `dice_size` | 1 | 0,1 | | |
| `dice_dmap` | 1 | | 0,1 | |
| `base_dice_tv`, `base_negativo` | 1 | | | 0,001 |

Fontes: Fonseca (2026), seção 6.1.1; notebooks em `notebooks/experiments/` e `notebooks/experiments/legado/`.

## 8.6 Como os termos interagem neste pipeline

Na tese, as regularizações convivem com a perda de segmentação porque o FMBS expande um marcador pequeno até o objeto inteiro:
um marcador pequeno e central ainda produz a segmentação certa. O ScribblePrompt expande pouco. Os oráculos mostram que, sem
canal de fundo, só marcadores quase do tamanho do núcleo levam a uma segmentação boa (capítulo 10). Daí uma tensão:

| Termo | O que pede ao marcador | O que o ScribblePrompt precisa | Relação |
|---|---|---|---|
| $L_{size}$ | pequeno | quase do tamanho do núcleo | conflito |
| $L_{Dmap}$ | no centro do núcleo | cobrindo o núcleo | conflito |
| Borda da imagem | longe da borda da imagem | cobrindo também os núcleos da borda | conflito |
| $L_{TV}$ | suave | indiferente | neutro, a princípio |

Por isso a configuração base usa só o Dice, e as regularizações são reintroduzidas uma a uma, para medir o efeito de cada
uma (capítulo 11).

**O caso do TV.** O TV parecia neutro, mas o treino com o canal de fundo do Cellpose mostrou outra coisa: o marcador aprendido
virou um campo quase uniforme de cerca de 0,5 (capítulo 11). Quatro fatos, cada um conferido, explicam o mecanismo
([09-correcoes-pontuais.md §18.17](../estudo/09-correcoes-pontuais.md)):

1. sem $L_{size}$ e $L_{Dmap}$, a única pressão sobre a forma do marcador é o Dice da segmentação final;
2. o canal de fundo cobre cerca de 65% da imagem, e onde ele já apaga a saída o Dice não depende do marcador;
3. abaixo do corte $\tau$, o gradiente do Dice sobre o marcador é exatamente zero (capítulo 7, seção 7.3), e só o TV age ali;
4. com peso 0,001, o TV de um marcador do tamanho dos núcleos vale cerca de 0,14 a 0,20, a mesma ordem da perda de Dice
   (cerca de 0,2), e é zero num campo constante.

Sem o canal de fundo, um marcador uniforme faria o ScribblePrompt marcar a imagem inteira, e o Dice obriga o marcador a se
localizar. Com o canal de fundo, localizar rende pouco Dice, e o TV vence. Hipótese: a combinação desses quatro fatos é a
causa do achatamento; o experimento `negativo_sem_tv` testa isso.

## Para o artigo

- **Métodos:** a equação da seção 8.1 e as fórmulas do Dice suave e das regularizações (seções 8.2 e 8.3), com a observação
  de que o mapa de distância é por núcleo; o motivo para não usar o $L_{comp}$.
- **Discussão:** a tensão entre as regularizações da tese e uma camada que expande pouco (seção 8.6). É um dos achados
  centrais para a pergunta do trabalho.
- **Detalhes de implementação:** normalização pelo lote e lotes sem núcleo (seção 8.5); a tabela de pesos.

## Referências

- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seções 5.3 e 6.1.1).
- Milletari, F.; Navab, N.; Ahmadi, S.-A. (2016). V-Net: fully convolutional neural networks for volumetric medical image
  segmentation. *International Conference on 3D Vision (3DV)*. arXiv:1606.04797. *Conferida* (seção 3, função objetivo
  baseada no Dice).
