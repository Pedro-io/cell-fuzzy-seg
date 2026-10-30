# 9. Treinamento e protocolo de avaliação

Este capítulo descreve como a MarkerUNet é treinada e como uma configuração é avaliada: a otimização, a validação cruzada
usada para escolher configurações, a seleção da época, o treino final, o uso único do conjunto de teste, a estatística da
comparação com o Cellpose e o que é gravado de cada experimento.

## 9.1 Otimização

| Item | Valor |
|---|---|
| Otimizador | Adam, taxa de aprendizado $10^{-4}$ |
| Agenda da taxa | cosseno, de $10^{-4}$ a 0, ao longo de 200 épocas, com passo a cada lote |
| Recorte do gradiente | norma L2 máxima 1,0 |
| Lote | 4 imagens inteiras, ou 16 recortes de 256 × 256 no modo por recortes (4 recortes por imagem por época) |
| Épocas | até 200, com parada antecipada (seção 9.3) |
| Aumentação | rotações de 90° e espelhamentos, sorteados a cada época (capítulo 3, seção 3.6) |
| Semente | base 42: a divisão dos *folds* usa 42; cada *fold* usa $42 + k$ e o treino final $42 + 5$ para a inicialização, a aumentação e a ordem dos lotes |

Fontes: [kfold.py](../../src/training/kfold.py) (`TrainConfig` e `_train`), notebooks de `notebooks/experiments/`. Só os pesos
da MarkerUNet entram no otimizador. O laço de treino é o de [trainer.py](../../src/training/trainer.py) e
[training_loop.py](../../src/training/training_loop.py): a cada lote, o pipeline roda, a perda é calculada, o gradiente é
recortado e o otimizador dá um passo.

A agenda cobre sempre 200 épocas, mesmo quando a parada antecipada interrompe antes, e mesmo no treino final, que roda menos
épocas. Assim a taxa de aprendizado segue a mesma curva em todos os treinos de uma configuração.

## 9.2 Validação cruzada

O conjunto de treino tem só 37 imagens. Separar uma parte fixa para validação tiraria imagens do treino e daria uma
estimativa baseada em poucas imagens. Por isso a escolha de configurações usa validação cruzada em 5 *folds* sobre as 37
imagens ([make_folds](../../src/training/kfold.py)):

- a divisão é por imagem, aleatória com a semente 42, sem estratificação; cada *fold* separa 7 ou 8 imagens para validação e
  treina com as outras 29 ou 30;
- a divisão é a mesma em todos os experimentos, então as comparações entre experimentos são pareadas por imagem;
- cada *fold* treina uma MarkerUNet nova, com o codificador pré-treinado e o decodificador aleatório.

**Como uma configuração é medida.** Cada uma das 37 imagens é prevista só pelo modelo do *fold* em que ela ficou de fora. Assim,
todas as imagens recebem uma predição feita por um modelo que não as viu no treino, e a nota da configuração é a média dessas 37
predições. No modo por recortes, a validação usa a imagem inteira, avaliada em blocos (capítulo 4, seção 4.3).

## 9.3 Seleção da época e parada antecipada

Em cada *fold*, o modelo é avaliado nas imagens de validação ao fim de cada época. Guarda-se o estado da época com o maior Dice
médio de validação, e o treino para depois de 20 validações seguidas sem melhora
([BestModelCallback](../../src/training/callbacks/best_model_callback.py)). O modelo avaliado no *fold* é o da melhor época, não
o da última.

## 9.4 Treino final e teste

Escolhida a configuração pela validação cruzada, um modelo final é treinado com as 37 imagens pelo número de épocas da
**mediana das melhores épocas** dos 5 *folds* ([final_epochs](../../src/training/kfold.py)) e avaliado nas 14 imagens de teste
**uma única vez**, no notebook `notebooks/experiments/experimento_final.ipynb`. Esse notebook confere que a configuração dele é
idêntica à do experimento escolhido e recusa seguir se não for. Os notebooks de experimento não carregam o conjunto de teste.

Os modelos dos *folds* e o modelo final servem a coisas diferentes. Os dos *folds* avaliam a configuração; o final é o que se
usaria numa imagem nova, e é ele que mede o teste.

**Alternativa registrada.** Em vez do treino final, cada imagem nova poderia passar pelos 5 modelos dos *folds*, com a média
das saídas como predição (*ensemble*). É o que o nnU-Net faz: treina com validação cruzada em 5 *folds* e usa as cinco redes
como *ensemble* no teste (Isensee et al., 2018, seção 2.4). Custa 5 vezes mais na predição. Fica como opção, já que os modelos
dos *folds* são gravados.

**As épocas do treino final da configuração escolhida.** O padrão do código é a mediana das melhores épocas dos *folds*, que
na configuração final dava 67. O autor decidiu, antes de ver o teste, treinar o modelo final por **200 épocas**
(`EPOCAS_FINAL = 200` no notebook final; [09-correcoes-pontuais.md §18.28](../estudo/09-correcoes-pontuais.md)):

- **a favor:** a agenda em cosseno cobre sempre 200 épocas (seção 9.1). Com 200, o treino completa o ciclo e termina com a taxa de
  aprendizado em zero, como foi desenhado; com 67, pararia com a taxa ainda alta;
- **contra:** as 200 épocas não foram validadas. Os *folds* pararam nas épocas 99, 23, 77, 43 e 67, pela parada antecipada, e o
  treino final não tem validação para parar. Há risco de sobreajuste.

O número de épocas vai para o texto junto com o resultado. A massa da segmentação no teste (1,15) ficou maior que na validação
cruzada (1,06); hipótese: efeito das épocas a mais (capítulo 11, seção 11.7). A alternativa de escolher as épocas pela curva média
de validação dos *folds* não chegou a ser usada.

**Números declarados antes do teste.** Antes de rodar, ficaram definidos os dois números a reportar: o **registro** (sobreposição
de 1/4, sem TTA) e a **inferência final** (sobreposição de 1/2 e TTA, seção 9.8), com o segundo como número principal.

## 9.5 Comparação com o Cellpose

Toda configuração é comparada com a máscara do Cellpose nas mesmas imagens (capítulo 5, seção 5.4). A função
[summarize](../../src/evaluation/report.py) calcula, para o Dice e as demais métricas:

- a média e o desvio por imagem, do pipeline e do Cellpose;
- a diferença pareada média (pipeline menos Cellpose) e o intervalo de confiança de 95% dela por *bootstrap* percentil, com
  10.000 reamostragens das imagens;
- quantas imagens melhoram e quantas pioram;
- o valor-p do teste de postos sinalizados de Wilcoxon (Wilcoxon, 1945) sobre as diferenças pareadas.

O teste pareado é o adequado aqui porque a variação entre imagens (densidade, escala, coloração) é grande em comparação com a
diferença entre métodos na mesma imagem. O mesmo resumo é feito na validação cruzada (37 imagens) e no teste (14 imagens).

## 9.6 O que é gravado

| O quê | Onde | Conteúdo |
|---|---|---|
| Resultados de cada experimento | `docs/estudo/resultados/<experimento>/`, no git | configuração, procedência (commit, versões, GPU), métricas por imagem da validação, melhor época de cada *fold*, curvas, resumo pareado, figura |
| Modelos dos *folds* | `MyDrive/cell-fuzzy-models/<experimento>/fold<k>.pt`, no Google Drive de quem roda | pesos da MarkerUNet na melhor época, em float32, com um `.json` que traz as imagens de treino e de validação do *fold*, a melhor época, a configuração e a procedência |
| Modelo final | `.../<experimento>/final.pt` | pesos da MarkerUNet treinada com as 37 imagens |
| Resultado do teste | `docs/estudo/resultados/final_<experimento>/` | métricas por imagem do teste, resumo pareado, figura |

Fontes: [report.py](../../src/evaluation/report.py) (`save_run`), [checkpoints.py](../../src/training/checkpoints.py). Só a
MarkerUNet é gravada; o ScribblePrompt vem do checkpoint dele, conferido por hash (capítulo 7, seção 7.8). Os pesos são lidos
com `torch.load(..., weights_only=True)`, que não executa código ao carregar. Gravar os 5 modelos dos *folds*, e não só o
melhor, é o que permite refazer a predição de cada imagem pelo modelo que não a viu.

Além do Dice e do IoU, a avaliação registra a precisão, a revocação, a razão de massa (capítulo 1, seção 1.6) e três medidas
do marcador: a fração de pixels acima de 0,5 e o valor médio do marcador dentro e fora dos núcleos do GT
([metrics.py](../../src/evaluation/metrics.py), `evaluate`). A diferença entre as duas médias mostra se o marcador ainda
distingue núcleo de fundo.

## 9.7 Editabilidade do marcador

O segundo critério de avaliação, depois da segmentação, é o quanto o marcador se aproxima de "um traço por núcleo", que uma
pessoa poderia aceitar ou corrigir com poucos gestos. A função [marker_editability](../../src/evaluation/editability.py) mede o
marcador **depois do corte em $\tau$**, que é o que o ScribblePrompt recebe como rabisco positivo, contra as **regiões do GT
binário** (componentes conexos; núcleos que se tocam formam uma região só):

| Medida | Definição | Ideal |
|---|---|---|
| componentes | pedaços conexos do marcador (vizinhança de 8) | um por região |
| fragmentos | componentes com menos de 20 px | nenhum |
| buracos | regiões sem marcador cercadas por marcador | nenhum |
| regiões sem marcador | fração das regiões do GT que nenhum componente toca | 0% |
| regiões com um componente | fração das regiões do GT tocadas por exatamente um componente | 100% |
| regiões partidas | fração das regiões do GT tocadas por dois ou mais componentes | 0% |
| componentes unindo regiões | componentes que tocam duas ou mais regiões do GT | nenhum |
| componentes fora | componentes que não tocam o GT | nenhum |

A medida principal é a fração de **regiões com um componente**. O limite de 20 px para fragmento é escolha deste trabalho. As
medidas são calculadas por imagem e resumidas pela média; nos notebooks atuais, para o marcador cru e para o fechado (capítulo 4,
seção 4.2). Elas não substituem um teste com usuários: medem a forma do marcador, não o esforço real de edição (capítulo 13).

## 9.8 Inferência final

Duas mudanças só de inferência, sem retreinar, melhoraram o Dice em quase todas as imagens nos testes com os modelos gravados
([09-correcoes-pontuais.md §18.23](../estudo/09-correcoes-pontuais.md)):

- **blocos com sobreposição de 1/2** em vez de 1/4: cada pixel passa a ser previsto por mais blocos, e a média ponderada fica mais
  estável (+0,0005, melhor em 32 de 37 imagens);
- **aumento no teste (TTA)**: a imagem passa pelo pipeline nas 8 combinações de rotação de 90° e espelhamento, cada saída é desfeita
  para a orientação original, e a predição é a média das 8 (+0,003, melhor em 36 de 37 imagens). É a mesma família de
  transformações da aumentação do treino (capítulo 3, seção 3.6).

Na configuração final, a validação de cada época e o registro da validação cruzada usam a sobreposição de 1/4 sem TTA; a
"melhor inferência" (sobreposição de 1/2 com TTA) é medida depois, com os modelos gravados, e é a que o modelo final usaria. As
duas estão hoje no notebook do experimento, não em `src/`.

## 9.9 Ressalvas

**Escolhas feitas nas mesmas imagens em que se mede.** A faixa do canal de fundo, a sobreposição dos blocos, o TTA e o raio do
fechamento foram escolhidos comparando opções nas mesmas 37 imagens da validação cruzada. Cada imagem é prevista por um modelo que
não a viu, mas a escolha das opções viu todas. Os ganhos dessas escolhas são, por isso, um pouco otimistas; o teste, usado uma vez
com a configuração final, é a medida limpa.

**O conjunto de teste.**

As 14 imagens de teste já foram usadas antes deste protocolo: serviram de validação em experimentos anteriores e tiveram o
Dice calculado em três experimentos da estrutura antiga (`base_dice_tv`, `dice_size`, `base_negativo`). As decisões desses
experimentos vieram da validação cruzada, mas o teste não é mais inteiramente cego. O MoNuSeg não tem outro conjunto de teste;
um teste independente exigiria um segundo conjunto de dados.

## Para o artigo

- **Métodos, treinamento:** a tabela da seção 9.1, em texto.
- **Métodos, protocolo:** validação cruzada em 5 *folds* por imagem, seleção da época, treino final pela mediana das melhores
  épocas, teste usado uma vez (seções 9.2 a 9.4).
- **Métodos, estatística:** comparação pareada com o Cellpose, intervalo por *bootstrap* e Wilcoxon (seção 9.5).
- **Métodos, avaliação do marcador:** as medidas de editabilidade (seção 9.7), com a fração de regiões com um componente como
  medida principal.
- **Métodos, inferência:** sobreposição de 1/2 e TTA (seção 9.8).
- **Limitações:** as ressalvas da seção 9.9 precisam aparecer no texto.

## Referências

- Isensee, F. et al. (2018). nnU-Net: self-adapting framework for U-Net-based medical image segmentation. arXiv:1809.10486.
  *Conferida* (seção 2.4: validação cruzada em 5 *folds* e *ensemble* das cinco redes). A versão publicada mais citada é
  Isensee et al. (2021), *Nature Methods*, 18, 203-211, *a conferir*.
- Kingma, D. P.; Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*. *A conferir.*
- Loshchilov, I.; Hutter, F. (2017). SGDR: stochastic gradient descent with warm restarts. *ICLR*. *A conferir* (agenda em
  cosseno).
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83. *A conferir.*
- Efron, B.; Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. *A conferir.*
