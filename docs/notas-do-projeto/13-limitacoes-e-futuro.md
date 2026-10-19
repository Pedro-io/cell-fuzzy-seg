# 13. Limitações e trabalhos futuros

Este capítulo junta as limitações espalhadas pelos capítulos anteriores e as direções de continuação. Ele serve de base às
seções de limitações e de trabalhos futuros do texto. As limitações estão agrupadas pelo que afetam: os dados, a avaliação, o
método e os resultados.

## 13.1 Limitações

**Dados**

- **Um único conjunto de dados, e pequeno.** São 37 imagens de treino e 14 de teste, todas do MoNuSeg. Não há como afirmar que
  os resultados valem para outros tecidos, colorações ou digitalizadores (capítulo 3).
- **Treino e teste com perfis diferentes.** O treino tem imagens mais densas e as três imagens em 20×; o teste não tem nenhuma
  (capítulo 3, seção 3.4).

**Avaliação**

- **O teste já foi visto.** As 14 imagens de teste serviram de validação em experimentos anteriores e tiveram o Dice calculado
  em três experimentos da estrutura antiga. Um teste realmente cego exigiria outro conjunto de dados (capítulo 9, seção 9.7).
- **Avaliação semântica.** A máscara binária não separa núcleos que se tocam, e os números não são comparáveis aos do desafio
  MoNuSeg, que usa métricas por instância (capítulo 1, seção 1.5).
- **A linha de base já viu as imagens de treino.** O MoNuSeg faz parte dos dados de treino do Cellpose-SAM. Na validação
  cruzada, tanto a linha de base quanto a entrada da MarkerUNet vêm de imagens que o Cellpose viu; no teste, não (capítulo 5,
  seção 5.4).
- **Linha de base única.** A comparação é só com o Cellpose, a própria entrada do método. Não há comparação com um modelo
  totalmente supervisionado treinado no MoNuSeg, que a tese sugere para este cenário (capítulo 2, seção 2.4).
- **Resultados ainda não salvos em arquivo.** O oráculo de resolução e o diagnóstico dos marcadores aprendidos só têm os
  resultados nas saídas dos notebooks ([08-pendencias.md, PD-47](../estudo/08-pendencias.md#pd-47)).

**Método**

- **O marcador deixa de explicar a segmentação.** No FMBS, a segmentação é inteiramente determinada pelo marcador e pela
  hierarquia, e editar o marcador muda o resultado de forma previsível. O ScribblePrompt é uma rede aprendida: ela segmenta
  regiões que o marcador não cobre e reage a valores suaves de um jeito que não se controla. O marcador aprendido é o que faz a
  rede acertar, não necessariamente um marcador que um humano desenharia ou editaria (capítulo 2, seção 2.5; capítulo 7,
  seção 7.6).
- **Entrada fora da distribuição da camada.** O ScribblePrompt foi treinado com rabiscos e cliques esparsos, numa sequência de
  interações. Aqui ele recebe um mapa contínuo, numa única interação. O corte em $\tau$ reduz o problema, mas os marcadores
  aprendidos ainda têm valores suaves que a rede expande demais (capítulo 11, seção 11.4).
- **Gradiente nulo abaixo do corte.** Onde o marcador fica abaixo de $\tau$, a perda de segmentação não envia nenhum sinal
  (capítulo 7, seção 7.3).
- **Dependência do Cellpose.** A entrada da rede de marcadores e o canal de fundo vêm do Cellpose. Os erros dele, em especial os
  núcleos que ele não encontra, limitam o que o pipeline pode recuperar (capítulo 10, seção 10.2).
- **Resolução.** No modo imagem inteira, as duas redes trabalham em 256², com o núcleo mediano em cerca de 5 px (capítulo 4,
  seção 4.3).
- **Escolhas sem ablação.** Ainda não foram medidos: a padronização do ImageNet, o congelamento do codificador, a probabilidade
  do Cellpose como entrada, a normalização das regularizações pelo lote em vez de por objeto e a largura da faixa livre do canal
  de fundo (capítulos 6, 7 e 8).
- **Oráculos idealizados.** Usam marcadores binários e uma única interação; servem de teto, não de previsão (capítulo 10, seção
  10.4).

**Resultados**

- Até 2026-10-17, nenhuma configuração treinada superou o Cellpose (capítulo 11).

## 13.2 Trabalhos futuros

**Sobre a camada de segmentação**

- **Uma camada que faça o papel do FMBS de forma mais fiel.** A troca do FMBS por uma rede interativa genérica trouxe a perda da
  relação determinística entre marcador e segmentação (seção 13.1). Uma direção é uma camada de segmentação neural mais próxima
  do FMBS: treinada para que a segmentação dependa só do marcador e da imagem por regras simples, por exemplo propagando o
  marcador por regiões, ou o próprio FMBS como camada, para comparação direta com a tese.
- **Ajustar a camada ao domínio.** Treinar ou ajustar o ScribblePrompt com marcadores no formato que a MarkerUNet produz, ou
  treinar as duas redes juntas. O método deixaria de usar uma camada congelada, e a pergunta passaria a ser outra.
- **Usar as interações sucessivas.** O ScribblePrompt aceita a predição anterior como entrada. Um esquema em duas ou mais
  passadas, com a primeira predição realimentada, ficaria mais perto da forma como a rede foi treinada.
- **Resolução maior com canal de fundo.** No oráculo, a entrada em 1024² com o fundo do Cellpose levou marcadores pequenos
  ideais acima do Cellpose (capítulo 10, seção 10.3). Falta testar isso no treino.

**Sobre o marcador**

- **Marcador quase binário.** Binarizar o marcador no passo para frente e manter o gradiente contínuo (*straight-through*), para
  que a camada receba algo parecido com rabiscos 0 ou 1 (capítulo 11, seção 11.5).
- **Marcador de fundo aprendido.** Na tese, o marcador de fundo é proposto pela mesma rede, a partir do complemento da
  segmentação intermediária (Fonseca, 2026, seção 5.4.3). Aqui, o canal de fundo é fixo e vem direto do Cellpose.
- **Segmentação por instância a partir dos marcadores.** Com um componente de marcador por núcleo, os marcadores poderiam servir
  de sementes para separar núcleos que se tocam, o que permitiria avaliar por instância (AJI, PQ) e comparar com o desafio.
  Hipótese: exigiria uma regularização que favoreça um componente por núcleo, o oposto do $L_{comp}$ da tese.

**Sobre a avaliação**

- **Comparação com um modelo supervisionado** treinado no MoNuSeg, como sugere a tese (seção 7.1.2).
- **Uma linha de base do Cellpose que não viu o MoNuSeg**, para medir quanto a linha de base atual está otimista na validação
  cruzada.
- **Outros conjuntos de dados** de núcleos em H&E, para um teste realmente independente. Antes, conferir se há tecido em comum
  entre esses conjuntos, o MoNuSeg e os dados de treino do ScribblePrompt (o PanNuke, por exemplo).
- **Ensemble dos *folds*** em vez do modelo final (capítulo 9, seção 9.4).
- **O usuário no laço.** Medir se os marcadores propostos reduzem o esforço de um anotador que os corrige, que é o objetivo
  original da tese (Fonseca, 2026, seção 7.1.1).

## Para o artigo

- **Limitações:** seção 13.1, em especial o teste já visto, a avaliação semântica, um único conjunto de dados e a perda da
  relação determinística entre marcador e segmentação.
- **Trabalhos futuros:** a camada mais fiel ao FMBS, a resolução maior com canal de fundo, o marcador quase binário e a
  segmentação por instância.

## Referências

- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seções 5.4.3, 7.1.1 e 7.1.2).
