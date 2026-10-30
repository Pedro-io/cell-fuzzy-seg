# 13. Limitações e trabalhos futuros

Este capítulo junta as limitações espalhadas pelos capítulos anteriores e as direções de continuação. Ele serve de base às
seções de limitações e de trabalhos futuros do texto. As limitações estão agrupadas pelo que afetam: os dados, a avaliação, o
método e os resultados.

## 13.1 Limitações

**Dados**

- **Um único conjunto de dados, e pequeno.** São 37 imagens de treino e 14 de teste, todas do MoNuSeg. Não há como afirmar que
  os resultados valem para outros tecidos, colorações ou digitalizadores (capítulo 3).
- **Treino e teste com perfis diferentes.** O treino tem imagens mais densas e as três imagens em 20×; o teste não tem nenhuma
  imagem em 20× e tem dois órgãos que o treino não tem (capítulo 3, seções 3.1 e 3.4).

**Avaliação**

- **O teste não era inteiramente cego.** As 14 imagens de teste serviram de validação em experimentos anteriores e tiveram o Dice
  calculado em três experimentos da estrutura antiga. A configuração final foi escolhida pela validação cruzada e avaliada no teste
  uma única vez (capítulo 9, seção 9.9; capítulo 11, seção 11.7).
- **Escolhas feitas nas mesmas imagens em que se mede.** A faixa do canal de fundo, o raio do fechamento, a sobreposição dos blocos
  e o TTA foram escolhidos comparando opções nas 37 imagens da validação cruzada. Os ganhos dessas escolhas são um pouco otimistas
  (capítulo 9, seção 9.9).
- **Duas mudanças numa rodada só.** A configuração final trocou a faixa do canal de fundo e acrescentou o fechamento no mesmo
  treino, por limite de execuções no Colab; o efeito de cada uma no treino não foi separado (capítulo 12, D-47).
- **Avaliação semântica.** A máscara binária não separa núcleos que se tocam, e os números não são comparáveis aos do desafio
  MoNuSeg, que usa métricas por instância (capítulo 1, seção 1.5).
- **A linha de base já viu as imagens de treino.** O MoNuSeg faz parte dos dados de treino do Cellpose-SAM. Na validação
  cruzada, tanto a linha de base quanto a entrada da MarkerUNet vêm de imagens que o Cellpose viu; no teste, não (capítulo 5,
  seção 5.4).
- **Linha de base única.** A comparação é só com o Cellpose, a própria entrada do método. Não há comparação com um modelo
  totalmente supervisionado treinado no MoNuSeg, que a tese sugere para este cenário (capítulo 2, seção 2.4).
- **A editabilidade é medida pela forma, não pelo uso.** As medidas de editabilidade contam componentes, buracos e fragmentos do
  marcador (capítulo 9, seção 9.7). Elas não medem quanto esforço uma pessoa gastaria para corrigir o marcador; isso exigiria a
  edição simulada com cliques ou um estudo com usuários (seção 13.2).
- **Resultados ainda não salvos em arquivo.** O oráculo de resolução e o diagnóstico dos marcadores aprendidos só existem nas
  saídas dos notebooks
  ([08-pendencias.md, PD-47](../estudo/08-pendencias.md#pd-47)).

**Método**

- **O marcador deixa de explicar a segmentação.** No FMBS, a segmentação é inteiramente determinada pelo marcador e pela
  hierarquia, e editar o marcador muda o resultado de forma previsível. O ScribblePrompt é uma rede aprendida: ela segmenta
  regiões que o marcador não cobre e reage a valores suaves de um jeito que não se controla. O marcador aprendido é o que faz a
  rede acertar, não necessariamente um marcador que um humano desenharia ou editaria (capítulo 2, seção 2.5; capítulo 7,
  seção 7.6). O fechamento melhora a forma do marcador, mas não devolve essa relação determinística.
- **Entrada fora da distribuição da camada.** O ScribblePrompt foi treinado com rabiscos e cliques esparsos, numa sequência de
  interações. Aqui ele recebe um mapa contínuo, numa única interação. O corte em $\tau$ reduz o problema, mas os marcadores
  aprendidos ainda têm valores suaves que a rede expande demais (capítulo 11, seção 11.5).
- **Gradiente nulo abaixo do corte.** Onde o marcador fica abaixo de $\tau$, a perda de segmentação não envia nenhum sinal
  (capítulo 7, seção 7.3).
- **Dependência do Cellpose.** A entrada da rede de marcadores e o canal de fundo vêm do Cellpose. Os erros dele, em especial os
  núcleos que ele não encontra, limitam o que o pipeline pode recuperar; e quanto mais estreita a faixa do canal de fundo, mais a
  saída se parece com a máscara do Cellpose (capítulo 7, seção 7.4; capítulo 10, seção 10.2).
- **Resolução da camada.** No modo por recortes, o ScribblePrompt trabalha em 128², com o núcleo mediano em cerca de 10 px; o
  espaço de 1 a 2 px entre núcleos vizinhos pode sumir, o que favorece a fusão de núcleos densos (hipótese; capítulo 11, seção
  11.7).
- **Textura do marcador cru.** O marcador que a MarkerUNet produz tem uma textura em grade, de origem não verificada, que gera a
  maior parte dos buracos; o fechamento a apaga, mas não a explica (capítulo 11, seção 11.4).
- **Escolhas sem ablação.** Ainda não foram medidos: a padronização do ImageNet, o congelamento do codificador, a probabilidade
  do Cellpose como entrada e a normalização das regularizações pelo lote em vez de por objeto (capítulos 6 e 8).
- **Oráculos idealizados.** Usam marcadores binários e uma única interação; servem de teto, não de previsão (capítulo 10, seção
  10.4).

**Resultados**

- **O pipeline não supera o Cellpose.** No teste, a configuração final fica 0,0037 abaixo dele (0,8333 contra 0,8370), melhor em 3
  das 14 imagens. O Wilcoxon não distingue a diferença de zero (p = 0,12), mas o intervalo por *bootstrap* fica todo abaixo de
  zero, por pouco (−0,0071 a −0,0002). Com 14 imagens, o resultado é "no nível do Cellpose", não "igual" nem "melhor"
  (capítulo 11, seção 11.7).
- **Treino final de 200 épocas sem validação.** Os *folds* pararam entre as épocas 23 e 99; o treino final foi até 200, decisão
  tomada antes do teste para completar a agenda em cosseno. A massa no teste (1,15) ficou maior que na validação cruzada (1,06);
  hipótese: efeito das épocas a mais. O modelo de 67 épocas existe, mas avaliá-lo seria olhar o teste uma segunda vez (capítulo 9,
  seção 9.4; capítulo 12, D-53 e D-55).
- **Validação cruzada da configuração final sem resultados por imagem.** Só os números agregados ficaram, das saídas do notebook;
  os CSVs por imagem precisam ser regerados dos modelos dos *folds* para as comparações pareadas (capítulo 11, seção 11.8).

## 13.2 Trabalhos futuros

**Edição com cliques (segundo artigo)**

A edição do marcador por uma pessoa é o objetivo original da tese e fica para um segundo artigo (capítulo 12, D-51). O desenho
está guardado em [09-correcoes-pontuais.md §18.27](../estudo/09-correcoes-pontuais.md) e não foi implementado:

- **entrada:** cliques positivos e negativos como máscaras binárias, que estão dentro da distribuição de treino do ScribblePrompt
  (Wong et al., 2024, apêndice B.2), aplicados depois do fechamento e com prioridade sobre o marcador e o canal de fundo;
- **usuário simulado:** a cada passo, cliques no centro dos maiores erros, positivos em falsos negativos e negativos em falsos
  positivos, como no protocolo de cliques no centro do erro do artigo do ScribblePrompt (seção 5.2), com vários cliques por passo
  porque há centenas de núcleos por imagem;
- **medida:** a curva de Dice por número de cliques, com e sem o marcador, e quantos cliques são necessários para passar do
  Cellpose. A diferença entre as duas condições mede quantos cliques o marcador economiza.

**Sobre a camada de segmentação**

- **Uma camada que faça o papel do FMBS de forma mais fiel.** A troca do FMBS por uma rede interativa genérica trouxe a perda da
  relação determinística entre marcador e segmentação (seção 13.1). Uma direção é uma camada de segmentação mais próxima do FMBS,
  em que a segmentação dependa do marcador e da imagem por regras simples, ou o próprio FMBS como camada, para comparação direta
  com a tese. O segundo artigo pode usar uma camada mais complexa que o ScribblePrompt-UNet.
- **Ajustar a camada ao domínio.** Treinar ou ajustar o ScribblePrompt com marcadores no formato que a MarkerUNet produz, ou
  treinar as duas redes juntas. O método deixaria de usar uma camada congelada, e a pergunta passaria a ser outra.
- **Resolução maior com canal de fundo.** No oráculo, a entrada em 1024² com o fundo do Cellpose levou marcadores pequenos
  ideais acima do Cellpose (capítulo 10, seção 10.3). Falta testar isso no treino.

Repetir o ScribblePrompt realimentando só a própria saída já foi testado e piorou (capítulo 7, seção 7.7); as passadas sucessivas
só fazem sentido com correções novas, que é o caso dos cliques.

**Sobre o marcador**

- **Morfologia como perda ou como camada treinável.** A tese propõe perdas baseadas em operadores morfológicos, como uma perda de
  fechamento que favoreça marcadores sem lacunas (Fonseca, 2026, seção 7.1.4); aqui o fechamento é uma operação fixa. Variações:
  um raio aprendido, camadas morfológicas treináveis (Franchi, Fehri e Yao, 2020) ou morfologia de conjuntos difusos aplicada ao
  marcador difuso (Bloch e Maître, 1995).
- **Marcador quase binário.** Binarizar o marcador no passo para frente e manter o gradiente contínuo (*straight-through*), para
  que a camada receba algo parecido com rabiscos 0 ou 1.
- **Um termo de componentes.** Um equivalente do $L_{comp}$ adaptado a muitos objetos, que favoreça um componente por núcleo, para
  equilibrar o Size e o Dmap, que hoje picotam o marcador. O fechamento de 3 px dentro do treino, com o Size e o Dmap sobre o
  marcador cru, não cumpriu esse papel (capítulo 8, seção 8.3; capítulo 11, seção 11.4).
- **Regularizar o marcador fechado.** Calcular o Size e o Dmap sobre o marcador depois do fechamento e do corte em $\tau$, que é o que
  o ScribblePrompt recebe, para que a perda cobre a forma que de fato importa.
- **Marcador de fundo aprendido.** Na tese, o marcador de fundo é proposto pela mesma rede, a partir do complemento da
  segmentação intermediária (Fonseca, 2026, seção 5.4.3). Aqui, o canal de fundo é fixo e vem direto do Cellpose.
- **Segmentação por instância a partir dos marcadores.** Com um componente de marcador por núcleo, que o fechamento já aproxima,
  os marcadores poderiam servir de sementes para separar núcleos que se tocam, o que permitiria avaliar por instância (AJI, PQ) e
  comparar com o desafio.
- **A origem da textura em grade** do marcador cru (seção 13.1).

**Sobre a avaliação**

- **Comparação com um modelo supervisionado** treinado no MoNuSeg, como sugere a tese (seção 7.1.2).
- **Uma linha de base do Cellpose que não viu o MoNuSeg**, para medir quanto a linha de base atual está otimista na validação
  cruzada.
- **Outros conjuntos de dados** de núcleos em H&E, para um teste realmente independente. Antes, conferir se há tecido em comum
  entre esses conjuntos, o MoNuSeg e os dados de treino do ScribblePrompt (o PanNuke, por exemplo).
- **Ensemble dos *folds*** em vez do modelo final (capítulo 9, seção 9.4).

## Para o artigo

- **Limitações:** seção 13.1, em especial o teste já visto, as escolhas feitas nas mesmas imagens, a linha de base que viu as
  imagens de treino, a editabilidade medida pela forma e a perda da relação determinística entre marcador e segmentação.
- **Trabalhos futuros:** a edição com cliques (o segundo artigo), a camada mais fiel ao FMBS, a morfologia como perda ou camada
  treinável e a segmentação por instância.

## Referências

- Bloch, I.; Maître, H. (1995). Fuzzy mathematical morphologies: a comparative study. *Pattern Recognition*, 28(9), 1341-1387.
  doi:10.1016/0031-3203(94)00312-A. *A conferir.*
- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seções 5.4.3, 7.1.1, 7.1.2 e 7.1.4).
- Franchi, G.; Fehri, A.; Yao, A. (2020). Deep morphological networks. *Pattern Recognition*, 102, 107246.
  doi:10.1016/j.patcog.2020.107246. *A conferir.*
- Wong, H. E.; Rakic, M.; Guttag, J.; Dalca, A. V. (2024). ScribblePrompt: fast and flexible interactive segmentation for
  any biomedical image. arXiv:2312.07381v3. *Conferida* (seção 5.2 e apêndice B.2).
