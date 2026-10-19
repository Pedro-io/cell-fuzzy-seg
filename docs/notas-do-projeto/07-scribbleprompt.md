# 7. Camada de segmentação: ScribblePrompt

A camada de segmentação transforma o marcador em segmentação, e é por ela que o gradiente chega à rede de marcadores
(capítulo 4). Neste trabalho, ela é o ScribblePrompt-UNet, uma rede de segmentação interativa pré-treinada e congelada, no
lugar do FMBS da tese (capítulo 2, seção 2.5). Este capítulo descreve a rede, o que ela espera receber, como o marcador é
convertido para a entrada dela, o canal de fundo, a resolução e como a rede funciona como camada diferenciável sem ser
treinada.

## 7.1 O que é o ScribblePrompt

O ScribblePrompt é uma ferramenta de segmentação interativa para imagens biomédicas: o usuário desenha rabiscos, clica ou
desenha caixas, e a rede devolve a segmentação. Ela foi feita para segmentar estruturas e tipos de imagem que não viu no
treino (Wong et al., 2024). Os pontos relevantes para este trabalho, todos conferidos no artigo:

- **Dados de treino:** 65 de uma coleção de 77 conjuntos biomédicos abertos (mais de 54 mil exames, 16 tipos de imagem),
  entre eles dois de núcleos: o BBBC038 (*Data Science Bowl* de 2018) e o PanNuke, de histologia. O MoNuSeg não está na lista
  (seções 3.3 e 4, tabela de conjuntos no apêndice).
- **Treino interativo simulado:** cada exemplo passa por cinco passos de interação simulada, com rabiscos, cliques e caixas
  gerados a partir do rótulo e correções nas regiões erradas da predição anterior (seções 3.1, 3.2 e 4).
- **Arquitetura:** uma rede convolucional de 8 camadas em forma de U, com 192 filtros por camada, ativação PReLU e sem
  normalização em lote (seção 3.4).
- **Perda de treino:** combinação de Dice suave e *focal loss* (seção 3.4).
- **Resolução:** as imagens foram reduzidas para 128 × 128 no treino, para encurtar o desenvolvimento; os autores afirmam que o
  método não depende de uma resolução específica (apêndice, *Image Resolution*).

O pacote também traz o ScribblePrompt-SAM, baseado no SAM (Kirillov et al., 2023). Ele não é usado aqui porque transforma cada pixel rabiscado num
clique, dado como uma lista de coordenadas, e essa conversão não é diferenciável em relação ao marcador (Wong et al., 2024,
seção 3.4; [08-pendencias.md, PD-39](../estudo/08-pendencias.md#pd-39)).

## 7.2 A entrada da rede

A rede recebe 5 canais, montados pela função `prepare_inputs` do pacote (conferido em `scribbleprompt/models/unet.py`):

| Canal | O que a rede espera | O que este trabalho entrega |
|---|---|---|
| 0, imagem | tons de cinza em $[0,1]$ | $0{,}299R + 0{,}587G + 0{,}114B$, dividido por 255 |
| 1, caixa | 1 dentro da caixa, 0 fora | zeros |
| 2, positivo | rabiscos e cliques sobre o objeto | o marcador convertido (seção 7.3) |
| 3, negativo | rabiscos e cliques sobre o fundo | zeros, ou o canal de fundo do Cellpose (seção 7.4) |
| 4, predição anterior | logits da interação anterior | zeros, como numa primeira interação |

Cliques entram como máscaras binárias e rabiscos como máscaras com valores em $[0,1]$ (Wong et al., 2024, apêndice B.2). A
saída é um mapa de logits, que o wrapper do projeto amplia de volta ao tamanho da entrada e passa por uma sigmoide
([scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py), `forward`).

## 7.3 Do marcador ao canal positivo

O marcador da MarkerUNet tem um canal, com valores em $(0,1)$ em todos os pixels: a sigmoide nunca dá exatamente 0. O
ScribblePrompt, por sua vez, foi treinado com interações esparsas, em que quase toda a imagem vale 0 nos canais de
interação. Entregar o marcador como está seria uma entrada fora da distribuição de treino da rede.

O efeito foi medido com marcadores tirados do ground truth: um fundo constante de apenas $10^{-3}$ no canal positivo já faz o
ScribblePrompt segmentar a mais (área prevista de 1,3 a 2,5 vezes a do GT), e a partir de $3 \times 10^{-3}$ ele marca a imagem
quase inteira ([08-pendencias.md, PD-49](../estudo/08-pendencias.md#pd-49); medido em 2026-09-28 nas imagens de treino, em
256²).

Por isso o marcador $m$ passa por um corte antes de virar o canal positivo (modo `positive`):

$$
\text{positivo} = \frac{\max(m - \tau,\ 0)}{1 - \tau}, \qquad \tau = 0{,}5
$$

Abaixo de $\tau$ o canal positivo é exatamente zero; acima, cresce linearmente até 1. Os experimentos atuais usam esse modo,
sem nenhum canal negativo derivado do marcador.

**Consequência para o gradiente.** Onde o marcador fica abaixo de $\tau$, o canal positivo é zero e a derivada dele em relação
ao marcador também é zero. Um pixel que a rede de marcadores deixou abaixo do corte não recebe nenhum sinal da perda de
segmentação para subir. Esse ponto entra na análise dos resultados (capítulo 11).

**Outros modos implementados.** O wrapper também oferece dois modos que derivam um canal negativo do próprio marcador, o
complemento dele ([scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py),
`_expand_scribble_channels`):

- `sharpened`: positivo $\sigma(T(m - 0{,}5))$ e negativo $\sigma(T(0{,}5 - m))$, com $T = 10$. É o padrão da classe;
- `dense_soft`: positivo $m$ e negativo $1 - m$.

Nesses modos, todo pixel com marcador baixo é informado à rede como fundo, inclusive o interior de um núcleo que o marcador
não cobriu: com $m = 0$, o canal negativo vale 0,993. Isso contradiz a ideia de um marcador pequeno que a camada expande até
o núcleo inteiro, e é por isso que os experimentos atuais não usam esses modos (capítulo 12).

## 7.4 O canal de fundo

No modo `positive`, a rede não recebe nenhuma indicação de fundo, e cabe a ela decidir onde cada núcleo termina. Os oráculos
mostram que essa é a principal fraqueza da camada: com marcadores pequenos e sem fundo, ela erra a extensão dos núcleos
(capítulo 10).

Como opção, um canal de fundo fixo é tirado da segmentação do Cellpose e entregue no canal negativo
([cellpose_negative](../../src/data/load/preprocessed_dataset.py)): ele vale 1 nos pixels a mais de $r$ pixels de qualquer
núcleo do Cellpose e 0 no resto, ou seja, dentro dos núcleos e numa faixa de $r$ pixels em volta deles, onde a rede decide
sozinha onde fica a borda. Os experimentos usam $r = 3$.

Esse canal é um verdadeiro marcador de fundo, mas diferente do da tese: ele não é aprendido, não passa pela MarkerUNet e não
recebe gradiente. Ele só depende da segmentação inicial. Na tese, o marcador de fundo é fixo ou proposto pela mesma rede, a
partir do complemento da segmentação intermediária (Fonseca, 2026, seção 5.4.3). A versão deste trabalho é próxima da ideia
da tese, com a diferença de que o complemento entra direto no ScribblePrompt, sem passar pela rede de marcadores.

## 7.5 Resolução

A rede trabalha em 128 × 128 por padrão, e o tamanho de entrada é configurável, desde que seja múltiplo de 16 (exigência da
arquitetura). A redução da imagem e dos canais para esse tamanho é feita pelo próprio pacote, por interpolação bilinear sem
*antialias*, e a saída é ampliada por interpolação bilinear.

- **Modo imagem inteira:** 1000² reduzida para 256². O núcleo mediano fica com cerca de 5 px de diâmetro.
- **Modo por recortes:** recortes de 256² reduzidos para 128². O núcleo mediano fica com cerca de 10 px.

**O teto imposto pela resolução.** Para saber quanto a redução sozinha limita o resultado, o próprio ground truth foi
reduzido e ampliado de volta, e o Dice foi calculado contra o original: 0,950 em 256² e 0,875 em 128², com redução bilinear
sem *antialias* como a do pacote (0,976 e 0,921 com *antialias*). Os dois tetos ficam acima do Cellpose. A resolução, sozinha,
não impede o método de superar a linha de base ([08-pendencias.md, PD-05](../estudo/08-pendencias.md#pd-05)). Atenção: esses
números foram medidos nas 14 imagens de teste com o GT da rasterização anterior à da seção 3.3 e não foram refeitos.

O *antialias* melhora o teto da redução do GT, mas piora o resultado do ScribblePrompt nos oráculos, provavelmente porque a
rede foi treinada com a redução sem *antialias* (hipótese). Por isso a redução do pacote foi mantida
([08-pendencias.md, PD-37](../estudo/08-pendencias.md#pd-37)).

## 7.6 Uma camada congelada que deixa passar o gradiente

O wrapper [ScribblePromptingNetwork](../../src/models/networks/final_segmentation/scribble_prompting_network.py) faz três coisas
para que a rede funcione como camada:

1. **Congela os pesos** (`requires_grad_(False)`) e mantém a rede sempre em modo de avaliação. Como ela não tem normalização
   em lote nem *dropout*, o modo não muda a saída.
2. **Chama a rede diretamente.** O método de predição do pacote roda sob `torch.no_grad()`, que cortaria o gradiente. O
   wrapper monta a entrada com as funções do pacote e chama a U-Net interna, de modo que o PyTorch calcula a derivada da
   saída em relação aos canais de entrada.
3. **Desliga o gradiente da imagem.** A imagem é dado de entrada; só o marcador é variável.

O caminho do gradiente fica:

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \hat{y}} \cdot
\frac{\partial \hat{y}}{\partial \text{positivo}} \cdot \frac{\partial \text{positivo}}{\partial m} \cdot
\frac{\partial m}{\partial \theta}
$$

em que $\hat{y}$ é a segmentação, $m$ o marcador e $\theta$ os pesos da MarkerUNet. O segundo fator vem da rede congelada,
calculado por diferenciação automática; o terceiro é o corte da seção 7.3. Na tese, o fator equivalente vem da derivada
explícita do FMBS, construída a partir do "elemento mais próximo" de cada pixel na hierarquia (Fonseca, 2026, seção 5.2).

**Uma consequência.** A rede de marcadores aprende o que faz o ScribblePrompt acertar, não necessariamente um marcador que um
humano desenharia. Nada impede que ela aprenda padrões que funcionam para a rede congelada sem parecerem marcadores. As
regularizações da tese são o mecanismo que puxaria o marcador para formas mais simples (capítulo 8).

## 7.7 Checkpoint

O modelo usado é o ScribblePrompt-UNet v1, baixado do endereço publicado pelos autores. O download confere o SHA-256 do
arquivo (`43f57ee8...c1acbe`) e grava primeiro num arquivo temporário, que só é renomeado se o hash bater
([scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py),
`download_checkpoint`). A conferência importa porque o pacote abre o arquivo com `torch.load` sem restringir o conteúdo, o que
executaria código embutido num arquivo adulterado.

## Para o artigo

- **Métodos:** seção 7.1 em um parágrafo (o ScribblePrompt, os dados de treino sem o MoNuSeg), a conversão da seção 7.3 com a
  fórmula e o motivo (a rede inunda com um fundo de $10^{-3}$), o canal de fundo da seção 7.4 e a frase sobre como o
  gradiente atravessa a rede congelada (seção 7.6).
- **Detalhes de implementação:** a tabela de entrada da seção 7.2 e as resoluções da seção 7.5.
- **Discussão:** o corte em $\tau$ zera o gradiente abaixo dele (seção 7.3); o marcador aprendido não precisa se parecer com um
  marcador humano (seção 7.6).
- **Ameaças à validade:** o ScribblePrompt foi treinado com o PanNuke, de histologia H&E. Hipótese a conferir: o PanNuke
  também vem do TCGA e pode ter tecido em comum com o MoNuSeg.

## Referências

- Kirillov, A. et al. (2023). Segment anything. *ICCV*. arXiv:2304.02643. *Conferida* (título e autores).
- Fonseca, G. B. da (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e
  Université Gustave Eiffel. *Conferida* (seções 5.2 e 5.4.3).
- Wong, H. E.; Rakic, M.; Guttag, J.; Dalca, A. V. (2024). ScribblePrompt: fast and flexible interactive segmentation for
  any biomedical image. arXiv:2312.07381v3. *Conferida* (seções 3.1 a 3.4 e 4, apêndices B.2 e *Image Resolution*, tabela de
  conjuntos de dados). Local de publicação definitivo *a conferir*.
