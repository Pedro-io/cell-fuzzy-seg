# 03 — Modelos: MarkerUNet e ScribblePrompt

> Legenda: ✅ verificado no código (do projeto ou das bibliotecas) ou medido · 📜 histórico · 📖 artigo/tese · ❓ a confirmar.
> Problemas ficam em [08-pendencias.md](08-pendencias.md) e são citados pelo ID.
>
> Última verificação: 2026-09-27 (commit `f9b1e9b`). As bibliotecas foram conferidas no código-fonte das versões fixadas no
> `requirements.txt`: `segmentation-models-pytorch==0.5.0` (`encoders/_utils.py`) e `scribbleprompt` no commit `182c449`
> (`models/unet.py` e `models/network.py`).

---

## 1. Resumo

```
rgba (B,4,1000,1000) ─► resize 256 ─► MarkerUNet (U-Net ResNet34, 24,4 M parâmetros, TREINÁVEL) ─► sigmoid ─► resize 1000
                                                                                                          │
                                                                                                  markers (B,1,1000,1000)
                                                                                                          │
image (B,3,1000,1000) ─► tons de cinza ─┐                         pos = σ(10·(m−0,5)) ; neg = σ(10·(0,5−m))
                                        ├─► resize 128 ─► [img, 0, pos, neg, 0] ─► ScribblePrompt-UNet (CONGELADA)
                                        │                                               │
                                        │                                   logits 128² ─► resize 1000 ─► sigmoid
                                                                                                          │
                                                                                          segmentation final (B,1,1000,1000)
```

- A **MarkerUNet** é a única rede treinada: uma U-Net padrão do `smp` com encoder ResNet34 pré-treinado no ImageNet e
  entrada de 4 canais. ✅
- O **ScribblePrompt** é uma rede de segmentação interativa biomédica, usada **congelada**. O gradiente atravessa a rede até
  os scribbles, e daí até a MarkerUNet. ✅
- Três achados deste tema:
  - O canal negativo marca como "fundo" **todo pixel não marcado, inclusive o interior dos núcleos**. Isso vai contra o
    L_size, que quer marcadores pequenos (reforça o PD-04).
  - As reduções de resolução não usam *antialias* (PD-37).
  - A entrada da MarkerUNet não recebe a normalização que o encoder ImageNet espera (PD-36).
- A resolução de 128² **não impede sozinha** de superar o Cellpose: uma predição perfeita em 128² ainda daria Dice
  ~0,88–0,92 (PD-05, atualizado).

---

## 2. MarkerUNet (a rede de marcadores)

### 2.1 O que é ✅ — [marker_unet.py](../../src/models/networks/marker_unet.py), [marker_unet_config.py](../../src/models/configs/marker_unet_config.py)

```python
UNET_CONFIG = {"encoder_name": "resnet34", "encoder_depth": 5, "encoder_weights": "imagenet",
               "in_channels": 4, "classes": 1, "activation": None}
class MarkerUNet(BaseNetwork):
    def __init__(self, config=UNET_CONFIG): self.model = smp.Unet(**config)
    def forward(self, x):  return self.model(x)                              # logits
    def predict(self, x):  with no_grad: return sigmoid(self.model(x))       # probabilidades
```

- **U-Net** 📖: um encoder reduz a resolução em 5 estágios, extraindo *features* cada vez mais abstratas; um decoder volta à
  resolução original. As *skip connections* levam os detalhes finos do encoder ao decoder.
- **Encoder ResNet34** com pesos do ImageNet: só o encoder vem pré-treinado; o **decoder começa aleatório**. 📖 A tese usa a
  mesma família: ResNet18 na prova de conceito e ResNet34 no modelo de objeto (5.4.1–5.4.2).
- **24.439.505 parâmetros**, todos treináveis (valor impresso no exp. 6). Nada é congelado, nem o encoder, e o treino usa
  30 imagens (PD-38).
- `activation=None`: a saída são **logits**. O sigmoid é aplicado no `MarkerStep`.

### 2.2 Como o ImageNet vira 4 canais ✅ (conferido no `smp` 0.5.0)

Os pesos do ImageNet esperam 3 canais (RGB). Com `in_channels=4`, a função `patch_first_conv` do smp **copia os pesos em
ciclo** (`peso_novo[:, i] = peso[:, i % 3]`) e multiplica tudo por `3/4`:

| Canal de entrada | Pesos iniciais da 1ª convolução |
|---|---|
| R | pesos de R × 0,75 |
| G | pesos de G × 0,75 |
| B | pesos de B × 0,75 |
| **alpha (máscara do Cellpose)** | **cópia dos pesos de R** × 0,75 |

No início do treino, a máscara do Cellpose é "vista" como se fosse mais vermelho. O fator 3/4 mantém a escala da ativação
parecida com a original. (A docstring da função no smp diz que canais > 3 recebem "inicialização aleatória", mas o código faz
a cópia descrita acima.)

### 2.3 Entrada e normalização ✅

O `MarkerStep` entrega `rgba` em **[0, 1]**, sem subtrair média nem dividir por desvio. Os pesos do ImageNet foram
treinados com entradas normalizadas (média ~0,45, desvio ~0,22 por canal, isto é, valores em torno de [−2, 2]); o smp
oferece `get_preprocessing_fn` para isso, e o projeto não usa. As BatchNorms do encoder compensam parte dessa diferença
durante o *fine-tuning*, mas o ponto de partida pré-treinado fica menos aproveitado (PD-36). ❓ O tamanho do impacto só se
sabe testando.

### 2.4 Como o `MarkerStep` a executa ✅ — [marker_step.py](../../src/pipeline/steps/inference/marker_step.py)

| | Treino (`differentiable=True`) | Inferência (`differentiable=False`) |
|---|---|---|
| Entrada | tensor (B,4,H,W) | array (H,W,4) ou tensor |
| Redimensionamento | `F.interpolate(bilinear)` para 256² e de volta | `cv2.resize` para 256² e de volta |
| Saída | probabilidades [0,1], **com grafo** | **binarizada** com limiar 0,5, NumPy |
| Modo da rede | `train()` | `eval()` |

- Código: [L177-215](../../src/pipeline/steps/inference/marker_step.py#L177-L215) (treino) e
  [L145-175](../../src/pipeline/steps/inference/marker_step.py#L145-L175) (inferência).
- Os notebooks usam **só o modo de treino**, inclusive na validação, via `Trainer.eval_step` com `no_grad`.
- `set_training` ([L90-93](../../src/pipeline/steps/inference/marker_step.py#L90-L93)) é chamado pelo `Trainer` para
  alternar `train()`/`eval()`. Isso só existe desde 22/09 (PD-09).
- Por que 256² 📜: é o "P7" da investigação do exp. 1 (reduzir memória e custo), e ficou como `target_size` fixo.
- `_to_tensor_bchw` ([L217-261](../../src/pipeline/steps/inference/marker_step.py#L217-L261)) tenta adivinhar o layout do
  tensor e divide por 255 quando o máximo passa de 1. Funciona para os dados atuais, mas é frágil para entradas diferentes.

---

## 3. ScribblePrompt (a rede final congelada)

### 3.1 O que é 📖 — Wong et al., *ScribblePrompt* (arXiv 2312.07381v3), em `docs/papers/scribble_prompt.pdf`

- É uma rede de **segmentação interativa** para imagens biomédicas: o usuário faz rabiscos, cliques ou caixas, e a rede
  segmenta. Ela é pensada para estruturas e modalidades **não vistas no treino** (seções 1 e 3).
- Foi treinada em **77 datasets** (54 mil exames, 16 tipos de imagem), **incluindo datasets de células** (seção 3.3), com
  rótulos sintéticos (superpixels) para generalizar.
- **Treino interativo simulado** (seções 3.1–3.2): 5 passos por exemplo, até 3 interações por passo. Os rabiscos são
  simulados como linhas, esqueletos ou contornos **finos e esparsos**. A predição anterior entra como 5º canal.
- **Arquitetura** (seção 3.4 e `network.py`): UNet de 8 camadas convolucionais, 192 filtros, ativação PReLU, **sem
  BatchNorm** (o apêndice mostra que foi a melhor opção), com MaxPool. Loss de treino: Dice + Focal.
- **Resolução:** treinada em **128×128** ("para reduzir o tempo de treino"; o método "não é preso a uma resolução"). O artigo
  também avalia em 256².
- Existe uma variante **ScribblePrompt-SAM** (ViT-b, 1024²) no mesmo pacote, que o projeto não usa. ❓ Pode ser uma
  alternativa de rede final com resolução maior.

### 3.2 A entrada de 5 canais ✅ (conferido em `scribbleprompt/models/unet.py`)

`prepare_inputs` monta `x = [img, box, pos, neg, mask_input]`, todos em (B,·,128,128):

| Canal | O que o ScribblePrompt espera (artigo, apêndice) | O que o projeto entrega |
|---|---|---|
| 0: imagem | tons de cinza em [0,1] | `0,299R + 0,587G + 0,114B`, dividido por 255 ✅ |
| 1: caixa | 1 dentro da caixa | **zeros** |
| 2: positivo | cliques/rabiscos positivos em [0,1], esparsos | `σ(10·(m − 0,5))` sobre **a imagem inteira** |
| 3: negativo | cliques/rabiscos negativos em [0,1], esparsos | `σ(10·(0,5 − m))` sobre **a imagem inteira** |
| 4: predição anterior | logits da interação anterior (zeros na 1ª) | **zeros**, como se fosse a 1ª interação |

`rescale_inputs` reduz imagem e scribbles com `F.interpolate(mode='bilinear')` **sem antialias**, e a saída de 128² volta a
1000² por bilinear no wrapper ([scribble_prompting_network.py:207-208](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L207-L208)).

### 3.3 O wrapper do projeto ✅ — [scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py)

- **Congelamento:** `self.unet.requires_grad_(False)` + `eval()`
  ([L151-155](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L151-L155)). O `train()` é
  sobrescrito para ficar sempre em `eval()` ([L163-171](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L163-L171)).
  Como a rede não tem BatchNorm nem Dropout, `train`/`eval` nem mudaria o resultado.
- **Por que não usar `ScribblePromptUNet.predict`:** ele é `@torch.no_grad()` e cortaria o gradiente. O wrapper chama
  `self.unet(x)` direto ([L205](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L205)). É isso
  que faz a rede congelada funcionar como "camada de loss": os pesos não mudam, mas a derivada da saída **em relação aos
  scribbles** existe.
- **Checkpoint:** `download_checkpoint` baixa `ScribblePrompt_unet_v1_nf192_res128.pt` do Dropbox sem conferir hash, e o
  pacote faz `torch.load` (PD-15). Com `checkpoint=...`, o wrapper sobrescreve `ScribblePromptUNet.weights`, um dicionário
  **da classe** ([L135-140](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L135-L140)); o
  comentário no código já registra que isso afeta outras instâncias.
- **Imagem:** vira tons de cinza e é dividida por 255 se o máximo passar de 1
  ([L224-276](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L224-L276)). Tensores de imagem são
  `detach()`ados; a imagem é dado, não variável.

### 3.4 De marcador (1 canal) para scribbles (2 canais) ✅

[`_expand_scribble_channels`](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L331-L357):

| Marcador `m` | `pos = σ(10(m−0,5))` | `neg = σ(10(0,5−m))` | ∂pos/∂m |
|---|---|---|---|
| 0,00 | 0,007 | **0,993** | 0,07 |
| 0,25 | 0,076 | 0,924 | 0,70 |
| 0,50 | 0,500 | 0,500 | **2,50** |
| 0,75 | 0,924 | 0,076 | 0,70 |
| 1,00 | 0,993 | 0,007 | 0,07 |

Leituras:
1. **Todo pixel com marcador baixo vira "rabisco negativo" quase cheio**, inclusive o interior de um núcleo que o marcador
   não cobriu. Na tese, o marcador de objeto é pequeno e o que não é marcado é decidido pela hierarquia. Aqui, "não
   marcado" é dito ao ScribblePrompt como "**isto é fundo**". Isso cria um conflito direto: o L_size empurra o marcador para
   ficar pequeno, o que aumenta a área negativa sobre os núcleos e faz o ScribblePrompt segmentar menos; o Dice empurra na
   direção contrária. ❓ Pode ser uma das causas do PD-01. Ver PD-04.
2. **O gradiente satura nos extremos:** com m perto de 0 ou 1, a derivada é ~0,07, cerca de 38× menor que no meio. Um marcador
   "decidido" quase para de aprender. `scribble_temperature` controla esse compromisso: T menor dá mais gradiente e
   scribbles menos binários.
3. `dense_soft` (`[m, 1−m]`, exp. 5) tem os mesmos canais densos, só que sem o sharpening. No exp. 5 ele foi pior
   (Dice 0,598 contra 0,648 do exp. 4). **A temperatura é ignorada nesse modo.**

---

## 4. Resolução: o que 128² permite e o que custa ✅ (medido)

A pergunta aqui: se a MarkerUNet fosse perfeita, quanto a cadeia de resoluções ainda limitaria o Dice? Para medir, reduzi o
**GT** a n² e voltei a 1000² (bilinear, limiar 0,5), calculando o Dice contra o GT original:

| Resolução intermediária | Redução com antialias (`INTER_AREA`) | Redução sem antialias (bilinear, como o código faz) |
|---|---|---|
| 256² | 0,976 | 0,950 |
| **128²** | **0,921** | **0,875** |

Números das 14 imagens de teste; no treino dá 0,919 / 0,879. Os pixels realmente lidos numa redução bilinear sem antialias:
1000→256 lê 512 de 1000 índices por eixo (26% dos pixels); **1000→128 lê 256 por eixo (6,6% dos pixels)**. O resto da
imagem e do marcador é ignorado pelo ScribblePrompt.

O que isso quer dizer:
- **A resolução não explica sozinha o PD-01.** Mesmo em 128², o teto (0,875–0,921) fica acima do Cellpose (0,810). Isso
  corrige a leitura anterior do PD-05, que tratava a resolução como provável causa principal.
- **A falta de antialias custa ~5 pontos de teto** em 128². Corrigir é barato: `F.interpolate(..., antialias=True)` ou
  `mode='area'` antes do ScribblePrompt. Mas a redução é feita dentro do pacote (`rescale_inputs`), então o wrapper teria de
  reduzir antes de chamá-la (PD-37).
- **O gradiente não fica esparso na MarkerUNet:** como o marcador de 1000² é uma ampliação do mapa de 256², as posições lidas
  pelo ScribblePrompt ainda alcançam 100% dos pixels de 256² (calculado a partir dos pesos da interpolação bilinear).
- A ScribblePrompt não enxerga detalhes menores que ~8 px da imagem original. O núcleo mediano (~24 px) fica com ~3 px; os
  núcleos das imagens em 20× ficam com ~1,5 px (PD-24).

---

## 5. Como a MarkerUNet é treinada através de uma rede congelada 📖✅

Na tese, a camada de segmentação (FMBS) é diferenciável **por construção**: a derivada da conexão difusa em relação ao
marcador é calculada explicitamente (cap. 5.2, "closest elements"). Aqui a camada é uma rede neural congelada, e o PyTorch
calcula a derivada por *autograd*:

```
L_seg(ScribblePrompt(img, pos(m), neg(m)), GT)
   └── ∂L/∂pos, ∂L/∂neg (via autograd, pesos fixos) ──► ∂pos/∂m, ∂neg/∂m (sigmoid) ──► ∂m/∂θ (MarkerUNet)
```

Diferenças que importam para o texto do TCC:
- **Interpretabilidade:** no FMBS, a segmentação é uma função determinística e explicável do marcador e da hierarquia. No
  ScribblePrompt, é o que uma rede faria com aqueles scribbles. O marcador aprendido é "o que convence o ScribblePrompt",
  não necessariamente um marcador que um humano editaria (ver [00-visao-geral.md](00-visao-geral.md), §3.1).
- **Distribuição de entrada:** o ScribblePrompt só foi treinado com rabiscos esparsos e poucas interações. A MarkerUNet pode
  aprender entradas "adversariais": padrões densos que produzem uma boa saída sem parecer marcadores. Os regularizadores
  (Size, DMap, TV) são o que puxa na direção de marcadores razoáveis.
- **Teste de sanidade que falta:** o E3 da investigação, o "oráculo". Entregar ao ScribblePrompt, sem MarkerUNet, versões do
  GT como marcador (o GT inteiro, o GT erodido, o esqueleto do GT, o centro de cada núcleo) e medir o Dice. Isso dá o **teto
  real do ScribblePrompt neste pipeline**, com a conversão pos/neg atual, e diz se vale continuar com ele (PD-03).

---

## 6. `BaseNetwork` e `BaseFinalSegmentation` ✅

- [BaseNetwork](../../src/models/networks/base_networks.py): um `nn.Module` abstrato com `forward` e `predict` obrigatórios
  e `get_config()`. Herdar de `nn.Module` dá `.to()`, `.parameters()`, `.state_dict()`, `.train()`/`.eval()`.
- [BaseFinalSegmentation](../../src/models/networks/final_segmentation/base_final_segmentation.py): a interface da rede
  final (padrão Strategy). O contrato é `forward({"image", "scribbles"}) → (N,1,H,W)`. É o ponto de troca que já recebeu
  watershed, U-Net congelada e ScribblePrompt, e é por ele que outra rede final entraria.

---

## 7. Dúvidas para você validar

**Respondidas (2026-09-27):**
- **Pergunta 1 (canal negativo):** usar **só scribbles positivos**. O oráculo confirmou que isso ajuda os marcadores pequenos
  ([PD-04](08-pendencias.md#pd-04), §8).
- **Pergunta 2 (oráculo):** feito, ver §8 e o notebook
  [oraculo_scribbleprompt.ipynb](../../notebooks/exploration/oraculo_scribbleprompt.ipynb).
- **Pergunta 3 (normalização + congelar o encoder):** sim. Vira uma ablação conjunta com 5 variantes, planejada na
  [PD-38](08-pendencias.md#pd-38).
- **Pergunta 4 (ScribblePrompt-SAM):** deixada de lado por enquanto. Registrada como ideia adiada na
  [PD-39](08-pendencias.md#pd-39), com o obstáculo encontrado no código: os scribbles viram coordenadas de clique, que não
  são diferenciáveis.

**Ainda em aberto:**

1. **Canal negativo (PD-04):** faz sentido para você o conflito do §3.4? A ideia de uma ablação com negativo zerado (só
   scribbles positivos) ou negativo só **longe** dos núcleos do Cellpose continua de pé? Sim, vamos manter só scribbler positivos, é mais simples e acredito que vai ser o correto 
2. **Oráculo (PD-03):** posso preparar um notebook curto para o Colab que rode só o ScribblePrompt com marcadores derivados
   do GT e meça o Dice? É o experimento que decide se vale insistir no ScribblePrompt. SIm 
3. **Normalização ImageNet (PD-36) e congelar o encoder (PD-38):** são duas mudanças simples. Quer que entrem na próxima
   rodada de experimentos, como ablação?
4. **ScribblePrompt-SAM:** chegou a considerar a variante SAM (1024²) como rede final alternativa?

---

## 8. Resultado do oráculo (2026-09-27)

### 8.1 Em linguagem simples

**O que foi testado.** Os marcadores do oráculo **não vieram da MarkerUNet**: foram tirados do próprio ground truth. São os
melhores marcadores possíveis, algo que nenhuma rede treinada alcança. A pergunta era: *se a MarkerUNet acertasse
perfeitamente, até onde o ScribblePrompt chegaria?*

```
Cellpose sozinho:                imagem ──► Cellpose ──► máscara                                   Dice 0,80
Oráculo:           marcador perfeito (tirado do GT) ──► ScribblePrompt ──► máscara                 Dice = ?
Pipeline real:     imagem + Cellpose ──► MarkerUNet ──► marcador ──► ScribblePrompt ──► máscara    Dice 0,65 (exp. 4)
```

**Pergunta 1: a segmentação do Cellpose sozinho é melhor que a feita a partir dos marcadores?**
**Hoje, sim**, para marcadores pequenos. O resultado depende do tamanho do marcador perfeito que o ScribblePrompt recebe:

| Marcador perfeito dado ao ScribblePrompt | Melhor Dice | Comparado com o Cellpose (≈0,80) |
|---|---|---|
| o **núcleo inteiro** (o próprio GT) | 0,93 | melhor |
| o **miolo** do núcleo (1/3 da área) | 0,78–0,80 | empata, sem passar |
| um **pontinho no centro** de cada núcleo | 0,57–0,61 | bem pior |

O ScribblePrompt só passa do Cellpose quando recebe o **núcleo inteiro** como marcador. Mas aí o marcador já é a própria
resposta, e não sobra trabalho para a rede final.

**Por que isso importa.** A ideia da tese é dar um marcador **pequeno** e deixar a camada de segmentação **expandi-lo** até o
núcleo inteiro. O ScribblePrompt expande pouco: se você marca só o miolo, ele devolve pouco mais que o miolo. Então o problema
não está só no treino da MarkerUNet. **A própria rede final tem um teto baixo** para marcadores pequenos. O exp. 4 (0,65) fica
ainda mais abaixo porque a MarkerUNet não é perfeita.

**Pergunta 2: usar 256 na entrada do ScribblePrompt é melhor?**
**Sim.** É o tamanho da imagem **dentro** do ScribblePrompt; os dados e a MarkerUNet ficam como estão.
- Hoje, a imagem de 1000×1000 é reduzida para **128×128** antes da rede final. Um núcleo típico (~24 px) vira ~3 px, e a rede
  quase não enxerga o formato.
- Em **256×256**, o núcleo fica com ~6 px. No oráculo, 256 foi melhor que 128 em **todos** os casos; por exemplo, com o miolo e
  só positivos, o Dice foi de 0,67 para 0,73.
- **512** ajuda ainda mais os marcadores pequenos, mas "vaza" com marcadores grandes. **256 é o meio-termo mais seguro.**
- É um único parâmetro (`input_size`), com custo ~4× maior de memória e tempo na rede final.

**Ressalvas.** O ganho do 256 foi medido com marcadores perfeitos; falta confirmar num treino de verdade. E **256 sozinho não
resolve a pergunta 1**: mesmo em 256, marcadores pequenos ficam abaixo do Cellpose.

**Em resumo:** subir para 256 é uma melhoria barata e vale fazer. A questão maior, para levar ao orientador, é se o
ScribblePrompt é a rede final certa para a ideia de "marcador pequeno que se expande" ([PD-40](08-pendencias.md#pd-40)).

### 8.2 Números completos

Notebook: [oraculo_scribbleprompt.ipynb](../../notebooks/exploration/oraculo_scribbleprompt.ipynb). Não há treino: o
ScribblePrompt recebe marcadores tirados do próprio GT, e o Dice é medido com as mesmas métricas dos experimentos. Foi rodado
em CPU, fora do Colab, com o mesmo checkpoint (SHA-256 `43f57ee8…acbe`). Os números detalhados e as ressalvas estão na
[PD-40](08-pendencias.md#pd-40).

**Dice no treino** (Cellpose sozinho = **0,802**):

| Marcador | Área / GT | 128² atual | 128² só pos. | 256² atual | 256² só pos. | 512² atual | 512² só pos. |
|---|---|---|---|---|---|---|---|
| GT inteiro | 100% | 0,873 | 0,796 | **0,931** | 0,851 | 0,899 | 0,690 |
| miolo | 33% | 0,452 | 0,668 | 0,532 | 0,731 | 0,507 | **0,784** |
| centros | 9% | 0,088 | 0,342 | 0,150 | 0,482 | 0,157 | 0,573 |
| máscara do Cellpose | 79% | 0,762 | 0,752 | 0,799 | 0,796 | 0,792 | 0,749 |

No teste (Cellpose = 0,810), o padrão se repete. O melhor marcador pequeno é o miolo com só positivo em 512², com **0,805**,
praticamente empatado com o Cellpose, mas sem passar dele.

**O que isso diz sobre os modelos:**
1. **O ScribblePrompt, neste dado, expande pouco os marcadores.** Com o negativo atual, a saída é quase exatamente o marcador
   (precisão ~1, recall ≈ área do marcador). Com só positivos, ele cresce, mas mesmo o miolo perfeito não passa do Cellpose.
   Na tese, quem faz a expansão é a hierarquia do FMBS; aqui não há um equivalente.
2. **O canal negativo atual é o principal limitador para marcadores pequenos** (confirma a PD-04). Mas, com o marcador grande,
   ele ajuda: segura o vazamento.
3. **A resolução importa, com compromisso.** 256² melhora tudo em relação a 128²; 512² faz marcadores pequenos crescerem mais
   e os grandes vazarem (PD-05).
4. **O antialias não ajuda** (PD-37, encerrada): ele tira os scribbles do formato 0/1.
5. **O ScribblePrompt não melhora a máscara do Cellpose.** Com ela como marcador, o resultado piora ou empata.

**Consequência para o TCC** (❓ para discutir com o orientador): com o ScribblePrompt como camada de segmentação, superar o
Cellpose exige marcadores quase do tamanho do núcleo inteiro. Isso contraria o L_size e a noção de "marcador" da tese. Os
caminhos possíveis estão na [PD-40](08-pendencias.md#pd-40): aceitar marcadores grandes, interação iterativa, outra camada
que expanda marcadores ou reformular o objetivo como correção da máscara do Cellpose.
