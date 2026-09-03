# Investigação — Resultados ruins no `experiment_1`

**Data:** 2026-08-10
**Escopo:** `notebooks/experiments/experiment_1.ipynb` (treinamento da MarkerNet via ScribblePrompt congelado)
**Objetivo deste documento:** consolidar **tudo** que pode estar contribuindo para os resultados ruins — desde problemas práticos de engenharia até problemas conceituais de método — para servir de guia na investigação. Não é um laudo definitivo: é um catálogo de suspeitas, ordenado por plausibilidade, com evidências e passos de verificação.

---

## 1. Resumo executivo

O treinamento **não aprende nada**: a loss fica perfeitamente plana a partir da época 2 (train `1.6960`, val `1.8406` por 99 épocas consecutivas), e a melhor validação é a da **época 1**. A decomposição por termo revela o padrão de um **output degenerado**: a segmentação prevista é aproximadamente constante (≈ 0,98 em toda a imagem — "tudo primeiro plano"), produzindo `rmse ≈ 0.86`, `dice ≈ 0.61` e `size ≈ 0.23`.

A hipótese principal (seção 6) é que o sinal de gradiente até a MarkerNet é **efetivamente zero** por causa de uma combinação de fatores: marcadores densos e suaves (fora da distribuição de treino do ScribblePrompt), saturação do sigmoid na saída da rede congelada e a cadeia profunda de backprop através da UNet congelada. O gradiente chega tecnicamente (os tensores têm `.grad` não-`None`), mas com magnitude desprezível.

---

## 2. Evidências observadas no notebook

### 2.1 Curvas de loss (célula 20)

| Época | train_loss | val_loss |
|---|---|---|
| 1 | 1.6720 | 1.8391 |
| 2 | 1.6956 | 1.8405 |
| 3–100 | **1.6960** (constante) | **1.8406** (constante) |

- Melhor val_loss: **época 1** → nada foi aprendido.
- "train_loss não decresceu (1.6720 -> 1.6960)" (célula 24).
- A loss é idêntica até a 4ª casa decimal por 98 épocas → **a saída do modelo não está mudando** (não é "overfitting", é ausência de aprendizado).

### 2.2 Decomposição por termo (célula 20, `history["train_terms"]`)

| Termo | Época 1 | Época 2+ | Interpretação |
|---|---|---|---|
| `dice` | 0.6194 | 0.6093 | Dice coefficient ≈ 0.39 — sobreposição muito ruim |
| `rmse` | 0.8364 | 0.8591 | RMSE ≈ 0.86 em GT binário — **quase máximo possível (1.0)** |
| `size` | 0.2163 | 0.2276 | razão pred/GT ≈ 4.55× — **prevê ~4.5× mais massa que o GT** |
| **total** | 1.6720 | **1.6960** | soma exata dos termos ✓ |

**Smoking gun:** para um GT com 22.4% de foreground em média, uma predição **constante p ≈ 0.98** (tudo primeiro plano) produz:
- `rmse = √(0.776·p² + 0.224·(p−1)²) ≈ 0.86` ✓
- `dice_loss = 1 − 2·p·f/(p² + f) ≈ 0.63` (observado 0.61) ✓
- `size = 0.05 · (p/f) ≈ 0.22` (observado 0.23) ✓

A razão exata do termo `size` (0.2276/0.05 = 4.55×) é levemente maior que a predição constante (0.98/0.224 = 4.38×) — o output está **saturado ainda mais perto de 1.0** na maior parte dos pixels. Ou seja: **a saída da rede congelada está saturada em ~1.0 em praticamente todos os pixels, independentemente da entrada.** O seg observado na célula 16 tem max `0.9909` — consistente.

### 2.3 Faixas de ativação (célula 16)

- `markers`: `[0.426, 0.794]` — marcadores densos e suaves, faixa estreita (logits ≈ −0.3..1.3).
- `segmentation`: `[5.8e-17, 0.991]` — satura em ~0.99.

### 2.4 "O gradiente chega" (célula 18)

- `MarkerNet com gradiente: 140 tensores` (não-`None`) e `Rede final: 0`.
- **Atenção:** o teste só verifica existência (não-`None`), **não magnitude**. Gradientes de magnitude ~1e-8 passariam no teste e ainda assim não moveriam a rede. Faltam métricas de grad-norm.

### 2.5 Estatísticas do GT (calculado sobre os 30 masks de treino)

- Foreground médio: **22.4%** (min 10.5%, max 40.0%) → fundo domina (~77.6%).
- GT é binário (0/1) derivado do XML; células sobrepostas são fundidas.

### 2.6 Outros fatos do run

- **Cellpose caiu no modelo default**: `WARNING:cellpose.models:pretrained model .../cpsam not found, using default model` → o alpha do RGBA não veio do `cpsam_v2` configurado.
- **Distance map computado mas NÃO usado**: o compositor é `[DiceTerm(), RMSETerm(), SizeTerm(weight=0.05)]` — sem `DMapTerm`.
- **Checkpoint comentado** (célula 28) — nada foi persistido.
- **NUM_EPOCHS = 100**, mas o markdown da célula 19 diz "50 épocas".
- Batch = **1 imagem**; 100 épocas × 35 batches = **3500 passos** para 24.4M parâmetros.

---

## 3. Como o treinamento funciona hoje (para ancorar a análise)

```
batch (1 imagem 1000×1000)
  ├─ image   (float [0,255], 3 canais)
  ├─ rgba    (H,W,4) [0,1]          ← CellposeStep + RGBAStep (pré-processamento)
  ├─ ground_truth (1,1,1000,1000)
  └─ distance_map (1,1,1000,1000)   ← DistanceMapStep (não consumido pela loss!)

TrainingPipeline
  1. MarkerStep (differentiable=True)
       rgba → F.interpolate 256² → MarkerUNet (smp.Unet resnet34, 4→1) → sigmoid
       → markers (1,1,1000,1000), dense soft [0.43, 0.79]
       (modelo em mode EVAL — ver P4)
  2. FrozenSegmentationStep
       image → grayscale → /255 → [0,1]
       markers → [s, 1−s] (2 canais)          ← cada pixel tem scribble + e − simultaneamente
       F.interpolate 128² (rescale_inputs)
       prepare_inputs: cat([img, 0, scribbles, 0], 5 canais) + torch.clamp(..., 0, 1)
       ScribblePromptUNet congelada (nf192×4) → logits → sigmoid
       → segmentation (1,1,1000,1000)

LossComposer (Dice + RMSE + Size) sobre segmentation vs ground_truth
  → backward atravessa: loss → seg(sigmoid) → UNet congelada → clamp → bilinear → [s,1−s] → markers(sigmoid) → MarkerNet
```

Nota: o `ctx` do `LossComposer` chama a predição de `"markers"`, mas o `Trainer` passa a **segmentation** (`prediction_key="segmentation"`). Todos os termos (Dice/RMSE/Size) operam sobre a saída da rede congelada — o nome é legado, não é um bug, mas confunde a leitura.

---

## 4. Problemas práticos (engenharia)

### P1 — Sinal de gradiente efetivamente nulo até a MarkerNet (CRÍTICO)
**Sintoma:** loss perfeitamente plana da época 2 em diante; melhor val na época 1.
**Por quê:** o gradiente atravessa uma cadeia longa com dois sigmoids e uma UNet congelada de 4 estágios (192 filtros) — e, principalmente, o sigmoid **final está saturado** (seg ≈ 0.98 ⇒ logits grandes ⇒ `σ'(logits) ≈ 0`). Detalhe importante: o Adam é **invariante à escala** — mesmo gradientes "pequenos" (ex.: 1e-6) moveriam os pesos ~lr por passo. Para a loss ficar **exatamente** plana (4 casas) por 98 épocas, o gradiente precisa estar em **zero de precisão float32** (abaixo de ~1e-7), o que acontece quando `σ'(logits)` (≈1e-3 ou menos) é atenuado pela profundidade da UNet congelada até underflow. A trajetória observada `1.6720 → 1.6956 → 1.6960` mostra o modelo se mover nas 2 primeiras épocas e congelar — o mecanismo de **auto-saturação**.
**Evidência:** seg max 0.9909; RMSE ≈ máximo possível; perda congelada por 98 épocas.
**Verificar:** logar grad-norm (L2) por camada da MarkerNet nas 3 primeiras épocas. Se < ~1e-6 (ou exatamente 0), confirmado.

### P2 — ScribblePrompt com entrada fora de distribuição (CRÍTICO — ver C1/C2)
**Sintoma:** mesmos do P1.
**Por quê:** ver seção 6. A rede congelada foi treinada com scribbles **esparsos e binários**; recebe um campo **denso e suave** em todos os pixels.
**Verificar:** teste de sensibilidade — perturbar markers e medir Δseg; e treinar com scribbles binários esparsos (GT) como oráculo (ver checklist E3).

### P3 — MarkerNet em mode `eval()` durante o treino (ALTO)
**Onde:** `MarkerStep.__init__` → `self.model.model.eval()`.
**Por quê:** com `eval()`, as BatchNorms usam *running stats* sem atualizá-las. O encoder (pretrained ImageNet) tem stats válidas, mas o **decoder recém-inicializado tem stats padrão** (mean=0, var=1) que **nunca se adaptam** — o forward do decoder fica mal calibrado durante todo o treino. Comportamento não convencional (espera-se `train()` para treinar BN).
**Verificar:** rodar com `train()` (e batch > 1) e comparar.

### P4 — Batch size = 1 + 24.4M parâmetros + 3500 passos (ALTO)
**Por quê:** gradientes por amostra são ruidosos; sem acúmulo de gradiente; 3500 passos é pouco para um ResNet34-based. Além disso, com batch=1, mesmo em `train()` as BN seriam instáveis — a decisão de forçar `eval()` provavelmente veio disso (comentário no `MarkerStep`).
**Verificar:** empilhar 2–4 imagens por batch (o notebook já monta batches como listas de dicts — basta agrupar) e/ou usar `accumulate_grad_batches`.

### P5 — Sem scheduler, sem grad clipping, lr fixo 1e-3 (MÉDIO)
**Por quê:** a cadeia congelada pode gerar picos de gradiente ou gradientes com escala instável. `Adam(lr=1e-3)` sem warmup/clipping é frágil nesse cenário. Não é a causa primária (a loss nem se move), mas impede diagnóstico limpo.
**Verificar:** grad clipping + scheduler + monitorar grad-norm.

### P6 — Sem data augmentation (MÉDIO)
**Por quê:** 35 imagens de treino (mesmo com 44 no total) é pouco; sem flips/rotações, o modelo não generaliza e o diagnóstico fica confundido com overfitting — embora aqui o problema seja anterior (não há nem fit do treino).

### P7 — Resolução 1000 → 256 → 128 → 1000 (MÉDIO)
**Por quê:** a predição final é upsampled bilinear de 128² para 1000². A loss é medida em 1000×1000: o RMSE é dominado pelas **bordas borradas** do upsample e o gradiente fino (fronteiras de células) se dilui. Além disso, os markers densos, ao serem reduzidos a 128², viram um campo quase uniforme (ver C2).
**Verificar:** treinar em 512 ou 256 e comparar; ou computar a loss em 128².

### P8 — Desbalanceamento de classes não tratado (MÉDIO)
**Por quê:** 77.6% de fundo. Sem `pos_weight`, o RMSE "empurra" para predições conservadoras e o Dice sozinho pode estagnar. O `size` atual (razão pred/GT) é um proxy fraco de balanceamento.
**Verificar:** adicionar `pos_weight` ao BCE/RMSE ou usar focal/Dice com pesos.

### P9 — Cellpose com modelo default (MÉDIO)
**Evidência:** aviso `cpsam not found, using default model`.
**Por quê:** o canal alpha do RGBA (entrada da MarkerNet) vem de uma segmentação de qualidade inferior à pretendida. Não explica a loss plana (que é pós-MarkerNet), mas degrada a entrada.
**Verificar:** baixar/instalar o modelo `cpsam_v2` e comparar a qualidade das segmentações.

### P10 — `SizeTerm` pode puxar na direção oposta ao Dice/RMSE (MÉDIO)
**Por quê:** `ObjectSizeLoss` usa `ratio = pred.sum()/gt.sum()` (gradiente constante). Com predição densa, o termo empurra a massa total para baixo — enquanto o Dice pode querer o oposto em regiões específicas. É um cabo de guerra com gradiente constante.
**Verificar:** ablation da loss (ver E4).

### P11 — Distance map computado e ignorado (MÉDIO)
**Por quê:** todo o trabalho do `DistanceMapStep` é inócuo neste experimento: o `DMapTerm` não está no compositor. Se a intenção era guiar os markers para os centros celulares, essa supervisão simplesmente **não existiu** (ver C4).
**Verificar:** adicionar `DMapTerm` e comparar.

### P12 — Higiene do experimento (BAIXO, mas importante)
- Checkpoint comentado (nada salvo, resultados não reproduzíveis sem re-rodar).
- `NUM_EPOCHS=100` vs markdown "50 épocas".
- Aviso de HF_TOKEN timeout (inofensivo).
- As 14 imagens de teste oficiais entraram no treino → qualquer comparação futura com a literatura do MoNuSeg fica inválida (ver C6).

---

## 5. Problemas conceituais (método/arquitetura)

### C1 — Usar o ScribblePrompt como "camada de loss" é um uso fora do propósito
O ScribblePrompt v1 é um modelo **interativo**: recebe traços esparsos do usuário (scribbles) e refina a segmentação. Ele **não foi treinado para ter gradientes significativos em relação aos scribbles** — foi treinado para *responder* a scribbles. Usá-lo como função diferenciável fixa que "traduz" marcadores em segmentação é um hack: nada garante que o gradiente ∂seg/∂scribble aponte para direções úteis à MarkerNet (o gradiente pode ser ruidoso, saturado ou desalinhado com o que "bom marcador" significa).

### C2 — Distribuição de entrada dos scribbles é incompatível (a explicação mais concreta)
- ScribblePrompt é treinado com scribbles **esparsos**: a maioria dos pixels tem 0 nos dois canais; traços são 0/1 (ou valores de brush), e os canais positivo/negativo são **mutuamente exclusivos**.
- O pipeline entrega **`[s, 1−s]` com s ∈ [0.43, 0.79] em TODOS os pixels**: ambos os canais são altos simultaneamente em toda a imagem (positivo ≈ 0.6 e negativo ≈ 0.4 ao mesmo tempo). Isso é uma entrada que o modelo nunca viu.
- Resultado observado: a rede congelada responde com **foreground saturado (~0.98) em tudo** — comportamento degenerado de "scribe em todo lugar ⇒ tudo é objeto". Com a saída saturada, o gradiente do sigmoid morre (P1).

### C3 — Supervisão dos markers é 100% indireta
A MarkerNet só recebe gradiente "por procuração", através do ScribblePrompt. Não existe **nenhum alvo direto** que diga o que é um bom marcador (ex.: centroides, mapa de distância, seeds das instâncias, ou o próprio GT diluído). Se o caminho indireto for insensível (como evidenciado), não há caminho de aprendizado alternativo.

### C4 — O mapa de distância existe exatamente para isso e não foi usado
O `DMapTerm` (penalizar ativação longe dos centros celulares) é a supervisão direta natural para a MarkerNet — e está fora do compositor. Vale notar também a **semântica invertida** do `compute_distance_map`: `1.0 − dt_norm` ⇒ **0 no interior, 1 em bordas/fundo**. Se o `DMapTerm` for adicionado, é preciso conferir se essa direção é a desejada (o `DistanceMapLoss` penaliza `pred × dmap`, ou seja, ativação em bordas/fundo — empurra para o interior; parece coerente, mas precisa validação).

### C5 — GT binário funde células sobrepostas
MoNuSeg tem células sobrepostas; a máscara binária (do XML) funde fronteiras entre células. Para uma tarefa de *markers* (pontos/sementes), o GT binário é ambíguo: não há distinção entre "borda de uma célula" e "borda entre duas células". O mapa de distância calculado sobre esse GT herdou a ambiguidade.

### C6 — Vazamento do conjunto de teste oficial (metodologia)
As 14 imagens oficiais de teste do MoNuSeg foram fundidas ao pool de treino (decisão do usuário: "mais dados para treinar"), e a validação é um split aleatório 80/20 sobre o pool combinado. Consequências: (a) qualquer métrica reportada não é comparável com a literatura; (b) a validação pode conter imagens "fáceis" ou correlatas das de treino. Se o objetivo for reportar no teste oficial, é preciso separar: `monuseg_training` → treino, `monuseg_test` → validação.

---

## 6. Hipótese principal — por que a loss é perfeitamente plana

Encadeamento completo, apoiado pelas evidências:

1. **MarkerNet** (inicializada, decoder com BN em stats padrão e mode `eval`) produz markers suaves e de baixo contraste `[0.43, 0.79]` — um "scribe em todo lugar".
2. **ScribblePrompt** (treinado para scribbles esparsos binários) recebe os dois canais de scribble altos em todos os pixels → responde com **foreground saturado** (seg ≈ 0.98 em toda a imagem).
3. A **loss** sobre essa saída é `dice 0.61 + rmse 0.86 + size 0.23 = 1.696` — valores próximos do **máximo teórico** do RMSE binário (1.0) e de uma predição "tudo primeiro plano".
4. O **backward** atravessa o sigmoid saturado da saída e uma UNet congelada profunda → o gradiente que chega à MarkerNet é **efetivamente zero em float32** (abaixo de ~1e-7 — ver P1 sobre por que o Adam não move gradientes "pequenos"): os 140 tensores com `.grad` não-`None` têm magnitude ≈ 0.
5. O Adam **não move** os pesos → a saída não muda → a loss fica **idêntica** da época 2 à 100.

Esse mecanismo explica todos os números ao mesmo tempo (faixa dos markers, saturação do seg, valores exatos dos termos, perfeitamente planos). Ele deve ser o **primeiro** a ser testado.

---

## 7. Hipóteses alternativas priorizadas

| # | Hipótese | Prob. | Como diferenciar da principal |
|---|---|---|---|
| H1 | Gradiente morto por saturação + cadeia profunda (seção 6) | ★★★★★ | Grad-norm por camada; teste de sensibilidade Δseg/Δmarkers |
| H2 | Loss dominada por RMSE com predição constante — o mínimo "razoável" é uma predição conservadora (vazia) | ★★★☆☆ | Ablation: só Dice; só Dice+Size; com pos_weight |
| H3 | BN do decoder em stats padrão (eval) impede aprendizado | ★★★☆☆ | Treinar com `train()` + batch 2–4 |
| H4 | O `prepare_inputs` do pacote scribbleprompt aplica operação não-diferenciável ou com escala inesperada | ★★☆☆☆ | Inspecionar a função (já verificado: `torch.clamp(x, 0, 1)` — diferenciável no interior; nossa faixa [0.21, 0.79] está no interior) |
| H5 | O checkpoint v1 carregado não é o esperado para essa versão de código | ★★☆☆☆ | Validar o checkpoint com o `predict` do próprio pacote num caso conhecido |
| H6 | O `SizeTerm` trava a dinâmica (gradiente constante puxando massa para baixo) | ★★☆☆☆ | Ablation sem SizeTerm |

---

## 8. Checklist de investigação (experimentos de isolamento)

Em ordem de custo-benefício:

1. **E1 — Grad-norm e evolução dos markers (diagnóstico, 5 min).** Nas 3 primeiras épocas, logar: grad-norm L2 total da MarkerNet, média/desvio dos gradientes por camada, e o range de `markers` e `segmentation` por batch. Se grad-norm < 1e-6 e/ou markers não mudam, H1 confirmada.

2. **E2 — Sensibilidade da rede congelada (diagnóstico, 10 min).** Com o pipeline em `no_grad`, rodar o forward com (a) markers = GT binarizado, (b) markers = constantes (0.5, 0.0, 1.0), (c) markers atuais. Comparar as segmentações. Se a saída mal muda entre (a) e (c), a rede congelada é insensível aos markers — a abordagem "loss através do ScribblePrompt" está fundamentalmente comprometida.

3. **E3 — Oráculo: markers = GT (treino, 30 min).** Treinar com os markers **substituídos pelo GT binarizado** (sem gradiente na MarkerNet, só para verificar o teto da rede congelada). Se mesmo assim a loss não cair, o problema está na **rede congelada/loss**, não na MarkerNet.

4. **E4 — Ablation da loss (treino, 3 runs × 20 min).** (a) só `DiceTerm`; (b) `DiceTerm + DMapTerm`; (c) `DiceTerm + RMSETerm + SizeTerm` (atual). Comparar as curvas. Isso separa o efeito do RMSE dominante e testa a supervisão direta dos markers (C4).

5. **E5 — Fallback DummyFrozen (treino, 30 min).** Usar o `DummyFinalNetwork` do próprio notebook (rede congelada de 2 convs) com **loss direta sobre a segmentação** — e também uma variante com `DiceTerm` sobre os **markers** diretamente. Se a MarkerNet aprender no dummy mas não no ScribblePrompt, a culpa é da rede congelada/C1-C2.

6. **E6 — Hiperparâmetros (treino, 1–2 h).** batch 2–4 (empilhar dicts), `train()` na MarkerNet, grad clip 1.0, scheduler (CosineAnnealing), lr menor (1e-4). Re-avaliar grad-norm.

7. **E7 — Resolução (treino, 1 h).** Treinar em 512² (ou 256²) mantendo o resto igual, para eliminar o efeito P7 do upsample 128→1000.

8. **E8 — Reproduzibilidade.** Descomentar o checkpoint e salvar também: `markers`, `segmentation`, grad-norm médio e a config completa, para que qualquer run futuro seja auditável.

---

## 9. Referências (arquivos e células)

**Notebook `notebooks/experiments/experiment_1.ipynb`:**
- cél. 8 — carregamento (44 amostras); cél. 10 — pré-processamento (aviso do Cellpose default); cél. 12 — `build_batch`; cél. 14 — redes (ScribblePrompt carregado); cél. 16 — ranges de markers/seg; cél. 18 — checagem de gradiente; cél. 20 — treino + termos; cél. 22 — curvas; cél. 26 — visualização; cél. 28 — checkpoint comentado.

**Código:**
- `src/pipeline/steps/inference/marker_step.py` — `eval()` no init; `_forward_differentiable` (sigmoid + `F.interpolate`); `_to_tensor_bchw`.
- `src/pipeline/steps/inference/frozen_segmentation_step.py` — forward com grafo preservado.
- `src/models/networks/final_segmentation/scribble_prompting_network.py` — `_prepare_scribbles` (`[s, 1−s]`), `forward`, `train()`→`eval()`.
- `src/losses/terms.py` + `src/losses/loss_composer.py` — nota: `ctx["markers"]` na prática recebe a `segmentation` (nomenclatura legada; os três termos usados atuam sobre a saída da rede congelada).
- `src/training/trainer.py` — `_compute_loss` (prediction_key = "segmentation").
- `src/losses/object_size_loss.py`, `rmse_loss.py`, `soft_dice_loss.py` — termos usados.
- `src/pipeline/steps/preprocessing/distance_map_step.py` — mapa calculado (não usado na loss).
- Pacote `scribbleprompt` (`scribbleprompt/models/unet.py`) — `prepare_inputs` faz `cat([img, box_embed, scribble_click_embed, mask_input])` com `torch.clamp(..., 0, 1)` nos scribbles; espera `img` em [0,1] (assert no `predict`).

---

## 10. Próximos passos sugeridos

1. Rodar **E1 + E2** (diagnóstico, sem custo de treino) para confirmar ou descartar a hipótese principal.
2. Rodar **E5** (dummy) e **E3** (oráculo) para isolar rede congelada vs. MarkerNet.
3. Decidir, com base nos resultados: corrigir a normalização/forma dos scribbles (binarizar ou esparsificar os markers), adicionar supervisão direta (`DMapTerm`/centroids), ou reavaliar a arquitetura de treino (ex.: treinar a rede final junto, ou usar perda direta sobre markers).

---

## 11. Correções aplicadas (2026-08-13)

| Problema | Correção aplicada | Onde |
|---|---|---|
| **P1** — gradiente efetivamente nulo (CRÍTICO) | Novo `GradNormCallback` (norma L2 total, média, máximo e fração de parâmetros com gradiente não-`None`) para confirmar/descartar gradiente morto em tempo real; notebook agora mede a **magnitude** do gradiente (não só existência) na checagem de gradiente. | `src/training/callbacks/grad_norm_callback.py`; notebook cél. 7/8/9 |
| **P2/C2** — scribbles fora de distribuição (CRÍTICO) | `ScribblePromptingNetwork` agora converte os marcadores em canais **quase binários e complementares** (modo padrão `sharpened`, `sigmoid(T·(s−0.5))` / `sigmoid(T·(0.5−s))`), em vez de `[s, 1−s]` suave em todos os pixels. O modo legado `dense_soft` continua disponível para ablação. | `src/models/networks/final_segmentation/scribble_prompting_network.py` |
| **P3** — MarkerNet em `eval()` no treino | `MarkerStep` agora coloca a rede em `train()` quando `differentiable=True` e em `eval()` quando `differentiable=False`. | `src/pipeline/steps/inference/marker_step.py` |
| **P4** — batch size 1 | Notebook usa **batch size 4** (empilhamento de amostras 256²), estabilizando BatchNorm e reduzindo ruído do gradiente. | `notebooks/experiments/experiment_1.ipynb` (cél. 4) |
| **P5** — sem scheduler/clipping | `Trainer` ganhou `grad_clip` (`clip_grad_norm_` após `backward`); notebook usa lr 1e-4 + `CosineAnnealingLR` + grad clip 1.0. | `src/training/trainer.py`; notebook cél. 8 |
| **P6** — sem augmentação | Notebook aplica rotação 90° e flips aleatórios (imagem, rgba, GT e dmap em conjunto) nos batches de treino. | `notebooks/experiments/experiment_1.ipynb` (cél. 4) |
| **P7** — resolução 1000→256→128→1000 | Notebook trabalha em **256²** (loss medida no output upsampled apenas 128→256, não 128→1000), e o mapa de distância é calculado sobre o GT **já redimensionado** (conforme a arquitetura). | `notebooks/experiments/experiment_1.ipynb` (cél. 3/4) |
| **P9** — Cellpose modelo default | Corrigido o typo `pretreined_model` → `pretrained_model`; modelo default alterado de `cpsam_v2` (inexistente) para `cpsam`; aviso explícito quando o modelo solicitado não está na lista conhecida. | `src/pipeline/steps/preprocessing/cellpose_step.py` |
| **P10** — `SizeTerm` assimétrico | `ObjectSizeLoss` agora é **simétrico** em torno de 1.0 (`weight·|ratio−1|`): penaliza tanto superestimar quanto subestimar, com ótimo em `pred.sum() == gt.sum()` e gradiente que se anula no ótimo (antes, `weight·ratio` empurrava a massa para zero). | `src/losses/object_size_loss.py` |
| **P11/C3/C4** — mapa de distância ignorado / supervisão indireta | O `LossComposer` agora recebe `prediction` (segmentação final) e `markers` (saída da MarkerNet) **separados**; o `DMapTerm` supervisiona os **marcadores diretamente** (penaliza ativação em bordas/fundo, empurrando para o interior das células) e foi adicionado ao compositor do notebook. | `src/losses/loss_composer.py`, `src/losses/terms.py`, `src/training/trainer.py`; notebook cél. 8 |
| **P12** — higiene | Checkpoint descomentado (salva `state_dict`, config, épocas, resolução, batch size, melhor val loss e grad-norm médio); `NUM_EPOCHS=50` consistente com a documentação e as asserções. | `notebooks/experiments/experiment_1.ipynb` (cél. 12) |
| **C6** — vazamento do teste oficial | Split corrigido: treino = **30 imagens oficiais de treino**, validação = **14 oficiais de teste** (sem misturar conjuntos). | `notebooks/experiments/experiment_1.ipynb` (cél. 2/4) |
| **Robustez do MarkerStep** | Fallback sem modelo não exige mais a chave `rgba`; modo inferência aceita tensor (não só NumPy). | `src/pipeline/steps/inference/marker_step.py` |

**Observações (não corrigidas nesta rodada, por serem decisões de método/experimento):**

- **C1** — usar o ScribblePrompt como "camada de loss" continua sendo um hack conceitual; o sharpening (C2) e a supervisão direta dos markers (C3/C4) mitigam o sintoma, mas a validação definitiva exige os experimentos de isolamento E2/E3/E5. Se persistirem resultados ruins com a cadeia congelada, a alternativa arquitetural é treinar a rede final junto ou usar perda direta sobre os markers.
- **C5** — a ambiguidade do GT binário (células sobrepostas fundidas) permanece; o mapa de distância herdou essa ambiguidade.
- **P8** — o desbalanceamento (77,6% fundo) é mitigado pelo Dice (invariante à escala) e pelo `DMapTerm` (que penaliza ativação no fundo), mas nenhum `pos_weight`/focal foi adicionado nesta rodada.

Mudanças de arquitetura correspondentes documentadas em `docs/ARCHITECTURE.md` (novo `training/callbacks/grad_norm_callback.py`, `grad_clip` no `Trainer`, contrato `prediction`/`markers` do `LossComposer`, modos de scribble, gerenciamento train/eval do `MarkerStep`).

> **Estudo detalhado, mudança a mudança** (o que foi feito, por que e como validar cada item,
> com diffs e ordem sugerida de estudo): [`docs/estudo_das_mudancas_experimento_1.md`](estudo_das_mudancas_experimento_1.md).
