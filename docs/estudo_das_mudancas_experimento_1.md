# Estudo das mudanças — `experiment_1`

> Documento de estudo, mudança a mudança, de tudo o que foi alterado para corrigir os
> problemas do `notebooks/experiments/experiment_1.ipynb`. A motivação de cada correção está em
> `docs/investigacao_experimento_1.md` (problemas **P1–P12** e conceituais **C1–C6**);
> este documento explica **o que** foi mudado, **por que** e **como validar** cada item.
>
> Data: 2026-08-13 · Branch: `feat/experiment_1`

---

## Sumário

| # | Mudança | Problema(s) | Arquivo(s) |
|---|---|---|---|
| 1 | `MarkerStep` gerencia `train()`/`eval()` por modo | P3 | `src/pipeline/steps/inference/marker_step.py` |
| 2 | `ObjectSizeLoss` simétrico | P10 | `src/losses/object_size_loss.py` |
| 3 | `grad_clip` no `Trainer` | P5 | `src/training/trainer.py` |
| 4 | `LossComposer` com contrato `prediction` + `markers` | C3, C4, P11 | `src/losses/loss_composer.py`, `loss_term.py`, `terms.py`, `trainer.py` |
| 5 | `GradNormCallback` (diagnóstico de gradiente morto) | P1, E1 | `src/training/callbacks/grad_norm_callback.py` |
| 6 | Scribbles "sharpened" na `ScribblePromptingNetwork` | C2, P2 | `src/models/networks/final_segmentation/scribble_prompting_network.py` |
| 7 | `CellposeStep`: typo + validação do modelo | P9 | `src/pipeline/steps/preprocessing/cellpose_step.py` |
| 8 | Robustez do `MarkerStep` (fallback sem `rgba`, tensor na inferência) | robustez | `src/pipeline/steps/inference/marker_step.py` |
| 9 | Notebook reescrito (split, batch, resolução, augmentação, scheduler, checkpoint) | C6, P4, P6, P7, P12 | `notebooks/experiments/experiment_1.ipynb` |
| 9b | **Visualização dos marcadores dentro do notebook** (antes e depois do treino) | pedido do usuário | `notebooks/experiments/experiment_1.ipynb` (cél. 20 e 28) |
| 10 | Script de visualização dos marcadores | — | `scripts/visualize_markers.py` |
| 11 | Testes novos/atualizados | — | `tests/` |
| 12 | Documentação de arquitetura | — | `docs/ARCHITECTURE.md`, `docs/investigacao_experimento_1.md` |

**Resumo dos resultados observados** (treino demo em CPU, `scripts/visualize_markers.py`):

- A loss de treino **decresceu** (1.148 → 1.101), confirmando que o gradiente chega à MarkerNet com o fluxo corrigido (antes, a loss era perfeitamente plana).
- Os marcadores pós-treino ficaram **muito mais seletivos**: a fração de pixels com ativação > 0.5 caiu de **~67%** (rede aleatória) para **11–25%** — próximo da cobertura real das células no GT (~20–25%).
- `GradNormCallback` reportou norma L2 dos gradientes da MarkerNet consistentemente **> 0** durante o treino.

---

## Mudança 1 — `MarkerStep` gerencia `train()`/`eval()` por modo (P3)

### O problema

`MarkerStep` forçava o modelo em `eval()` **sempre**, inclusive durante o treinamento. Com o
decoder da MarkerNet recém-inicializado, as BatchNorms nunca aprendiam as estatísticas do lote
corrente (ficavam presas às `running stats` de init), degradando o sinal que a rede final
congelada recebia.

### O que mudou

`src/pipeline/steps/inference/marker_step.py` — no `__init__`, a decisão passou a depender do modo:

```python
# antes (sempre eval)
self.model.model.eval()

# depois (modo explícito, igual ao FrozenSegmentationStep)
if differentiable:
    self.model.model.train()
else:
    self.model.model.eval()
```

### Por que

- `differentiable=True` = treinamento → `train()` para que as BatchNorms sejam atualizadas com as estatísticas do lote corrente (com batch ≥ 2, ver Mudança 9).
- `differentiable=False` = inferência → `eval()` para usar `running stats` estáveis.

### Como validar

- `tests/test_marker_step.py` cobria apenas o modo não-diferenciável; `tests/test_training_integration.py` (novo) agora verifica que o modelo está em `train()` durante o treino demo:
  ```python
  assert marker_net.model.training is True   # durante o treino
  ```
- O próprio run demo imprimiu `MarkerNet em train() durante o treino: True`.

---

## Mudança 2 — `ObjectSizeLoss` simétrico (P10)

### O problema

A formulação anterior `loss = weight * ratio` (com `ratio = pred.sum() / gt.sum()`) tinha dois defeitos:

1. **Assimétrica**: minimizar `ratio` empurra a massa prevista **sempre para baixo** — o ótimo está em `pred.sum() → 0`, não em `pred.sum() == gt.sum()`.
2. **Gradiente constante**: `d(weight*ratio)/d(pred) = weight / gt.sum()`, nunca se anula — um "cabo de guerra" permanente contra os termos Dice/RMSE.

### O que mudou

`src/losses/object_size_loss.py`:

```python
# antes
ratio = y_pred.sum() / y_true.sum()
return self.weight * ratio

# depois
gt_sum = y_true.sum()
if gt_sum == 0:
    # GT vazio: a massa prevista deve ser (idealmente) zero também.
    return self.weight * y_pred.sum().abs()
ratio = y_pred.sum() / gt_sum
return self.weight * (ratio - 1.0).abs()
```

### Por que

- `|ratio − 1|` é **simétrico**: penaliza tanto superestimar quanto subestimar, com ótimo em `ratio == 1` (massa prevista == massa do GT).
- O gradiente se **anula no ótimo**: não há mais força residual contra o Dice/RMSE.
- Caso especial `gt_sum == 0` (máscara vazia): penaliza qualquer massa prevista — comportamento defensável e explícito.

### Como validar

- `tests/test_object_size_loss.py` (novo): verifica que `loss(2x) ≈ loss(0.5x)` (simetria) e que o gradiente em relação à predição se anula em `ratio == 1`.
- O `SizeTerm` no notebook passou a usar `weight=0.05` e agora opera sobre `prediction` (a segmentação final), não mais sobre `markers` (ver Mudança 4).

---

## Mudança 3 — `grad_clip` no `Trainer` (P5)

### O problema

O treino rodava sem scheduler e sem clipping, com lr fixo 1e-3. A cadeia de backprop é profunda
(markers → scribbles → rede congelada → loss), o que favorece gradientes instáveis/explosivos
nos estágios iniciais.

### O que mudou

`src/training/trainer.py`:

```python
# novo parâmetro no construtor
grad_clip: Optional[float] = None,   # None desabilita o clipping

# no passo de treinamento, após backward() e antes de optimizer.step()
loss.backward()
if self.grad_clip is not None:
    params = [p for group in self.optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
self.optimizer.step()
```

### Por que

`clip_grad_norm_` limita a norma L2 global dos gradientes ao valor configurado, evitando passos
explosivos sem alterar a direção do gradiente (apenas a magnitude). O notebook usa `grad_clip=1.0`.

### Como validar

- `tests/test_trainer.py` (atualizado): `make_trainer` aceita `grad_clip`; o teste do passo de treino confirma que `clip_grad_norm_` é chamado (via spy) quando `grad_clip` está definido.
- `tests/test_training_integration.py`: treino demo roda com `grad_clip=1.0` e todas as losses finitas.

---

## Mudança 4 — `LossComposer` com contrato `prediction` + `markers` (C3, C4, P11)

### O problema (o coração da investigação)

- **C3**: a supervisão da MarkerNet era **100% indireta** — só via o gradiente que atravessava a rede final congelada (que, por sua vez, recebia entrada fora de distribuição — C2).
- **C4/P11**: o mapa de distância era computado e **ignorado**; o `DMapTerm` existia mas supervisionava a segmentação, não os marcadores.
- Todos os termos (Dice/RMSE/Size) apontavam para o mesmo tensor (`markers`), misturando duas responsabilidades diferentes: qualidade da **segmentação final** vs qualidade dos **marcadores**.

### O que mudou

**`src/losses/loss_composer.py`** — o `forward` agora recebe dois tensores de predição distintos:

```python
def forward(self, prediction, distance_maps, gt_masks, markers=None):
    ctx = {
        "prediction": prediction,                          # segmentação final (alvo do Trainer)
        "markers": markers if markers is not None else prediction,  # saída da MarkerNet
        "distance_maps": distance_maps,
        "gt_masks": gt_masks,
    }
```

- `ctx["prediction"]` = a predição supervisionada pelo `Trainer` (por padrão a **segmentação final** — `prediction_key="segmentation"`). Alvo dos termos Dice/RMSE/Size.
- `ctx["markers"]` = os **marcadores da MarkerNet** (`data["markers"]`), quando presentes. Alvo da supervisão direta (`DMapTerm`). Se a MarkerNet não estiver no pipeline, assume o valor de `prediction` (compatibilidade).

**`src/losses/terms.py`**:

- `SizeTerm`, `DiceTerm`, `RMSETerm` passaram a ler `ctx["prediction"]`.
- `DMapTerm` continua lendo `ctx["markers"]` — agora supervisionando os **marcadores diretamente**: `DistanceMapLoss(markers, distance_maps, gt_masks)` penaliza ativação dos marcadores em regiões de alto valor no mapa de distância (bordas/fundo), empurrando os marcadores para o interior das células.

**`src/training/trainer.py`** — `_compute_loss` repassa os markers:

```python
markers = data.get("markers")
return self.loss_composer(prediction, distance_maps, gt_masks, markers=markers)
```

**`src/losses/loss_term.py`** — docstring do contrato de contexto atualizada.

### Por que

Separa as duas responsabilidades que estavam colapsadas num único tensor:

| Termo | Supervisiona | Objetivo |
|---|---|---|
| `DiceTerm`, `RMSETerm`, `SizeTerm` | `prediction` (segmentação final) | qualidade da segmentação |
| `DMapTerm` | `markers` (saída da MarkerNet) | geometria dos marcadores (interior das células) |

Isso dá à MarkerNet um **sinal direto e geométrico**, sem depender apenas do gradiente que
atravessa a rede congelada — e usa o mapa de distância que já era computado.

### Como validar

- `tests/test_training_integration.py` (novo): compõe `LossComposer` com `DiceTerm/RMSETerm/SizeTerm/DMapTerm` e verifica que `ctx["prediction"]` e `ctx["markers"]` são tensores distintos (a segmentação final e os marcadores) — e que o `DMapTerm` de fato recebe os marcadores.
- `tests/test_trainer.py` (atualizado): o `DummyLossComposer` aceita o novo argumento `markers`.
- `tests/test_training_pipeline.py`: segue passando (contrato retroativo preservado pelo fallback `markers = prediction`).

---

## Mudança 5 — `GradNormCallback` (P1, E1)

### O problema

A hipótese principal da investigação era **gradiente morto**: a loss plana podia significar que
o gradiente que chegava à MarkerNet era zero (em precisão float32). O notebook antigo só checava
a **existência** de gradiente, não a **magnitude**.

### O que mudou

Novo módulo `src/training/callbacks/grad_norm_callback.py` (e `__init__.py` no pacote):

- `GradNormCallback(TrainerCallback)` implementa `on_train_step_end`.
- Para cada passo, coleta os parâmetros com `requires_grad=True` dos steps do `TrainingPipeline` (atributos `model`/`final_network` — na prática, só a MarkerNet) e calcula:
  - `total`: norma L2 combinada de todos os gradientes;
  - `mean_abs`: média do valor absoluto;
  - `max_abs`: maior valor absoluto;
  - `nonzero_frac`: fração de parâmetros com gradiente não-`None` (detecta parâmetros "mortos").
- Acumula tudo em `self.history` (para plotagem) e loga a cada `log_every` passos.

### Por que

É o instrumento que **confirma ou descarta** a hipótese P1 em tempo real: `total`/`max_abs`
abaixo de ~1e-6 (ou exatamente 0) indicam gradiente morto; valores saudáveis (≥ 1e-3) indicam
que o problema estava em outro lugar (distribuição de entrada — C2).

### Como validar

- `tests/test_training_integration.py` (novo): roda o treino demo com o callback e verifica que `grad_norm_cb.history["total"]` tem entradas **> 0** e `nonzero_frac == 1.0`.
- O run demo registrou `total` na faixa de ~1e-2 a ~1e-1 — gradiente **vivo** com o fluxo corrigido.
- No notebook, o callback roda com `log_every=1` e a célula de resultados imprime a série completa do grad-norm.

---

## Mudança 6 — Scribbles "sharpened" na `ScribblePromptingNetwork` (C2, P2)

### O problema

O ScribblePrompt foi treinado com **traços de scribble 0/1 mutuamente exclusivos** (o usuário
desenha em células ou no fundo). A cadeia antiga alimentava a rede com `[s, 1−s]`, onde `s` era o
marcador **suave** (probabilidade contínua) em **todos os pixels** da imagem — uma entrada
totalmente fora da distribuição de treino do modelo congelado.

### O que mudou

`src/models/networks/final_segmentation/scribble_prompting_network.py`:

- Novos parâmetros no construtor: `scribble_mode: Literal["sharpened", "dense_soft"] = "sharpened"` e `scribble_temperature: float = 10.0`.
- Nova lógica de expansão de canais (`_expand_scribble_channels`), usada tanto para tensor quanto para numpy:

```python
# modo padrão "sharpened": canais complementares quase binários
t = self.scribble_temperature
pos = torch.sigmoid(t * (s - 0.5))   # "é célula"
neg = torch.sigmoid(t * (0.5 - s))   # "é fundo"
return torch.cat([pos, neg], dim=1)

# modo legado "dense_soft": [s, 1 - s]
return torch.cat([s, 1.0 - s], dim=1)
```

- O tensor preserva o grafo computacional (sem `detach`) — a backprop continua chegando à MarkerNet (o sharpening é diferenciável).

### Por que

- Com `T=10`, `sigmoid(10·(s−0.5))` satura rapidamente: pixels com `s > 0.5` viram ≈1 no canal positivo, pixels com `s < 0.5` viram ≈0 — **quase binário e complementar**, aproximando a distribuição de treino do ScribblePrompt.
- Mantém diferenciabilidade (ao contrário de uma binarização dura `> 0.5`), então o gradiente ainda flui.
- O modo `dense_soft` fica disponível para **ablação** (comparar os dois na mesma configuração).

### Como validar

- `tests/test_scribble_prompting_network.py` (atualizado): novos testes do modo `sharpened` — verifica saturação (`pos ≈ 1` para `s > 0.5`, `pos ≈ 0` para `s < 0.5`), complementaridade (`pos + neg ≈ 1`), e que o modo legado `dense_soft` continua `[s, 1−s]`.
- `tests/test_training_integration.py`: a cadeia completa (markers → scribbles sharpened → rede final) roda e produz gradiente.

---

## Mudança 7 — `CellposeStep`: typo + validação do modelo (P9)

### O problema

- Typo no nome do parâmetro: `pretreined_model` (com `ei` trocado) — o notebook passava `pretrained_model` e caía no default.
- O default era `cpsam_v2`, um modelo **inexistente** no Cellpose 4.1.1 → o Cellpose caía silenciosamente no modelo padrão (`cyto3`), degradando o canal alpha do RGBA sem aviso.

### O que mudou

`src/pipeline/steps/preprocessing/cellpose_step.py`:

- Parâmetro renomeado para `pretrained_model` (com `ai` correto); default `cpsam_v2` → `cpsam` (modelo existente).
- Novo helper `_warn_if_model_unavailable` que compara o nome com `models.MODEL_LIST` (lista conhecida do Cellpose) e registra um aviso explícito se o modelo não estiver lá — o Cellpose ainda tentará carregar, mas o usuário fica sabendo.

### Por que

O canal alpha do RGBA (máscara de entrada) era uma fonte silenciosa de degradação; um modelo
errado ou inexistente muda completamente a entrada da cadeia sem nenhum erro.

### Como validar

- Sem GPU local não dá para instanciar `CellposeModel`, então a validação é estática: o helper `_warn_if_model_unavailable` é testável isoladamente (aviso emitido para nome desconhecido, silêncio para nome conhecido).
- `tests/test_training_integration.py` não exercita o Cellpose (fallback do notebook, sem GPU).

---

## Mudança 8 — Robustez do `MarkerStep` (fallback sem `rgba`, tensor na inferência)

### O problema

- O fallback sem modelo exigia a chave `rgba` **antes** de checar se havia modelo — um fallback que não funcionava sem o pré-processamento completo.
- O caminho de inferência (`_forward_inference`) só aceitava NumPy, quebrando quando o pipeline recebia tensores.

### O que mudou

`src/pipeline/steps/inference/marker_step.py`:

1. O bloco de fallback (`model is None`) agora roda **antes** da checagem de `rgba`, e aceita `segmentation` como tensor:
   ```python
   if self.model is None:
       segmentation = data["segmentation"]
       if isinstance(segmentation, torch.Tensor):
           segmentation = segmentation.detach().cpu().numpy()
       data["markers"] = (segmentation > 0).astype(np.float32)
       return data
   ```
2. `_forward_inference` aceita tensor `(N, 4, H, W)`/`(4, H, W)` (converte para NumPy via `_to_tensor_bchw`).

### Por que

Permite usar o `MarkerStep` com modelos ausentes (útil para testes/demos — inclusive o run demo
local) e aceitar a saída de qualquer pipeline anterior, independentemente do formato.

### Como validar

- `tests/test_marker_step.py` (pré-existente, corrigido): o fallback sem modelo não exige mais `rgba`.
- `tests/test_training_integration.py`: o pipeline completo roda com batches de tensores.

---

## Mudança 9 — Notebook `experiment_1.ipynb` reescrito

O notebook foi reconstruído (via `build_nb.py`, removido depois de gerar o `.ipynb`) com as
células abaixo. Mapeamento célula → problema:

| Célula | Conteúdo | Problema(s) |
|---|---|---|
| 2–4 | Setup Colab/local (clone, install) | P12 (higiene) |
| 7 | Imports | — |
| 9 | **Split sem vazamento**: `MonusegDataset` oficial de treino (30) e teste (14) carregados **separadamente** | **C6** |
| 11 | `WORK_SIZE = 256`; `resize_sample` (imagem + GT); pré-processamento (Cellpose opcional / fallback RGBA + DistanceMap) | **P7** |
| 13 | `BATCH_SIZE = 4`; `build_batch` com **augmentação conjunta** (rotação 90° + flips em img/rgba/gt/dmap); `train_batches` (treino) e `val_batches` (teste oficial) | **P4, P6, C6** |
| 15–17 | MarkerNet (treinável) + `MarkerStep(differentiable=True)` + `FrozenSegmentationStep` + `TrainingPipeline` | P3 (via Mudança 1) |
| 19 | Checagem de gradiente com **magnitude** | P1/E1 |
| 21 | `LossComposer([DiceTerm, RMSETerm, SizeTerm(0.05), DMapTerm(0.1)])`; Adam lr 1e-4; `CosineAnnealingLR`; `grad_clip=1.0`; `GradNormCallback(log_every=1)`; `TrainingLoop(NUM_EPOCHS=50)` | **P5, P11, P10, E1, P12** |
| 23–25 | Plots de loss/terms/grad-norm; asserções de finitude e de que `train_loss` decresceu | E1, P12 |
| 27–29 | Checkpoint com config completa (`state_dict`, config, épocas, size, batch_size, best_val_loss, grad_norm_mean) | **P12/E8** |

Detalhes relevantes:

- **P7 — resolução**: o treino roda em 256²; a loss é medida sobre o output da rede final (que já redimensiona 128→256), **não** sobre upsampling para 1000. O mapa de distância é calculado sobre o GT **já redimensionado** (`DistanceMapStep` dentro do fluxo de pré-processamento).
- **P4 — batch**: `BATCH_SIZE=4` empilhado (imagens 256²), estabilizando BatchNorm e reduzindo o ruído do gradiente.
- **P6 — augmentação**: rotação 90° (múltiplos) + flips horizontais/verticais, aplicados **em conjunto** a imagem, rgba, GT e dmap (sem desalinhamento).
- **P12 — consistência**: `NUM_EPOCHS=50` em todo o notebook (sem o descompasso antigo em que o treino rodava 50 épocas mas as células seguintes usavam 20/30).

---

## Mudança 9b — Visualização dos marcadores dentro do notebook

> Adicionada a pedido do usuário: "quero ver os marcadores no nosso notebook de experimento no momento em que estivermos rodando ele".

### O que mudou

Duas células novas/alteradas no `notebooks/experiments/experiment_1.ipynb`:

1. **Célula 20 (nova, logo após a checagem de gradiente) — "Marcadores ANTES do treino"**:
   roda o pipeline sobre 2 batches de validação com a MarkerNet recém-inicializada
   (`marker_net.model.eval()`) e plota 4 colunas por imagem: **imagem, ground truth,
   marcadores (antes do treino), scribble positivo** (o que a rede final realmente recebe,
   já com o sharpening C2). Ao final, devolve a MarkerNet a `train()`.

   A coluna do scribble positivo é o ponto didático da mudança C2: mostra como os marcadores
   suaves viram canais quase binários/ complementares na entrada da rede congelada.

2. **Célula 28 (atualizada, pós-treino)**: passou de 4 para **5 colunas** — imagem, ground
   truth, **marcadores (pós-treino)**, segmentação prevista e prevista binária. Assim é
   possível comparar visualmente o antes (cél. 20) com o depois (cél. 28) dos marcadores.

### Por que

O script `scripts/visualize_markers.py` servia para validar localmente, mas o usuário quer ver
os marcadores **no fluxo real do experimento**, no momento em que o notebook roda no Colab com
GPU e com o ScribblePrompt de verdade. As duas células usam apenas o `training_pipeline` já
configurado e os `val_batches` já existentes — nada de infraestrutura nova.

### Como validar

- As células 20 e 28 foram validadas isoladamente em ambiente replicado (fallback local):
  `marker_net.model.eval()` → gera marcadores → `marker_net.model.train()` restaurado;
  layout de 5 colunas da célula 28 renderiza sem erro.
- O `Trainer` chama `optimizer.zero_grad()` no início de cada `train_step`, então o backward
  manual da célula 19 não contamina o treino que vem a seguir.

---

## Mudança 10 — Script de visualização dos marcadores

`scripts/visualize_markers.py` — treina brevemente a MarkerNet em CPU **com o fluxo corrigido**
(6 imagens de treino, 4 épocas, batch 3) e salva em `outputs/markers/`:

- `comparativo_markers.png`: para cada imagem de validação, 5 colunas — imagem, GT, **marcadores antes** (rede aleatória), **marcadores depois** (pós-treino), segmentação prevista.
- `TCGA-*_markers_before.png` / `_markers_after.png` / `_markers_after_bin.png` / `_seg.png`: PNGs individuais.
- `checkpoints/marker_net_demo.pt`: checkpoint demo (state_dict, config, loss final).

Fallbacks (sem GPU/sem pacotes locais, iguais aos do notebook): alpha do RGBA derivado do GT
(sem Cellpose) e `DummyFinalNetwork` congelada (sem `scribbleprompt`).

**Resultado observado** (CPU, ~7,6 s/passo):

- Loss de treino 1.148 → 1.101 (decresceu).
- Marcadores pós-treino muito mais seletivos: fração de pixels > 0.5 caiu de ~67% para 11–25% (cobertura real das células ~20–25%).
- Grad-norm total consistentemente > 0 ao longo do treino.

> O notebook completo (GPU, ScribblePrompt real, 50 épocas) ainda **não foi executado** — este
> script é a validação local de que o fluxo corrigido funciona de ponta a ponta.

---

## Mudança 11 — Testes

| Arquivo | Status | O que cobre |
|---|---|---|
| `tests/test_object_size_loss.py` | **novo** | Simetria do `ObjectSizeLoss` (`loss(2x) ≈ loss(0.5x)`), ótimo em `ratio == 1`, gradiente nulo no ótimo, caso GT vazio |
| `tests/test_training_integration.py` | **novo** | Cadeia completa corrigida: `MarkerStep` em `train()` no treino, gradiente vivo (`GradNormCallback`), `LossComposer` com `prediction`/`markers` distintos, `DMapTerm` supervisionando markers, grad clip, losses finitas |
| `tests/test_scribble_prompting_network.py` | atualizado | Modo `sharpened` (saturação, complementaridade) e modo legado `dense_soft` |
| `tests/test_trainer.py` | atualizado | `DummyLossComposer` aceita `markers`; `grad_clip` aplicado no passo; scheduler ligado ao otimizador **do trainer** (bug pré-existente corrigido) |
| `tests/test_marker_step.py` | corrigido | Fallback sem modelo não exige `rgba` |
| `tests/test_rgba_step.py` | corrigido | Import quebrado (`RGBA` → `RGBAStep`) — bug pré-existente |

**Estado final**: 68 testes passando, `ruff check` limpo.

---

## Mudança 12 — Documentação

- `docs/ARCHITECTURE.md`:
  - Árvore de diretórios com `src/training/callbacks/grad_norm_callback.py`.
  - Seção do `MarkerStep`: gerenciamento `train()`/`eval()` por modo.
  - Seção da `ScribblePromptingNetwork`: modos `sharpened`/`dense_soft` e temperatura.
  - Seção do `LossComposer`: contrato `prediction`/`markers`.
  - Seção do `Trainer`: `grad_clip`.
  - Nova seção do `GradNormCallback`.
  - Exemplo da seção 8 atualizado para a API nova do `LossComposer`.
- `docs/investigacao_experimento_1.md`: nova seção **11. Correções aplicadas** — tabela problema → correção → local, com as observações de itens **não** corrigidos (C1, C5, P8) e por quê.

---

## O que ficou de fora (deliberadamente)

| Item | Motivo |
|---|---|
| **C1** — usar o ScribblePrompt como "camada de loss" é um hack conceitual | Mitigado por C2 + C3/C4; validação definitiva exige os experimentos de isolamento E2/E3/E5 |
| **C5** — GT binário funde células sobrepostas | O mapa de distância herdou a ambiguidade; corrigir é mudança de dataset/experimento |
| **P8** — desbalanceamento de classes (77,6% fundo) | Mitigado por Dice (invariante à escala) + `DMapTerm`; `pos_weight`/focal ficaram para um próximo experimento |

---

## Como estudar cada mudança

1. **Leia a seção correspondente deste documento** (o "por que" e o "como validar").
2. **Abra o diff**: `git show a0bb5c4 -- <arquivo>` (ou `git diff` no working tree).
3. **Rode os testes do item**: `pytest tests/test_object_size_loss.py -q`, etc.
4. **Para as mudanças de fluxo (1, 4, 5, 6)**: `python scripts/visualize_markers.py` reproduz a cadeia completa em CPU e gera os marcadores.

Ordem sugerida de estudo (do mais isolado ao mais sistêmico):

1. Mudança 7 (Cellpose — mais simples, typo)
2. Mudança 2 (SizeLoss simétrico)
3. Mudança 3 (grad_clip)
4. Mudança 8 (robustez do MarkerStep)
5. Mudança 1 (train/eval do MarkerStep)
6. Mudança 6 (scribbles sharpened — C2, conceitual)
7. Mudança 4 (contrato prediction/markers — o redesenho central)
8. Mudança 5 (GradNormCallback — o instrumento de diagnóstico)
9. Mudança 9 (notebook — como tudo se encaixa)
10. Mudança 10 (visualização — o resultado)
