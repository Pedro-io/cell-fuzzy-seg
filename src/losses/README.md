# Losses

> 🇧🇷 [Português](#português) · 🇺🇸 [English](#english)

---

## Português

### Visão geral

Este módulo contém todas as funções de perda e regularização utilizadas no treinamento dos modelos de segmentação celular. As perdas estão organizadas em dois grupos: **segmentação** e **regularização**.

---

### Estrutura

```
losses/
├── __init__.py                       # Exporta todos os símbolos públicos
│
├── loss_term.py                      # LossTerm (ABC — interface do padrão Strategy)
├── terms.py                          # Termos concretos: SizeTerm, TVTerm, DMapTerm…
├── loss_composer.py                  # LossComposer (composição livre de termos)
│
├── rmse_loss.py                      # RMSELoss
├── rmse_accuracy.py                  # RMSEAccuracy
├── soft_dice_loss.py                 # SoftDiceLoss
│
├── object_size_loss.py               # ObjectSizeLoss
├── total_variation_loss.py           # TotalVariationLoss
├── distance_map_loss.py              # DistanceMapLoss
├── border_loss.py                    # BorderLoss
├── not_too_thin_loss.py              # NotTooThinLoss
```

---

### Perdas de Segmentação

Medem a qualidade da predição em relação ao ground truth.

#### `RMSELoss`
**Arquivo:** `rmse_loss.py`

Calcula o *Root Mean Square Error* como `sqrt(MSE(pred, gt))`. Utilizada como função de perda principal quando se deseja penalizar erros grandes de forma quadrática.

#### `RMSEAccuracy`
**Arquivo:** `rmse_accuracy.py`

Métrica derivada do RMSE calculada como `1 - RMSE(pred, gt)`. Quanto mais próximo de 1, melhor a predição. Útil para monitoramento durante o treinamento.

#### `SoftDiceLoss`
**Arquivo:** `soft_dice_loss.py`

Versão diferenciável do coeficiente de Dice. Lida bem com desbalanceamento de classes (células pequenas em fundo dominante). Recebe tensores de forma `(B, C, H, W)` e retorna a perda média sobre o batch e canais.

```python
loss_fn = SoftDiceLoss(epsilon=1e-9)
loss = loss_fn(y_pred, y_true)
```

---

### Perdas de Regularização

Penalizam comportamentos indesejados na predição, independentemente do ground truth de segmentação.

#### `ObjectSizeLoss`
**Arquivo:** `object_size_loss.py`

Penaliza desvios no tamanho total da predição em relação ao ground truth. Calcula a razão `sum(pred) / sum(gt)` ponderada por um escalar. Evita que o modelo preveja marcadores excessivamente grandes ou pequenos.

#### `TotalVariationLoss`
**Arquivo:** `total_variation_loss.py`

Penaliza variações abruptas entre pixels vizinhos nas direções horizontal e vertical. Incentiva predições espacialmente suaves. Normalizada pela raiz da massa total do ground truth para ser invariante à densidade de objetos.

#### `DistanceMapLoss`
**Arquivo:** `distance_map_loss.py`

Penaliza ativações em regiões de alta distância a partir de um mapa de distâncias pré-calculado. Empurra os marcadores preditos para o interior das células, longe das bordas entre objetos.

#### `BorderLoss`
**Arquivo:** `border_loss.py`

Penaliza ativações nas bordas da imagem dentro de uma margem configurável (em pixels). Suprime falsos positivos nas extremidades do campo de visão, onde a informação é frequentemente incompleta.

#### `NotTooThinLoss`
**Arquivo:** `not_too_thin_loss.py`

Penaliza estruturas muito finas (filamentos) na predição. Aplica uma abertura morfológica (erosão seguida de dilatação) para identificar regiões que seriam eliminadas por serem finas demais e penaliza sua presença. Requer um kernel morfológico como parâmetro.

```python
kernel = torch.ones(5, 5)
loss_fn = NotTooThinLoss(kernel=kernel, weight=0.5)
loss = loss_fn(image)
```


---

### Padrão Strategy — Composição Livre de Losses

O padrão Strategy permite montar qualquer combinação de funções de perda sem alterar o código-fonte, bastando trocar a lista de termos passada ao `LossComposer`.

#### `LossTerm`
**Arquivo:** `loss_term.py`

Classe base abstrata (`nn.Module` + `ABC`) que define a interface comum a todos os termos. Subclasses devem implementar a propriedade `name` e o método `compute(ctx)`.

O argumento `ctx` é um dicionário com as chaves:

| Chave | Forma | Descrição |
|-------|-------|-----------|
| `"markers"` | `(N, C, H, W)` | Marcadores preditos |
| `"distance_maps"` | `(N, C, H, W)` | Mapas de distância |
| `"gt_masks"` | `(N, C, H, W)` | Máscaras ground truth |

Cada termo extrai do contexto apenas as chaves que precisa.

#### `LossComposer`
**Arquivo:** `loss_composer.py`

Recebe uma lista de `LossTerm` no construtor e os registra como submódulos PyTorch (`nn.ModuleList`). Retorna o loss total e um log por termo.

```python
from src.losses import LossComposer, SizeTerm, TVTerm, DMapTerm

# Experimento A
composer = LossComposer([SizeTerm(0.1), TVTerm(0.05)])

# Experimento B — mesma classe, outra composição, zero mudança no fonte
composer = LossComposer([SizeTerm(0.2), DMapTerm(0.1)])

total_loss, log = composer(markers, distance_maps, gt_masks)
# log = {"size": tensor, "dmap": tensor}
```

#### Termos disponíveis
**Arquivo:** `terms.py`

| Classe | `name` | Entradas do contexto |
|--------|--------|----------------------|
| `SizeTerm(weight)` | `"size"` | `markers`, `gt_masks` |
| `TVTerm(weight, power)` | `"tv"` | `markers`, `gt_masks` |
| `DMapTerm(weight)` | `"dmap"` | `markers`, `distance_maps`, `gt_masks` |
| `BorderTerm(weight, border_size)` | `"border"` | `markers` |
| `DiceTerm(epsilon)` | `"dice"` | `markers`, `gt_masks` |
| `RMSETerm()` | `"rmse"` | `markers`, `gt_masks` |
| `NotTooThinTerm(kernel, weight)` | `"not_too_thin"` | `markers` |

#### `TrainingStep`
**Arquivo:** `src/pipeline/steps/training_step.py`

Passo de pipeline que injeta o `LossComposer` no fluxo de dados. Consome `"markers"`, `"distance_maps"` e `"gt_masks"` do dicionário e adiciona `"loss"` e `"loss_log"`. O laço de treinamento chama `loss.backward()` e `optimizer.step()` após o step.

```python
from src.losses import LossComposer, SizeTerm, TVTerm
from src.pipeline.steps.training_step import TrainingStep

step = TrainingStep(LossComposer([SizeTerm(0.1), TVTerm(0.05)]))
data = step(data)
data["loss"].backward()
optimizer.step()
```

#### Criando um termo customizado

```python
from src.losses import LossTerm
import torch
from typing import Dict

class MyTerm(LossTerm):
    @property
    def name(self) -> str:
        return "my_term"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        markers = ctx["markers"]
        # ... lógica customizada ...
        return loss_value
```

---

### Como importar

Todos os símbolos públicos são exportados pelo `__init__.py` do pacote:

```python
# Primitivos
from src.losses import (
    RMSELoss, RMSEAccuracy, SoftDiceLoss,
    ObjectSizeLoss, TotalVariationLoss, DistanceMapLoss,
    BorderLoss, NotTooThinLoss,
)

# Padrão Strategy
from src.losses import (
    LossTerm,       # interface base
    LossComposer,   # composição
    SizeTerm, TVTerm, DMapTerm,
    BorderTerm, DiceTerm, RMSETerm, NotTooThinTerm,
)
```

---
---

## English

### Overview

This module contains all loss functions and regularization terms used during training of the cell segmentation models. Losses are organized into two groups: **segmentation** and **regularization**.

---

### Structure

```
losses/
├── __init__.py                       # Exports all public symbols
│
├── loss_term.py                      # LossTerm (ABC — Strategy pattern interface)
├── terms.py                          # Concrete terms: SizeTerm, TVTerm, DMapTerm…
├── loss_composer.py                  # LossComposer (free composition of terms)
│
├── rmse_loss.py                      # RMSELoss
├── rmse_accuracy.py                  # RMSEAccuracy
├── soft_dice_loss.py                 # SoftDiceLoss
│
├── object_size_loss.py               # ObjectSizeLoss
├── total_variation_loss.py           # TotalVariationLoss
├── distance_map_loss.py              # DistanceMapLoss
├── border_loss.py                    # BorderLoss
├── not_too_thin_loss.py              # NotTooThinLoss
```

---

### Segmentation Losses

Measure prediction quality against the ground truth mask.

#### `RMSELoss`
**File:** `rmse_loss.py`

Computes the *Root Mean Square Error* as `sqrt(MSE(pred, gt))`. Used as a primary training loss when large errors should be penalized quadratically.

#### `RMSEAccuracy`
**File:** `rmse_accuracy.py`

A monitoring metric computed as `1 - RMSE(pred, gt)`. Values closer to 1 indicate better predictions. Not a loss — intended for tracking during training.

#### `SoftDiceLoss`
**File:** `soft_dice_loss.py`

Differentiable Dice coefficient loss. Handles class imbalance well (small cells against a dominant background). Accepts tensors of shape `(B, C, H, W)` and returns the mean loss over the batch and channels.

```python
loss_fn = SoftDiceLoss(epsilon=1e-9)
loss = loss_fn(y_pred, y_true)
```

---

### Regularization Losses

Penalize undesirable prediction behaviours independently of the segmentation ground truth.

#### `ObjectSizeLoss`
**File:** `object_size_loss.py`

Penalizes deviations in the total predicted activation relative to the ground truth mass. Computes the weighted ratio `sum(pred) / sum(gt)`. Prevents the model from producing markers that are systematically too large or too small.

#### `TotalVariationLoss`
**File:** `total_variation_loss.py`

Penalizes abrupt changes between neighboring pixels in both spatial directions. Encourages spatially smooth predictions. Normalized by the square root of the ground truth total mass to remain invariant to object density.

#### `DistanceMapLoss`
**File:** `distance_map_loss.py`

Penalizes activations in high-distance regions using a precomputed distance map. Pushes predicted markers towards cell interiors and away from inter-object boundaries.

#### `BorderLoss`
**File:** `border_loss.py`

Penalizes activations within a configurable pixel margin at image borders. Suppresses false positives at image edges where contextual information is often incomplete.

#### `NotTooThinLoss`
**File:** `not_too_thin_loss.py`

Penalizes thin, filament-like structures in the prediction. Applies a morphological opening (erosion followed by dilation) to identify regions that would be removed for being too thin, then penalizes their presence. Requires a morphological kernel as a constructor parameter.

```python
kernel = torch.ones(5, 5)
loss_fn = NotTooThinLoss(kernel=kernel, weight=0.5)
loss = loss_fn(image)
```

### Strategy Pattern — Free Loss Composition

The Strategy pattern lets you assemble any combination of loss functions without modifying source code — just swap the list of terms passed to `LossComposer`.

#### `LossTerm`
**File:** `loss_term.py`

Abstract base class (`nn.Module` + `ABC`) defining the common interface for all terms. Subclasses must implement the `name` property and the `compute(ctx)` method.

The `ctx` argument is a dictionary with the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `"markers"` | `(N, C, H, W)` | Predicted markers |
| `"distance_maps"` | `(N, C, H, W)` | Precomputed distance maps |
| `"gt_masks"` | `(N, C, H, W)` | Ground truth masks |

Each term reads only the keys it needs from the context.

#### `LossComposer`
**File:** `loss_composer.py`

Accepts a list of `LossTerm` at construction and registers them as PyTorch submodules (`nn.ModuleList`). Returns the total loss and a per-term log.

```python
from src.losses import LossComposer, SizeTerm, TVTerm, DMapTerm

# Experiment A
composer = LossComposer([SizeTerm(0.1), TVTerm(0.05)])

# Experiment B — same class, different composition, zero source changes
composer = LossComposer([SizeTerm(0.2), DMapTerm(0.1)])

total_loss, log = composer(markers, distance_maps, gt_masks)
# log = {"size": tensor, "dmap": tensor}
```

#### Available terms
**File:** `terms.py`

| Class | `name` | Required context keys |
|-------|--------|-----------------------|
| `SizeTerm(weight)` | `"size"` | `markers`, `gt_masks` |
| `TVTerm(weight, power)` | `"tv"` | `markers`, `gt_masks` |
| `DMapTerm(weight)` | `"dmap"` | `markers`, `distance_maps`, `gt_masks` |
| `BorderTerm(weight, border_size)` | `"border"` | `markers` |
| `DiceTerm(epsilon)` | `"dice"` | `markers`, `gt_masks` |
| `RMSETerm()` | `"rmse"` | `markers`, `gt_masks` |
| `NotTooThinTerm(kernel, weight)` | `"not_too_thin"` | `markers` |

#### `TrainingStep`
**File:** `src/pipeline/steps/training_step.py`

Pipeline step that injects a `LossComposer` into the data flow. Consumes `"markers"`, `"distance_maps"`, and `"gt_masks"` from the data dict and adds `"loss"` and `"loss_log"`. The training loop calls `loss.backward()` and `optimizer.step()` after the step.

```python
from src.losses import LossComposer, SizeTerm, TVTerm
from src.pipeline.steps.training_step import TrainingStep

step = TrainingStep(LossComposer([SizeTerm(0.1), TVTerm(0.05)]))
data = step(data)
data["loss"].backward()
optimizer.step()
```

#### Writing a custom term

```python
from src.losses import LossTerm
import torch
from typing import Dict

class MyTerm(LossTerm):
    @property
    def name(self) -> str:
        return "my_term"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        markers = ctx["markers"]
        # ... custom logic ...
        return loss_value
```

---

### Importing

All public symbols are exported from the package `__init__.py`:

```python
# Primitives
from src.losses import (
    RMSELoss, RMSEAccuracy, SoftDiceLoss,
    ObjectSizeLoss, TotalVariationLoss, DistanceMapLoss,
    BorderLoss, NotTooThinLoss,
)

# Strategy pattern
from src.losses import (
    LossTerm,       # base interface
    LossComposer,   # composition
    SizeTerm, TVTerm, DMapTerm,
    BorderTerm, DiceTerm, RMSETerm, NotTooThinTerm,
)
```
