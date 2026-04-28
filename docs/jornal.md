# Estrutura do repo
Claro — aqui está o relatório em **Markdown pronto para uso** (você pode salvar como `ARCHITECTURE.md` no seu projeto 👇)

---

```markdown
# 🧠 📄 Arquitetura do Projeto — cell-fuzzy-seg

---

## 1. 🎯 Objetivo do sistema

Construir um pipeline modular para:

- Processar imagens
- Gerar marcadores (via redes neurais)
- Aplicar segmentação baseada em marcadores
- Otimizar tudo end-to-end com loss composta

Fluxo geral:

```

Imagem → Marcadores → Segmentação → Loss

```

---

## 2. 🏗️ Visão em camadas

```

[ Interface (notebooks / scripts) ]
↓
[ Training / Inference ]
↓
[ Pipeline ]
↓
[ Models (Networks) ]
↓
[ Data ]
↓
[ Utils / Infra ]

```

---

## 3. 📦 Camadas do sistema

---

### 🟦 3.1 `data/` — Camada de dados

#### 🎯 Responsabilidade
- Carregar dados
- Aplicar transformações
- Retornar batches padronizados

#### 📁 Estrutura

```

src/data/
├── dataset.py
├── dataloader.py
└── transforms.py

````

#### 🧩 Exemplo

```python
class CellDataset:
    def __getitem__(self, idx):
        return {
            "image": ...,
            "ground_truth": ...,
            "cues": ...
        }
````

#### ⚠️ Regras

* ❌ Não usar modelos
* ❌ Não calcular loss

---

### 🟩 3.2 `models/networks/` — Redes neurais

#### 🎯 Responsabilidade

* Definir modelos (PyTorch)

#### 📁 Estrutura

```
src/models/networks/
├── feature_net.py
├── marker_net.py
└── segmentation_net.py
```

#### 🧩 Exemplo

```python
class MarkerNet(nn.Module):
    def forward(self, x, cues=None):
        return markers
```

#### ⚠️ Regras

* ❌ Não acessar `data dict`
* ❌ Não conhecer pipeline
* ❌ Não calcular loss

---

### 🟨 3.3 `pipeline/` — Orquestração

#### 🎯 Responsabilidade

* Controlar fluxo entre etapas
* Encadear redes

#### 📁 Estrutura

```
src/pipeline/
├── pipeline.py
├── base_step.py
└── steps/
    ├── feature_step.py
    ├── marker_step.py
    └── segmentation_step.py
```

#### 🧩 Pipeline

```python
class ModelPipeline:
    def __init__(self, steps):
        self.steps = steps

    def forward(self, data):
        for step in self.steps:
            data = step.forward(data)
        return data
```

#### 🧩 Step base

```python
class PipelineStep:
    def forward(self, data: dict) -> dict:
        raise NotImplementedError
```

#### 🧩 Exemplos de steps

* FeatureStep → adiciona `features`
* MarkerStep → adiciona `markers`
* SegmentationStep → adiciona `segmentation`

#### ⚠️ Regras

* ✔ Sempre usar `data: dict`
* ❌ Não fazer backprop aqui

---

### 🟥 3.4 `losses/` — Funções de perda

#### 🎯 Responsabilidade

* Calcular loss total

#### 📁 Estrutura

```
src/losses/
└── composite_loss.py
```

#### 🧩 Exemplo

```python
class CompositeLoss:
    def __call__(self, data):
        seg = data["segmentation"]
        gt = data["ground_truth"]
        markers = data["markers"]

        return loss_seg + loss_reg
```

#### ⚠️ Regras

* ❌ Não acessar modelos diretamente
* ❌ Não modificar pipeline

---

### 🟪 3.5 `training/` — Treinamento

#### 🎯 Responsabilidade

* Loop de treino
* Backpropagation
* Otimização

#### 📁 Estrutura

```
src/training/
├── trainer.py
└── evaluator.py
```

#### 🧩 Exemplo

```python
class Trainer:
    def __init__(self, pipeline, loss_fn, optimizer):
        self.pipeline = pipeline
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, batch):
        output = self.pipeline.forward(batch)
        loss = self.loss_fn(output)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
```

#### ⚠️ Regras

* ❌ Não definir arquitetura
* ❌ Não implementar redes

---

### 🟫 3.6 `registry/` — Registro dinâmico

#### 🎯 Responsabilidade

* Mapear nomes → classes

#### 📁 Estrutura

```
src/registry/
└── model_registry.py
```

#### 🧩 Exemplo

```python
REGISTRY = {}

def register(name):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator
```

#### 🧩 Uso

```python
@register("marker_net")
class MarkerNet(nn.Module):
    ...
```

---

### ⚙️ 3.7 `configs/` — Configuração

#### 🎯 Responsabilidade

* Definir experimentos

#### 🧩 Exemplo

```yaml
pipeline:
  - marker_step
  - segmentation_step

models:
  marker_step: marker_net
  segmentation_step: segmentation_net
```

---

### 🟨 3.8 `scripts/` — Execução

#### 🎯 Responsabilidade

* Rodar o sistema

#### 📁 Estrutura

```
scripts/
├── train.py
├── evaluate.py
└── inference.py
```

#### ⚠️ Regras

* ❌ Não conter lógica pesada
* ✔ Apenas orquestrar execução

---

### 🟦 3.9 `notebooks/` — Experimentação

#### 🎯 Responsabilidade

* Debug
* Visualização
* Testes rápidos

#### ⚠️ Regras

* ❌ Não conter lógica do sistema
* ✔ Importar de `src/`

---

## 4. 🧠 Contrato de dados (CRÍTICO)

Formato padrão:

```python
data = {
    "image": ...,
    "cues": ...,
    "features": ...,
    "markers": ...,
    "segmentation": ...,
    "ground_truth": ...
}
```

### 📌 Regra de ouro

> Cada etapa lê e escreve nesse dicionário

---

## 5. 🔁 Fluxo completo

```
Dataset → batch(dict)
        ↓
Pipeline
        ↓
FeatureStep
        ↓
MarkerStep
        ↓
SegmentationStep
        ↓
Loss
        ↓
Trainer (backprop)
```

---

## 6. 🚀 Checklist de implementação

### 🔹 Fase 1 — Mínimo funcional

* [ ] Dataset
* [ ] MarkerStep
* [ ] SegmentationStep
* [ ] Pipeline
* [ ] Loss
* [ ] Trainer

---

### 🔹 Fase 2 — Organização

* [ ] Configs
* [ ] Registry
* [ ] Scripts

---

### 🔹 Fase 3 — Avançado

* [ ] Logging
* [ ] Visualização de markers
* [ ] Experimentos múltiplos

---

## 7. ⚠️ Anti-patterns (EVITAR)

* ❌ Lógica de pipeline dentro de models
* ❌ Loss dentro da rede
* ❌ Notebook com código crítico
* ❌ Dados fora do padrão dict
* ❌ Scripts com lógica duplicada

---

## 8. 💬 Resumo

Você está construindo:

* Um pipeline modular de deep learning
* Com separação clara entre:

  * dados
  * modelo
  * fluxo
  * treino

### 🎯 Benefícios

* Escalabilidade
* Reprodutibilidade
* Facilidade de experimentação
* Código limpo

---

# Sobre o Poetry
Optamos por remover o poetry devido a necessidade de utilizar o Google colab para rodar o projeto. 
Com isso temos uma perda significativa em relação ao controle de depêndencias do repositório, porém é necessário devido a características do Colab

