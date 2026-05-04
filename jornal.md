# Jornal de Desenvolvimento

---

## [2026-05-04] Padrão Strategy para Funções de Perda

### Por que fizemos isso?

Antes desta mudança, para treinar a rede com um conjunto de funções de perda era necessário instanciar o `MultiRegularization` passando todos os pesos diretamente:

```python
reg = MultiRegularization(size=0.1, tv=0.05, dmap=0.1, topo_weight=0.2)
```

O problema: **a escolha das losses estava presa no código-fonte**. Para testar uma combinação diferente — digamos, só Dice + Topologia — você precisava ir no arquivo, comentar parâmetros, mudar valores, e torcer para não quebrar nada. Com dezenas de experimentos isso vira um pesadelo de controle de versão.

A pergunta que motivou a mudança foi:

> "Posso montar o conjunto de losses de fora, como uma configuração, sem tocar no fonte?"

A resposta é o **padrão Strategy**.

---

### O que é o Padrão Strategy?

É um padrão de projeto clássico. A ideia é simples:

- Você tem um **comportamento que muda** (quais losses usar, com quais pesos).
- Em vez de hardcodar esse comportamento, você define uma **interface comum** e injeta qual implementação usar na hora de criar o objeto.

Analogia do dia-a-dia: imagine uma cafeteira que aceita qualquer cápsula padronizada. Você não troca a cafeteira para mudar o sabor — só troca a cápsula. O `LossComposer` é a cafeteira. Cada `LossTerm` é uma cápsula.

---

### O que foi criado — peça por peça

#### 1. `LossTerm` — a interface (o contrato da cápsula)

**Arquivo:** `src/losses/loss_term.py`

É uma classe abstrata. Ela define o contrato que toda função de perda deve respeitar:

```
LossTerm
├── name  → uma string identificando o termo (ex: "size", "tv")
└── compute(ctx) → recebe um dicionário com os tensores e devolve um escalar
```

O `ctx` (contexto) é um dict com três chaves:
- `"markers"` — o que a rede prediz
- `"distance_maps"` — mapas de distância pré-calculados
- `"gt_masks"` — o ground truth

Cada term só lê as chaves que precisa. Isso resolve o problema de assinaturas diferentes entre losses (algumas precisam de `distance_maps`, outras não).

**Ninguém instancia `LossTerm` diretamente** — ela só existe para garantir que todos os termos concretos tenham a mesma cara.

---

#### 2. `terms.py` — as implementações concretas (as cápsulas)

**Arquivo:** `src/losses/terms.py`

São 8 classes finas, cada uma embrulhando uma loss existente e adaptando sua assinatura para o contrato `LossTerm`:

| Classe | O que faz | Loss interna |
|--------|-----------|--------------|
| `SizeTerm` | Penaliza tamanho errado | `ObjectSizeLoss` |
| `TVTerm` | Penaliza variações bruscas | `TotalVariationLoss` |
| `DMapTerm` | Penaliza ativação longe do centro | `DistanceMapLoss` |
| `TopologyTerm` | Controla número de máximos | `TopologyLoss` |
| `BorderTerm` | Penaliza bordas da imagem | `BorderLoss` |
| `DiceTerm` | Mede sobreposição com GT | `SoftDiceLoss` |
| `RMSETerm` | Erro quadrático médio | `RMSELoss` |
| `NotTooThinTerm` | Penaliza estruturas finas | `NotTooThinLoss` |

Exemplo de como um term é por dentro:

```python
class SizeTerm(LossTerm):
    def __init__(self, weight=0.1):
        super().__init__()
        self._loss = ObjectSizeLoss(weight=weight)  # loss original

    @property
    def name(self):
        return "size"

    def compute(self, ctx):
        # puxa só o que precisa do contexto
        return self._loss(ctx["markers"], ctx["gt_masks"])
```

Note que o `weight` ainda é configurado aqui. O term é instanciado **uma vez** com os parâmetros do experimento.

---

#### 3. `LossComposer` — o compositor (a cafeteira)

**Arquivo:** `src/losses/loss_composer.py`

Recebe uma lista de `LossTerm` e os soma, devolvendo o total e um log por termo.

```python
composer = LossComposer([SizeTerm(0.1), TVTerm(0.05)])
total, log = composer(markers, distance_maps, gt_masks)
# total = soma de todos os termos
# log   = {"size": tensor(0.012), "tv": tensor(0.003)}
```

Internamente, o `LossComposer`:
1. Monta o `ctx` com os três tensores
2. Passa o `ctx` para cada term
3. Soma os resultados
4. Retorna o total e o dicionário de log

Os terms são registrados como submódulos PyTorch (`nn.ModuleList`), o que significa que seus parâmetros aparecem no `state_dict` do composer automaticamente.

---

#### 4. `TrainingStep` — o passo de pipeline

**Arquivo:** `src/pipeline/steps/training_step.py`

Liga o `LossComposer` ao sistema de pipeline que já existia no projeto. Cada step recebe um `data: dict`, processa, e devolve o dict atualizado.

```
data entra com:   "markers", "distance_maps", "gt_masks"
data sai com:     + "loss" (tensor escalar), + "loss_log" (dict por termo)
```

O laço de treinamento, depois do step, faz:
```python
data["loss"].backward()
optimizer.step()
```

O step em si **não faz backward** — ele só calcula e armazena. Quem decide quando otimizar é o laço externo.

---

### Como montar um experimento agora

```python
from src.losses import LossComposer, DiceTerm, TVTerm, TopologyTerm
from src.pipeline.steps.training_step import TrainingStep

# Monta a composição desejada — zero mudança no fonte
step = TrainingStep(
    LossComposer([
        DiceTerm(epsilon=1e-9),
        TVTerm(weight=0.05),
        TopologyTerm(weight=0.2, num_components=3),
    ])
)

# Usa no pipeline normalmente
data = step(data)
```

Para o próximo experimento, você só muda a lista. O resto do código não sabe nem liga.

---

### Diagrama mental

```
ANTES
─────
MultiRegularization(size=0.1, tv=0.05, dmap=0.1, topo=0.2)
  └── todos os pesos hardcoded no construtor
  └── para mudar: editar o fonte ← PROBLEMA

DEPOIS
──────
LossComposer([SizeTerm(0.1), TVTerm(0.05)])   ← Experimento A
LossComposer([DiceTerm(), TopologyTerm(0.2)]) ← Experimento B
      │
      ▼
  cada term implementa LossTerm
      │
      ▼
  compute(ctx) ──► ctx["markers"], ctx["distance_maps"], ctx["gt_masks"]
      │
      ▼
  soma tudo ──► (total_loss, log)
      │
      ▼
  TrainingStep injeta no pipeline
```

---

### O que NÃO mudou

- As losses individuais (`ObjectSizeLoss`, `TotalVariationLoss`, etc.) continuam iguais.
- O `MultiRegularization` ainda existe no arquivo, só foi removido dos exports públicos. Não foi apagado.
- O pipeline (`ModelPipeline`, `PipelineStep`) não mudou.

---

### Por onde começar a ler o código

1. Leia `src/losses/loss_term.py` — é pequeno, define tudo em ~40 linhas.
2. Leia qualquer term em `src/losses/terms.py` — `SizeTerm` é o mais simples.
3. Leia `src/losses/loss_composer.py` — veja como ele itera os terms.
4. Leia `src/pipeline/steps/training_step.py` — veja como encaixa no pipeline.
