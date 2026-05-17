# Jornal de Desenvolvimento

---

## [2026-05-17] Notebook de Experimento Template

### Por que fizemos isso?

Depois de implementar o MarkerNet e o sistema de losses, o próximo passo natural era rodar um treinamento real. O problema: cada vez que ia experimentar, precisava reescrever o mesmo boilerplate — criar DataLoader, montar o loop de treino, configurar optimizer, salvar checkpoint, plotar curvas. Com múltiplos experimentos planejados, isso virava fonte de erro e inconsistência.

A solução foi criar um template que:
1. Centraliza todos os hiperparâmetros em blocos no topo (fácil comparar experimentos)
2. Resolve o problema do `collate_fn` de uma vez
3. Serve de referência para o que o `MarkerStep` precisará fazer quando implementado

---

### O problema do `collate_fn`

O `DataLoader` padrão não funciona direto com o `MonusegDataset` por três razões:

**1. Shapes diferentes:** imagens do MoNuSeg têm tamanhos distintos. O stack em batch exige shapes iguais.

**2. O 4º canal:** `MarkerNet` espera `(N, 4, H, W)`. Os 3 primeiros canais são RGB da imagem. O 4º precisa ser construído — por enquanto é o `ground_truth` como proxy da máscara Cellpose (quando o pipeline estiver completo, será a saída real do `CellposeStep`).

**3. O `distance_map`:** A `DistanceMapLoss` precisa de um mapa de distância euclidiana calculado a partir do `ground_truth`. Isso não é parte do dataset — precisa ser computado na hora do batch.

O `collate_fn` resolve tudo isso em um lugar só, antes de os tensores chegarem no modelo.

---

### Decisão de design: backprop fora do `MarkerNet.train_step()`

O notebook usa `model.forward()` + `loss.backward()` + `optimizer.step()` diretamente em vez de `model.train_step()`. Por quê?

O `train_step()` encapsula tudo internamente (incluindo `zero_grad`). Mas no notebook é mais claro expor cada passo separadamente para fins didáticos e para ter controle explícito do loop (ex.: gradient clipping, logging granular). Os dois caminhos funcionam — `train_step()` é para quando você confia na abstração, o loop explícito é para quando precisa de controle total.

---

### Como usar o template

1. Copie `notebooks/experiment_template.ipynb`
2. Mude `EXPERIMENT_NAME`, `MODEL_CONFIG`, `TRAIN_CONFIG`, e a lista `LOSS_TERMS`
3. Rode tudo — os outputs ficam em `results/{EXPERIMENT_NAME}/`
4. Preencha a tabela da seção 11 com os resultados

Os checkpoints ficam em `results/{EXPERIMENT_NAME}/checkpoints/best.pt`. Para retomar:
```python
model = MarkerNet(config=MODEL_CONFIG)
model.load("results/exp_001_baseline/checkpoints/best.pt")
```

---

## [2026-05-11] Implementação do MarkerNet e Dataset Handling

### Por que fizemos isso?

As abstrações (`BaseNetwork`, `BaseDataset`) estavam definidas há semanas mas sem implementação concreta. Com o sistema de losses pronto (entry anterior), a prioridade virou ter uma rede real para treinar e um dataset confiável para alimentá-la.

---

### O que mudou no `MarkerNet`

A implementação concreta do `MarkerNet` saiu do rascunho e virou código funcional:

- **Configuração via dict:** em vez de parâmetros fixos no construtor, o `MarkerNet` aceita um `config: dict` com `encoder_name`, `pretrained`, `in_channels`, `threshold`. Isso alinha com o padrão de configuração do projeto (YAML → dict → modelo).

- **`optimizer.zero_grad()` dentro do `train_step()`:** antes, era responsabilidade do chamador zerar os gradientes. Agora o `train_step()` faz isso internamente antes do forward pass. Elimina uma classe de bug onde você esquecia o zero_grad e gradientes acumulavam entre iterações.

- **`get_config()` adicionado:** a interface `BaseNetwork` ganhou um método abstrato `get_config() -> dict` que toda subclasse deve implementar. O `MarkerNet` retorna `self._config`. Isso permite recriar um modelo com exatamente a mesma configuração ao carregar um checkpoint — sem precisar manter o dict de config separado.

- **`evaluate()` implementado:** itera sobre o DataLoader, chama `predict()` em cada batch, acumula métricas. Retorna a média de cada métrica. As métricas são passadas como lista de callables (qualquer objeto com `__call__(pred, target) -> Tensor`).

---

### O que mudou no `MonusegDataset`

- **Suporte a `transform`:** `__getitem__` agora verifica `self.transform is not None` e aplica antes de retornar o sample. A transform recebe e deve retornar o dict completo — não só a imagem.

- **`_load_mask()` mais robusto:** ganhou o parâmetro `image_shape` opcional. Para máscaras `.xml`, o shape é necessário para criar a array com `cv2.fillPoly()`. Se não for passado, o método carrega a imagem correspondente só para inferir o shape. Isso evita passar o shape explicitamente em todo caller mas mantém a opção quando ele já está disponível (economiza I/O).

---

### O que NÃO mudou

- As abstrações `BaseDataset` e `BaseNetwork` continuam com a mesma interface pública.
- O sistema de losses (`LossComposer`, `LossTerm`, etc.) não foi alterado.
- O pipeline de inferência (`ModelPipeline`, steps) continua igual.

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
