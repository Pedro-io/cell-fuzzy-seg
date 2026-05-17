# Overview do Repositório cell-fuzzy-seg

Minhas notas de referência sobre o projeto. O objetivo é segmentar núcleos celulares em imagens histológicas (dataset MoNuSeg) usando uma abordagem híbrida: Cellpose gera uma segmentação inicial, e uma rede treinada (MarkerNet) aprende a gerar marcadores que guiam uma segmentação fuzzy downstream.

---

## Estrutura do Projeto

```
cell-fuzzy-seg/
├── configs/
│   └── datasets.yml              # Configurações dos datasets (paths, batch size, etc.)
├── data_source/                  # Dados brutos (não versionar)
│   ├── MoNuSegTrainingData/
│   └── MoNuSegTestData/
├── src/
│   ├── data/load/                # Dataset classes
│   ├── data/transforms/          # Transforms (vazio ainda)
│   ├── io/                       # Salvar outputs em disco
│   ├── losses/                   # Funções de perda e regularizadores
│   │   └── topology/             # Perda topológica (component trees com Higra)
│   ├── models/networks/          # Arquiteturas de rede
│   ├── pipeline/                 # Orquestrador de steps
│   │   └── steps/
│   └── utils/                   # Logger, image_utils
├── docs/
├── notebooks/
├── requirements.txt
└── pyproject.toml
```

---

## Fluxo Geral de Dados

```
Imagem .tif + Anotação .xml
        ↓
   MonusegDataset.__getitem__()
        ↓
   {'image': ndarray (H,W,C), 'ground_truth': ndarray (H,W)}
        ↓
   DataLoader → batch de tensores
        ↓
   MarkerNet.forward(images (N,4,H,W)) → markers (N,1,H,W)
        ↓
   LossComposer(markers, distance_maps, gt_masks) → loss scalar
        ↓
   loss.backward() + optimizer.step()
```

No pipeline de inferência:

```
image → CellposeStep → segmentation (H,W)
                              ↓
                        MarkerStep [stub]
                              ↓
                      SegmentationStep [stub]
                              ↓
                         RGBAStep → rgba (H,W,4)
                              ↓
                         OutputWriter
```

---

## 1. Configuração — `configs/datasets.yml`

Tudo que controla dataset e dataloader fica aqui. As classes de dataset leem esse arquivo via `BaseDataset.__init__()`. Keys relevantes:

- `root_dir`, `image_dir`, `mask_dir`, `image_extension`, `mask_extension`
- `train_split`: float, ex. `0.8` → 80% treino / 20% val
- `loader_config`: `batch_size`, `num_workers`, `shuffle_train`, `pin_memory`
- `preprocessing`: `image_size`, `normalize`, `mean`, `std`

---

## 2. Dataset — `src/data/load/`

### `BaseDataset` (abstrata)

Herda de `torch.utils.data.Dataset`.

**O que `torch.utils.data.Dataset` exige de mim:**
- `__len__() -> int`: quantos exemplos no dataset
- `__getitem__(idx) -> dict`: retorna um único exemplo

`BaseDataset` adiciona mais métodos abstratos que toda subclasse precisa implementar:
- `_load_image(path) -> Any`
- `_load_mask(path) -> Any`
- `_get_file_pairs() -> List[Tuple[str, str]]`

No `__init__`, ela carrega o YAML e seta `self.root_dir`, `self.config`, `self.loader_config`, `self.preprocessing`.

### `MonusegDataset(BaseDataset)`

Implementação concreta para o MoNuSeg.

**`__getitem__(idx)`** retorna:
```python
{
    'id':           str,          # nome do arquivo (sem extensão)
    'image':        np.ndarray,   # (H, W) ou (H, W, C), dtype variável
    'ground_truth': np.ndarray,   # (H, W) binário 0/1
    'meta':         dict          # {'image_path': ..., 'mask_path': ...}
}
```

Se `self.transform is not None`, o dict é passado pela transform antes de ser retornado. A transform recebe e deve retornar o dict completo (não só a imagem).

**Carregamento de imagem:** usa `cellpose.io.imread()` — lida bem com `.tif` multicanal.

**Carregamento de máscara (`.xml`):**
1. Parseia XML com as regiões anotadas
2. Para cada região, extrai vértices dos polígonos
3. Usa `cv2.fillPoly()` para preencher os polígonos em uma máscara binária
4. Retorna `np.ndarray` shape `(H, W)` com 0s e 1s

Se a máscara for `.npy`, apenas `np.load()`. `_load_mask()` recebe `image_shape` opcional — se não for fornecido e a extensão for `.xml`, carrega a imagem correspondente só para inferir o shape antes de criar a máscara.

### API auxiliar do `BaseDataset`

Métodos públicos disponíveis em toda subclasse (não precisam ser sobrescritos):

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `get_config()` | `dict` | Config específica do `config_key` (ex.: `monuseg_training`) |
| `get_loader_config()` | `dict` | Seção `loader_config` do YAML |
| `get_preprocessing_config()` | `dict` | Seção `preprocessing` do YAML |
| `get_dataset_name()` | `str` | `dataset_name` passado no construtor |
| `get_config_key()` | `str` | `config_key` passado no construtor |
| `get_image_dir()` | `str` | Path completo: `root_dir/image_dir` |
| `get_mask_dir()` | `str` | Path completo: `root_dir/mask_dir` |
| `set_transform(transform)` | `None` | Troca a transform em runtime |
| `get_stats()` | `dict` | Resumo: tamanho, config, paths |

---

## 3. Modelo — `src/models/networks/`

### `BaseNetwork` (abstrata)

Interface que toda rede deve implementar:
- `forward(x) -> Tensor`: logits/features crus
- `predict(x) -> Tensor`: predição pós-processada (ex. threshold)
- `train_step(batch, optimizer, loss_fn) -> (float, dict)`: uma iteração de treino
- `validation_step(batch, loss_fn) -> (float, dict)`: uma iteração de validação
- `evaluate(data_loader, metrics) -> dict`: avaliação completa
- `save(path)` / `load(path)`: persistência
- `get_config() -> dict`: retorna a config com que a rede foi instanciada

### `MarkerNet(BaseNetwork)`

**O que faz:** detecta os centros/marcadores dos núcleos. A ideia é que os marcadores sirvam de sementes para um algoritmo de watershed ou segmentação fuzzy posterior.

**Arquitetura interna:**
```python
self.model = smp.Unet(
    encoder_name="resnet34",   # padrão, configurável
    encoder_weights="imagenet",
    in_channels=4,             # espera 4 canais de entrada
    classes=1,                 # saída: 1 mapa de probabilidade
)
```

**Por que `smp.Unet`?** É a implementação do `segmentation_models_pytorch`. Internamente é um `nn.Module` com encoder (ResNet34) + decoder com skip connections. Eu não defino as camadas manualmente — passo os hyperparâmetros no construtor.

**`forward(x: Tensor) -> Tensor`**
- Entrada: `(N, 4, H, W)` — N = batch size, 4 canais, H×W pixels
- Internamente: `self.model(x)` → logits shape `(N, 1, H, W)`
- Aplica `torch.sigmoid()` nos logits → probabilidades em [0, 1]
- Saída: `(N, 1, H, W)`

> **`torch.sigmoid(x)`**: converte qualquer real em probabilidade. `sigmoid(0) = 0.5`, valores positivos → >0.5, negativos → <0.5. Usado aqui porque a tarefa é binária (marcador ou não).

**`predict(x) -> Tensor`**
```python
self.model.eval()
with torch.no_grad():
    probs = self.forward(x)
    return (probs >= self.threshold).float()
```

> **`torch.no_grad()`**: contexto que desabilita o grafo de autodiferenciação. Economiza memória e acelera inferência porque não preciso calcular gradientes. Uso sempre que não vou chamar `.backward()`.

> **`.eval()`**: coloca o modelo em modo de avaliação. Afeta `BatchNorm` (usa estatísticas globais em vez de batch) e `Dropout` (desativa). Preciso chamar explicitamente — o PyTorch não troca sozinho.

**`train_step(batch, optimizer, loss_fn)`**
```python
images       = batch["image"]         # (N, 4, H, W)
distance_maps = batch["distance_map"] # (N, C, H, W)
gt_masks     = batch["ground_truth"]  # (N, C, H, W)

markers = self.forward(images)        # (N, 1, H, W)
loss, loss_log = loss_fn(markers, distance_maps, gt_masks)
loss.backward()
optimizer.step()
return loss.item(), loss_log
```

> **`loss.backward()`**: calcula gradientes de `loss` em relação a todos os parâmetros que têm `requires_grad=True`. Os gradientes ficam acumulados em `.grad` de cada parâmetro.

> **`optimizer.step()`**: atualiza os parâmetros usando os gradientes acumulados. Preciso chamar `optimizer.zero_grad()` antes (ou usar `set_to_none=True`) para não acumular gradientes de iterações anteriores.

> **`loss.item()`**: extrai o valor escalar do tensor para um float Python. Não mantém grafo — uso para logging.

**`save` / `load`:**
```python
# save
torch.save({"model_state_dict": self.model.state_dict(), "config": self._config}, path)

# load
checkpoint = torch.load(path, map_location="cpu")
self.model.load_state_dict(checkpoint["model_state_dict"])
```

> **`state_dict()`**: dicionário `{nome_camada: tensor_de_pesos}`. É o que preciso salvar para preservar os pesos treinados.

> **`load_state_dict()`**: carrega os pesos de volta. `map_location="cpu"` evita erro se o modelo foi salvo em GPU e estou carregando em CPU.

**`get_config() -> dict`:** retorna `self._config` — o dicionário passado no construtor. Útil para recriar o modelo com os mesmos hyperparâmetros ao carregar um checkpoint.

---

## 4. Funções de Perda — `src/losses/`

### Arquitetura geral (Strategy Pattern)

```
LossTerm (abstrata, nn.Module)
    ├── SizeTerm       → ObjectSizeLoss
    ├── TVTerm         → TotalVariationLoss
    ├── DMapTerm       → DistanceMapLoss
    ├── TopologyTerm   → TopologyLoss
    ├── BorderTerm     → BorderLoss
    ├── DiceTerm       → SoftDiceLoss
    ├── RMSETerm       → RMSELoss
    └── NotTooThinTerm → NotTooThinLoss

LossComposer(nn.Module)
    ├── recebe List[LossTerm]
    ├── chama cada term com o contexto compartilhado
    └── retorna (total_loss, loss_log)
```

**`LossTerm`** é abstrata — cada subclasse implementa `compute(ctx: Dict[str, Tensor]) -> Tensor`. O `ctx` é um dicionário com os tensores do batch. As keys esperadas pelo contexto são:

| Key | Shape | Descrição |
|-----|-------|-----------|
| `"markers"` | `(N, C, H, W)` | Predição da rede |
| `"distance_maps"` | `(N, C, H, W)` | Mapa de distância entre núcleos |
| `"gt_masks"` | `(N, C, H, W)` | Máscara ground truth |

**`LossComposer.forward(markers, distance_maps, gt_masks)`**
- Monta o `ctx` dict
- Itera sobre os `LossTerm`s, chama `term.compute(ctx)`
- Soma tudo em `total_loss`
- Retorna `(total_loss, {term.name: valor})`

---

### Funções de perda concretas

#### `SoftDiceLoss` — `src/losses/soft_dice_loss.py`

```
Entrada: y_pred (B,C,H,W), y_true (B,C,H,W)
Saída: scalar ∈ [0, 1]
```

**Fórmula:**
```
dice_per_channel = (2 * sum(pred * true) + eps) / (sum(pred² + true²) + eps)
loss = 1 - mean(dice_per_channel)
```

Dice Loss é uma perda baseada no coeficiente de sobreposição. É diferenciável porque `y_pred` é contínuo (saída do sigmoid, não binário). O `eps` evita divisão por zero. Loss = 0 quando predição = ground truth perfeito.

---

#### `RMSELoss` — `src/losses/rmse_loss.py`

```
Entrada: y_pred, y_true (qualquer shape)
Saída: scalar = sqrt(mean((y_pred - y_true)²))
```

Internamente usa `nn.MSELoss()` e tira a raiz.

> **`nn.MSELoss()`**: `nn.Module` que calcula `mean((input - target)²)`. Por padrão `reduction='mean'`.

---

#### `ObjectSizeLoss` — `src/losses/object_size_loss.py`

```
Entrada: y_pred (N,C,H,W), y_true (N,C,H,W)
Saída: weight * sum(y_pred) / sum(y_true)
```

Penaliza quando a massa total da predição diverge da massa do ground truth. Se prevejo muito mais pixels do que há de verdade, essa perda sobe.

---

#### `DistanceMapLoss` — `src/losses/distance_map_loss.py`

```
Entrada: y_pred (N,C,H,W), distance_map (N,C,H,W), y_true (N,C,H,W)
Saída: weight * sum(y_pred * distance_map) / sum(y_true)
```

O `distance_map` tem valores altos nas regiões entre núcleos. Essa perda penaliza predições que aparecem longe dos centros (onde o marcador deveria estar). Quanto mais o marcador previsto coincidir com regiões de alta distância, maior a perda.

---

#### `TotalVariationLoss` — `src/losses/total_variation_loss.py`

```
Entrada: y_pred (N,C,H,W), y_true (N,C,H,W)
Saída: weight * (TV_h + TV_w) / sqrt(sum(y_true))
```

**TV (Total Variation):**
```
TV_h = sum(|y_pred[:,:,1:,:] - y_pred[:,:,:-1,:]|^power)  # diferença vertical
TV_w = sum(|y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]|^power)  # diferença horizontal
```

Penaliza variações bruscas na predição → favorece outputs espacialmente suaves. É um regularizador clássico de imagens.

---

#### `BorderLoss` — `src/losses/border_loss.py`

```
Entrada: y_pred (N,1,H,W)
Saída: weight * sum(y_pred * border_mask) / (N*C*H*W)
```

Cria uma máscara binária com `border_size` pixels nas bordas da imagem. Penaliza predições que caem nessa região — evita detecções espúrias nas bordas.

---

#### `NotTooThinLoss` — `src/losses/not_too_thin_loss.py`

```
Entrada: image (H, W) — um único slice 2D
Saída: scalar
```

Aplica abertura morfológica (erosão seguida de dilatação) de forma diferenciável usando `F.unfold()`. Penaliza estruturas finas/filamentosas que desaparecem após a erosão. O objetivo é que os marcadores tenham uma "massa" mínima.

> **`F.unfold(input, kernel_size, padding)`**: extrai patches locais de um tensor 4D como colunas. Shape `(N, C*kH*kW, L)` onde L = número de patches. Permite operações de vizinhança (como morfologia) de forma vetorizada.

---

#### `TopologyLoss` — `src/losses/topology/topology_loss.py`

A mais complexa. Controla quantos máximos proeminentes existem na predição — queremos que o número de máximos corresponda ao número de núcleos.

```
Entrada: markers (N,C,H,W)
Saída: weight * mean(loss_per_channel)
```

**Parâmetros principais:** `weight`, `num_target_maxima`, `margin=1.0`, `power=2`, `cpus=2`.

**Funcionamento interno:**
1. Achata o batch: `(N,C,H,W)` → lista de `N*C` slices 2D `(H,W)`
2. Para cada slice (em paralelo via `multiprocessing.Pool`):
   - Constrói uma max-tree via Higra (`ComponentTree("max")`)
   - Calcula a *dinâmica* de cada máximo: `extrema_altitude - saddle_altitude`
   - Ordena as dinâmicas em ordem decrescente
3. Aplica `_loss_ranked_selection()` em cada slice:
   - Os top `num_target_maxima` devem ter dinâmica > `margin` → penaliza se não tiverem
   - O restante deve ter dinâmica ≈ 0 → penaliza qualquer máximo extra
4. Soma tudo e divide por `N*C` para a média

**Paralelismo:** usa `get_context("spawn").Pool(cpus)` para processar canais simultaneamente. O `"spawn"` é necessário para compatibilidade com PyTorch + Higra (evita deadlocks do `fork` com threads CUDA).

**`ComponentTreeFunction(autograd.Function)`**: wrapper autograd customizado para a construção da max-tree. Permite que gradientes fluam da perda de volta para `vertex_weights` (os pixels da predição).

> **`torch.autograd.Function`**: classe base para operações com gradientes customizados. Implemente `forward()` (computa output, salva no `ctx`) e `backward()` (recebe `grad_output`, retorna gradientes para cada input). Uso quando a operação não é composta de ops PyTorch nativos (como aqui, onde usamos Higra em Python/C++).

---

### `MultiRegularization` — `src/losses/multi_regularization.py`

Composição alternativa, mais antiga. Não usa o padrão `LossTerm`/`LossComposer`. Recebe os pesos diretamente no construtor e computa tudo no `forward()`. Funciona do mesmo jeito — retorna `(total_loss, loss_log)`.

---

## 5. Pipeline — `src/pipeline/`

### `ModelPipeline`

```python
pipeline = ModelPipeline(steps=[...])
result = pipeline.forward(data_dict, verbose=True)
```

Recebe uma lista de `PipelineStep`s. Em `forward()`, passa o `data` dict por cada step em sequência — cada step recebe o dict e o retorna modificado.

`forward()` valida que `data` é um `dict` (levanta `TypeError` caso contrário). Se um step falhar, loga o nome do step que falhou e re-levanta a exceção.

### `PipelineStep` (abstrata)

Cada step implementa `forward(data: dict) -> dict`. O `data` é o estado compartilhado — qualquer key adicionada por um step fica disponível para os seguintes.

### Steps concretos

**`CellposeStep`**
- Requer GPU (levanta `RuntimeError` se não detectar GPU)
- Construtor: `CellposeStep(batch_size=10, name="CellposeStep", **eval_kwargs)` — qualquer `**eval_kwargs` é repassado para `model.eval()` em cada chamada (ex.: `diameter=30`, `channels=[0,0]`, `flow_threshold=0.4`)
- Entrada: `data["image"]` — `(H,W)` ou `(H,W,C)`
- Converte para uint8 RGB via `to_uint8_rgb()`
- Chama `CellposeModel.eval()` com batch_size configurável
- Saída adicionada ao dict:
  - `data["segmentation"]`: `(H,W)` labels inteiros (0 = background, 1..N = instâncias)
  - `data["flows"]`: outputs internos do Cellpose
  - `data["styles"]`: vetores de estilo

**`RGBAStep`**
- Entrada: `data["image"]` + `data["segmentation"]`
- Cria imagem RGBA: RGB da imagem original + canal alpha = 255 onde segmentação > 0
- Saída: `data["rgba"]` — `(H,W,4)` uint8

**`TrainingStep`**
- Entrada esperada no dict:
  - `data["markers"]`: `(N,C,H,W)` — predição da rede
  - `data["distance_maps"]`: `(N,C,H,W)`
  - `data["gt_masks"]`: `(N,C,H,W)`
- Chama o `LossComposer`
- Saída: `data["loss"]` (tensor escalar) + `data["loss_log"]` (dict)

**`MarkerStep`** e **`SegmentationStep`**: stubs — levantam `NotImplementedError`.

---

## 6. I/O — `src/io/output_writer.py`

```python
writer = OutputWriter(output_dir="results/")
writer.save_all(data)
```

Cria automaticamente subdirs `segmentations/`, `markers/`, `overlays/`.

`save_all(data)` espera keys: `data["id"]`, `data["segmentation"]`, `data["markers"]`, `data["image"]`, e opcionalmente `data["ground_truth"]`.

`save_overlay()` blende a imagem com a segmentação prevista (verde semitransparente) e opcionalmente o ground truth (vermelho).

---

## 7. Utilitários

**`to_uint8_rgb(image: np.ndarray) -> np.ndarray`** — `src/utils/image_utils.py`
1. Se grayscale `(H,W)`: expande para `(H,W,3)` replicando o canal
2. Se não é uint8: normaliza para [0,255] e converte
3. Retorna `(H,W,3)` uint8

**Logger** — `src/utils/logger.py`
```python
from src.utils.logger import logger
logger.info("mensagem")
```
Loguru — formatação automática, sem configuração extra.

---

## 8. PyTorch — Conceitos Chave

### `nn.Module`

Classe base de todo modelo e loss no PyTorch. Ao herdar dela:
- Parâmetros declarados como `nn.Parameter` ou sub-módulos são automaticamente rastreados
- `self.parameters()` retorna todos os parâmetros (para o optimizer)
- `state_dict()` / `load_state_dict()` funcionam recursivamente
- `.train()` / `.eval()` propagam para sub-módulos

### Autograd

Tensores com `requires_grad=True` constroem um grafo computacional enquanto ops são executadas. Ao chamar `.backward()` num tensor escalar, o PyTorch percorre esse grafo de trás para frente calculando derivadas parciais (regra da cadeia). Os gradientes ficam em `tensor.grad`.

### Shapes que importam

| Contexto | Shape | Significado |
|----------|-------|-------------|
| Batch de imagens | `(N, C, H, W)` | N = batch, C = canais, H = altura, W = largura |
| MarkerNet entrada | `(N, 4, H, W)` | 4 canais (RGBA esperado) |
| MarkerNet saída | `(N, 1, H, W)` | 1 mapa de probabilidade |
| Máscara dataset | `(H, W)` | numpy, sem dim de batch/canal |
| Segmentação Cellpose | `(H, W)` | inteiros, 0=bg, 1..N=instâncias |
| RGBA final | `(H, W, 4)` | uint8, canal alpha = região segmentada |

### Operações por localização no código

| Operação | Onde aparece | Para que serve |
|----------|-------------|----------------|
| `torch.sigmoid()` | `MarkerNet.forward` | Logits → probabilidades [0,1] |
| `torch.no_grad()` | `MarkerNet.predict`, `validation_step` | Desliga autograd para inferência |
| `model.eval()` | `MarkerNet.predict`, `validation_step` | BatchNorm/Dropout em modo inferência |
| `model.train()` | `MarkerNet.train_step` | Reativa BatchNorm/Dropout |
| `loss.backward()` | `MarkerNet.train_step` | Calcula gradientes |
| `optimizer.step()` | `MarkerNet.train_step` | Atualiza pesos |
| `loss.item()` | `MarkerNet.train_step` | Tensor → float para logging |
| `F.unfold()` | `NotTooThinLoss._morpho` | Extrai patches para morfologia diferenciável |
| `torch.sort()` | `TopologyLoss` | Ordena dinâmicas para ranked selection |
| `autograd.Function` | `ComponentTreeFunction` | Gradiente customizado através do Higra |

---

## 9. Notebook de Experimento — `notebooks/experiment_template.ipynb`

Template completo para rodar um experimento com o MarkerNet. Estrutura:

**Seções do notebook:**

| Seção | O que faz |
|-------|-----------|
| 0. Ambiente | Clona o repo do GitHub (para Colab), instala deps |
| 1. Configuração | Define `MODEL_CONFIG`, `TRAIN_CONFIG`, `LOSS_TERMS`, paths de output |
| 2. Dataset e DataLoader | Instancia `MonusegDataset`, define `collate_fn`, cria `DataLoader` |
| 3. Modelo | Instancia `MarkerNet`, conta parâmetros |
| 4. Loss e Otimizador | Monta `LossComposer`, `Adam`, `ReduceLROnPlateau` |
| 5. Loop de Treino | `run_epoch()` completa para treino e validação |
| 6. Curvas de Aprendizado | Plota loss total e por termo |
| 7. Inferência e Visualização | Carrega best checkpoint, visualiza predições |
| 8. Métricas Finais | Dice e IoU no val set completo |
| 9. Salvar Outputs | Usa `OutputWriter` para gerar overlays |
| 10. Pipeline de Inferência | Demo do pipeline CellposeStep → RGBAStep → MarkerNet manual |
| 11. Resumo | Tabela de resultados para preencher |

### Por que `collate_fn` customizado?

O `DataLoader` padrão do PyTorch não consegue batchificar o dataset diretamente porque:
1. Imagens podem ter shapes diferentes entre amostras
2. O 4º canal de entrada do `MarkerNet` precisa ser construído a partir do `ground_truth`
3. O `distance_map` precisa ser calculado a partir do `ground_truth`

A `collate_fn` resolve tudo isso:

```python
def collate_fn(samples):
    # Para cada amostra:
    # 1. Redimensiona imagem e máscara para TARGET_SIZE (256, 256)
    # 2. Monta RGBA: [img_R, img_G, img_B, gt_mask] → (4, H, W)
    # 3. Computa distance_map = EDT(1 - binary_mask), normalizado [0,1]
    # Retorna:
    return {
        'image':        Tensor (N, 4, H, W),   # RGBA: RGB + canal gt_mask
        'distance_map': Tensor (N, 1, H, W),   # distância euclidiana normalizada
        'ground_truth': Tensor (N, 1, H, W),   # máscara binária
        'id':           List[str],
    }
```

**Por que 4 canais?** Enquanto o `CellposeStep` não está integrado ao loop de treino, o `ground_truth` serve de proxy para o 4º canal — a rede aprende a usar a máscara de referência como dica de onde estão as células.

### Padrão de experimento

```python
# 1. Configuração central — tudo em um bloco
MODEL_CONFIG = {"encoder_name": "resnet34", "pretrained": True, "in_channels": 4, "threshold": 0.5}
LOSS_TERMS = [DiceTerm(), SizeTerm(weight=0.1), TVTerm(weight=0.05)]

# 2. Composição limpa — sem tocar no fonte
loss_fn = LossComposer(terms=LOSS_TERMS).to(DEVICE)
model   = MarkerNet(config=MODEL_CONFIG).to(DEVICE)

# 3. Loop de treino simples
markers = model.forward(images)
loss, log = loss_fn(markers, dmaps, gt)
loss.backward(); optimizer.step()

# 4. Checkpoint do melhor modelo
model.save(str(best_ckpt_path))   # salva state_dict + config
model.load(str(best_ckpt_path))   # carrega de volta
```

---

## 10. O que está incompleto

- `src/data/transforms/transforms.py` — vazio
- `src/models/networks/segmentation_net.py` — vazio
- `MarkerStep.forward()` — `NotImplementedError`
- `SegmentationStep.forward()` — `NotImplementedError`

A ligação entre MarkerNet (que treina os marcadores) e o pipeline de inferência (que usa esses marcadores para segmentação refinada) ainda não está implementada. O notebook `experiment_template.ipynb` seção 10 mostra como executar esse fluxo manualmente como referência para quando `MarkerStep` for implementado.

---

## 11. Dependências externas relevantes

| Lib | Versão | Papel |
|-----|--------|-------|
| `torch` | ≥2.1 | Framework principal |
| `torchvision` | ≥0.16 | Transforms de imagem |
| `segmentation-models-pytorch` | 0.5.0 | Arquitetura U-Net pronta |
| `cellpose` | 4.1.1 | Segmentação inicial (inference) |
| `higra` | não declarado | Component trees para TopologyLoss |
| `loguru` | 0.7.3 | Logger |
| `opencv` (cv2) | implícita | `fillPoly` nas máscaras XML |
