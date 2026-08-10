# Documentação Oficial da Arquitetura

**Projeto:** Pipeline de Geração de Marcadores Nebulosos (Fuzzy Markers) para Segmentação de Imagens Histológicas
**Tipo de documento:** Referência oficial de arquitetura
**Público-alvo:** Todos os desenvolvedores do projeto

> Este documento descreve a arquitetura **oficial e aprovada** do projeto. Ele não é uma proposta, uma revisão ou uma discussão de alternativas — é a referência que todo código novo deve seguir. Qualquer desenvolvedor deve conseguir implementar novas funcionalidades lendo apenas este documento.

---

## Sumário

1. [Introdução](#1-introdução)
2. [Visão Geral da Arquitetura](#2-visão-geral-da-arquitetura)
3. [Organização do Projeto](#3-organização-do-projeto)
4. [Documentação de Cada Arquivo](#4-documentação-de-cada-arquivo)
5. [Fluxo Completo de Execução](#5-fluxo-completo-de-execução)
6. [Contratos Entre as Camadas](#6-contratos-entre-as-camadas)
7. [Regras Arquiteturais](#7-regras-arquiteturais)
8. [Exemplos de Uso](#8-exemplos-de-uso)
9. [Princípios Arquiteturais](#9-princípios-arquiteturais)

---

## 1. Introdução

O objetivo desta arquitetura é organizar, de forma modular e sustentável, um pipeline de Deep Learning responsável por gerar automaticamente marcadores nebulosos (*fuzzy markers*) a partir de imagens histológicas, e por utilizar esses marcadores em uma rede final de segmentação treinável.

O sistema é composto por três componentes de modelo, com naturezas distintas:

- **Cellpose** — uma etapa de **pré-processamento**. Não é treinada neste projeto e não faz parte do grafo computacional treinável.
- **MarkerNet** — a **primeira rede treinável** do pipeline. Recebe a imagem (e a segmentação inicial produzida pelo Cellpose) e gera marcadores nebulosos.
- **Rede Final de Segmentação** (FMBS, Scribble Prompting ou outra) — recebe os marcadores da MarkerNet e produz a segmentação final utilizada no cálculo da loss.

A arquitetura foi dividida em camadas para que cada uma dessas naturezas distintas — pré-processamento, inferência treinável, orquestração de treinamento, cálculo de perdas e persistência de resultados — tivesse um lugar próprio no código, com fronteiras de dependência explícitas. Essa divisão é guiada pelos seguintes princípios:

- **Single Responsibility Principle (SRP):** cada classe e cada módulo têm um único motivo para mudar.
- **Separation of Concerns (SoC):** preparação de dados, inferência, treinamento e persistência são preocupações independentes.
- **Baixo Acoplamento:** camadas dependem de abstrações (interfaces/contratos), não de implementações concretas de outras camadas.
- **Alta Coesão:** cada pasta agrupa apenas componentes que mudam pelos mesmos motivos.
- **Pipeline Pattern:** transformações de dados são compostas como uma sequência de Steps independentes.
- **Strategy Pattern:** pontos de variação reais do projeto (como a rede final de segmentação) são modelados como implementações intercambiáveis de uma interface comum.

Essas escolhas equilibram modularidade e simplicidade: a arquitetura é extensível exatamente nos pontos onde o projeto precisa variar (rede final de segmentação, novos Steps de pré-processamento ou de inferência), e permanece simples em todo o restante, evitando abstrações desnecessárias.

---

## 2. Visão Geral da Arquitetura

O fluxo de alto nível do sistema é o seguinte:

```text
Dataset
    │
    ▼
Preprocessing Pipeline
    │
    ▼
Training Pipeline
    │
    ▼
Loss
    │
    ▼
Trainer
```

E, em termos dos componentes de modelo envolvidos, o fluxo completo do dado é:

```text
Imagem
    │
    ▼
Cellpose            (pré-processamento — não treinável)
    │
    ▼
MarkerNet            (1ª rede treinável)
    │
    ▼
Rede Final de Segmentação   (FMBS, Scribble Prompting ou outra)
    │
    ▼
Loss
```

### Papel de cada camada

| Camada | Papel |
|---|---|
| **Dataset** | Fornece imagens e ground truth brutos, ou já enriquecidos com resultados de pré-processamento persistidos. Nunca executa modelos diretamente. |
| **Preprocessing Pipeline** | Executa, uma única vez por imagem, as transformações que não fazem parte do grafo treinável (ex.: Cellpose, conversão RGBA, mapa de distância) e persiste os resultados. |
| **Training Pipeline** | Executa o *forward* das redes treináveis (MarkerNet e rede final) durante o treinamento. Não treina nada — apenas encadeia inferências diferenciáveis. |
| **Loss** | Calcula o valor de perda a partir da segmentação produzida e do ground truth, combinando múltiplas funções de perda via `LossComposer`. |
| **Trainer** | Orquestra o ciclo de treinamento: forward (via Training Pipeline), cálculo da loss, backward, otimização, scheduler e callbacks. |

Cada camada só conhece a camada imediatamente abaixo dela no fluxo, nunca a implementação interna das camadas seguintes. Essa é a base do baixo acoplamento em todo o sistema.

---

## 3. Organização do Projeto

### Árvore de diretórios

```text
src/
│
├── data/
│   └── load/
│       ├── base_dataset.py
│       ├── monuseg_dataset.py
│       └── monuseg_preprocessed_dataset.py
│
├── io/
│   └── output_writer.py
│
├── losses/
│   ├── border_loss.py
│   ├── distance_map_loss.py
│   ├── object_size_loss.py
│   ├── total_variation_loss.py
│   ├── rmse_loss.py
│   ├── soft_dice_loss.py
│   └── loss_composer.py
│
├── models/
│   ├── configs/
│   └── networks/
│       ├── base_network.py
│       ├── marker_unet.py
│       └── final_segmentation/
│           ├── base_final_segmentation.py
│           ├── fmbs_network.py
│           └── scribble_prompting_network.py
│
├── pipeline/
│   ├── preprocessing_pipeline.py
│   ├── training_pipeline.py
│   └── steps/
│       ├── base_step.py
│       ├── preprocessing/
│       │   ├── cellpose_step.py
│       │   ├── distance_map_step.py
│       │   └── rgba_step.py
│       ├── inference/
│       │   ├── marker_step.py
│       │   └── frozen_segmentation_step.py
│       └── persistence/
│           └── save_results_step.py
│
├── training/
│   ├── trainer.py
│   ├── training_loop.py
│   └── callbacks/
│
├── registry/
│
├── utils/
│
└── tests/
```

### Detalhamento por pasta

#### `data/`

- **Responsabilidade:** fornecer amostras de dados (imagem, ground truth e, quando aplicável, dados já pré-processados) para o restante do sistema, na forma de objetos `Dataset`.
- **Dependências permitidas:** `utils/` (para funções auxiliares de I/O, ex. leitura de imagem), `pipeline/preprocessing_pipeline.py` (apenas para compor pré-processamento no momento da leitura — nunca modelos individuais).
- **Dependências proibidas:** `models/` (nenhuma classe de `data/` pode importar uma rede neural diretamente), `training/` (Dataset não sabe que está sendo treinado).
- **Exemplos do que deve existir aqui:** classes `Dataset` (no sentido de framework de Deep Learning), lógica de leitura de arquivos, split de treino/validação, augmentations que não dependam de modelos.

#### `io/`

- **Responsabilidade:** persistência de resultados em disco (imagens processadas, máscaras, marcadores, logs de saída).
- **Dependências permitidas:** `utils/`.
- **Dependências proibidas:** `models/`, `training/`, `pipeline/` (o `io/` não decide *quando* salvar, apenas *como* salvar; a decisão de quando persistir pertence a um Step de `pipeline/steps/persistence/`).
- **Exemplos do que deve existir aqui:** escritores de arquivos (`output_writer.py`), serializadores de formatos de imagem/máscara.

#### `losses/`

- **Responsabilidade:** implementar funções de perda individuais e sua composição.
- **Dependências permitidas:** bibliotecas de tensor (ex. PyTorch), `utils/`.
- **Dependências proibidas:** `models/` (losses nunca instanciam ou executam redes), `data/`, `pipeline/`, `training/`.
- **Exemplos do que deve existir aqui:** uma classe por função de perda (`border_loss.py`, `soft_dice_loss.py` etc.) e o `loss_composer.py`, que combina múltiplas losses em um único valor escalar.

#### `models/`

- **Responsabilidade:** definir arquiteturas de redes neurais treináveis — a estrutura dos módulos, camadas e forward pass de cada rede.
- **Dependências permitidas:** bibliotecas de tensor, `models/configs/` (hiperparâmetros de arquitetura).
- **Dependências proibidas:** `data/` (uma rede nunca conhece de onde vêm os dados), `pipeline/` (uma rede não sabe que está inserida em um Step), `training/` (uma rede não conhece o otimizador ou o loop de treinamento que a treina).
- **Exemplos do que deve existir aqui:** `base_network.py`, `marker_unet.py`, e o subpacote `final_segmentation/`, que contém a interface e as implementações concretas de redes finais de segmentação.

#### `pipeline/`

- **Responsabilidade:** compor sequências de transformações (Steps) sobre um dicionário de dados compartilhado. Existem dois pipelines distintos: um de pré-processamento (execução única, não treinável) e um de treinamento (execução repetida, treinável).
- **Dependências permitidas:** `models/` (Steps de inferência instanciam e executam redes), `io/` (Steps de persistência gravam resultados), `utils/`.
- **Dependências proibidas:** `training/` (o pipeline nunca conhece o Trainer — a relação de dependência é sempre `training/` → `pipeline/`, nunca o contrário).
- **Exemplos do que deve existir aqui:** `preprocessing_pipeline.py`, `training_pipeline.py`, e o subpacote `steps/`, organizado em três categorias:
  - `steps/preprocessing/` — Steps não treináveis, executados uma vez (`cellpose_step.py`, `distance_map_step.py`, `rgba_step.py`).
  - `steps/inference/` — Steps que executam forward de redes treináveis (`marker_step.py`, `frozen_segmentation_step.py`).
  - `steps/persistence/` — Steps que gravam resultados em disco (`save_results_step.py`).

#### `training/`

- **Responsabilidade:** orquestrar o processo de treinamento — épocas, batches, forward via `TrainingPipeline`, cálculo de loss, backward, otimização, scheduler, checkpoints e callbacks.
- **Dependências permitidas:** `pipeline/` (consome o `TrainingPipeline` como dependência), `losses/` (consome o `LossComposer`), `data/` (consome Datasets/DataLoaders).
- **Dependências proibidas:** `models/` diretamente (o Trainer nunca instancia ou implementa uma rede; ele recebe o `TrainingPipeline` já configurado com as redes que precisa executar).
- **Exemplos do que deve existir aqui:** `trainer.py`, `training_loop.py`, e o subpacote `callbacks/` (early stopping, logging, checkpointing).

#### `registry/`

- **Responsabilidade:** mapear nomes/identificadores de configuração para classes concretas (ex.: qual classe de rede final instanciar a partir de uma string em um arquivo de configuração).
- **Dependências permitidas:** `models/`, `pipeline/steps/` (para registrar Steps disponíveis).
- **Dependências proibidas:** `training/` não deve depender diretamente do `registry/` para tomar decisões de treinamento — o registry serve para resolver **qual componente instanciar**, não **como treinar**.
- **Exemplos do que deve existir aqui:** funções ou classes de *factory* que resolvem nomes de configuração em implementações concretas.

#### `utils/`

- **Responsabilidade:** funções auxiliares, genéricas e sem regra de negócio, reutilizáveis por qualquer camada (manipulação de tensores, conversões de formato, logging genérico).
- **Dependências permitidas:** bibliotecas de terceiros de propósito geral.
- **Dependências proibidas:** qualquer dependência de `models/`, `data/`, `pipeline/` ou `training/` — `utils/` deve poder ser importado por qualquer camada sem criar dependência circular, e por isso não pode depender de nenhuma delas.
- **Exemplos do que deve existir aqui:** conversão de tensores para numpy, normalização de imagens, helpers de logging.

#### `tests/`

- **Responsabilidade:** testes automatizados de todas as camadas acima, organizados espelhando a estrutura de `src/`.
- **Dependências permitidas:** todas as camadas (testes podem importar qualquer módulo do projeto).
- **Dependências proibidas:** nenhuma — mas o inverso não é permitido: nenhum código de `src/` pode importar de `tests/`.

---

## 4. Documentação de Cada Arquivo

### `base_dataset.py`

**Objetivo:** definir a interface comum que todo Dataset do projeto deve implementar.

**O que deve fazer:** declarar o contrato mínimo (ex. `__len__`, `__getitem__`) que qualquer Dataset concreto deve respeitar, garantindo que Datasets sejam intercambiáveis do ponto de vista de quem os consome (DataLoader, Trainer).

**O que NÃO deve fazer:** não deve conter nenhuma lógica de leitura de arquivo específica de um dataset (isso pertence às subclasses concretas), nem lógica de pré-processamento.

**Como deve ser utilizado:** toda nova classe de Dataset (`MonusegDataset`, `MonusegPreprocessedDataset`, ou datasets de outras bases futuras) deve herdar desta classe.

```python
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict: ...
```

---

### `monuseg_dataset.py`

**Objetivo:** carregar os dados brutos da base MoNuSeg.

**O que deve fazer:** ler imagens e máscaras de ground truth do disco e retorná-las em um dicionário simples, ex. `{"image": ..., "ground_truth": ...}`.

**O que NÃO deve fazer:** nunca executa modelos (Cellpose, MarkerNet ou qualquer outra rede); nunca realiza pré-processamento além de operações estritamente de leitura/decodificação de arquivo (ex. não aplica Cellpose, não gera RGBA). Essas responsabilidades pertencem ao `PreprocessingPipeline`.

**Como deve ser utilizado:**

```python
dataset = MonusegDataset(root_dir="data/monuseg")
sample = dataset[0]
# sample == {"image": ..., "ground_truth": ...}
```

---

### `monuseg_preprocessed_dataset.py`

**Objetivo:** combinar um Dataset já existente com um `PreprocessingPipeline`, entregando amostras já enriquecidas com os resultados do pré-processamento.

**O que deve fazer:** receber, no construtor, uma instância de `BaseDataset` e uma instância de `PreprocessingPipeline`; para cada amostra, obter o dicionário base do Dataset interno e passá-lo pelo pipeline antes de retorná-lo.

**O que NÃO deve fazer:** esta classe **nunca** conhece o Cellpose, o RGBA ou qualquer outro Step específico — ela conhece apenas o conceito genérico de `PreprocessingPipeline`. Trocar o conteúdo do pipeline (por exemplo, adicionar um novo Step) nunca exige alterar esta classe.

**Como deve ser utilizado:**

```python
dataset = MonusegDataset(root_dir="data/monuseg")

pipeline = PreprocessingPipeline(
    steps=[
        CellposeStep(),
        RGBAStep(),
        DistanceMapStep(),
    ]
)

dataset = MonusegPreprocessedDataset(
    dataset,
    pipeline
)

sample = dataset[0]
# sample == {"image": ..., "ground_truth": ..., "segmentation": ..., "rgba": ..., "distance_map": ...}
```

---

### `preprocessing_pipeline.py`

**Objetivo:** encadear e executar uma sequência de Steps de pré-processamento sobre o dicionário de dados.

**O que deve fazer:** receber uma lista de Steps de pré-processamento no construtor e executá-los em ordem, passando o dicionário de um Step para o próximo, sem conhecer o conteúdo específico de cada Step.

**O que NÃO deve fazer:** não executa Steps de inferência de redes treináveis (isso é responsabilidade do `TrainingPipeline`); não realiza backpropagation; não decide onde os resultados serão persistidos (isso é responsabilidade de um `SaveResultsStep` dentro da própria lista de Steps, se necessário).

**Como deve ser utilizado:**

```python
pipeline = PreprocessingPipeline(
    steps=[
        CellposeStep(),
        RGBAStep(),
        DistanceMapStep(),
    ]
)

data = pipeline.run({"image": image, "ground_truth": ground_truth})
```

> **Observação importante:** novos Steps de pré-processamento podem ser adicionados à lista sem qualquer modificação na classe `PreprocessingPipeline`. Essa é a extensibilidade real oferecida pelo Pipeline Pattern neste projeto.

---

### `training_pipeline.py`

**Objetivo:** encadear e executar o *forward* das redes treináveis (MarkerNet e rede final de segmentação) durante o treinamento.

**O que deve fazer:** receber os Steps de inferência (`MarkerStep`, `FrozenSegmentationStep`) e executá-los em sequência, mantendo o grafo computacional diferenciável intacto para permitir backpropagation posterior pelo `Trainer`.

**O que NÃO deve fazer:** **não realiza treinamento.** Não calcula loss, não executa `backward()`, não conhece o otimizador. Sua única responsabilidade é produzir a segmentação final a partir da imagem (e, quando aplicável, da segmentação inicial do Cellpose já persistida).

**Como deve ser utilizado:**

```python
training_pipeline = TrainingPipeline(
    steps=[
        MarkerStep(marker_net),
        FrozenSegmentationStep(final_network),
    ]
)

data = training_pipeline.run(data)
# data agora contém data["segmentation"]
```

---

### `trainer.py`

**Objetivo:** controlar o ciclo de otimização de uma etapa de treinamento (um passo de treino).

**O que deve fazer:** orquestrar, para cada batch, o forward (delegando ao `TrainingPipeline`), o cálculo da loss (delegando ao `LossComposer`), o `backward()`, o passo do otimizador (`optimizer.step()`), o passo do scheduler e a execução de callbacks.

**O que NÃO deve fazer:** **nunca implementa redes neurais.** Não define arquiteturas, não conhece detalhes internos de `MarkerNet` ou da rede final — recebe o `TrainingPipeline` já configurado com essas redes e o trata como uma caixa-preta diferenciável.

**Como deve ser utilizado:**

```python
trainer = Trainer(
    training_pipeline=training_pipeline,
    loss_composer=loss_composer,
    optimizer=optimizer,
    scheduler=scheduler,
    callbacks=[EarlyStoppingCallback(), CheckpointCallback()],
)

trainer.train_step(batch)
```

---

### `training_loop.py`

**Objetivo:** organizar o laço de mais alto nível do treinamento — épocas, iteração sobre batches, validação periódica e checkpoints.

**O que deve fazer:** iterar sobre o `DataLoader` de treino por múltiplas épocas, invocar `Trainer.train_step()` para cada batch, executar validação nos intervalos configurados, e acionar checkpoints/callbacks ao final de cada época.

**O que NÃO deve fazer:** não implementa a lógica de um único passo de otimização (isso é responsabilidade do `Trainer`); não conhece detalhes de arquitetura de rede.

**Como deve ser utilizado:**

```python
training_loop = TrainingLoop(
    trainer=trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
)

training_loop.run()
```

---

### `base_network.py`

**Objetivo:** definir a interface comum a todas as redes treináveis do projeto.

**O que deve fazer:** declarar o contrato mínimo esperado de qualquer rede (ex. método `forward`), garantindo que redes sejam utilizáveis de forma intercambiável pelos Steps de inferência.

**O que NÃO deve fazer:** não deve conter lógica específica de uma arquitetura concreta (isso pertence às subclasses, como `marker_unet.py`).

**Como deve ser utilizado:** toda nova rede treinável do projeto (presente ou futura) deve herdar desta classe.

```python
class BaseNetwork(nn.Module, ABC):
    @abstractmethod
    def forward(self, x): ...
```

---

### `base_final_segmentation.py`

**Objetivo:** implementar o **Strategy Pattern** para o ponto de variação explícito do projeto — a rede final de segmentação (FMBS, Scribble Prompting ou outra).

**O que deve fazer:** declarar a interface comum que qualquer rede final deve implementar, permitindo que o `FrozenSegmentationStep` dependa apenas dessa interface, sem conhecer qual implementação concreta está sendo usada.

**O que NÃO deve fazer:** não deve conter lógica específica de nenhuma rede final concreta.

**Como deve ser utilizado:** para adicionar uma nova rede final de segmentação, basta criar uma nova classe no subpacote `final_segmentation/` que implemente esta interface — nenhuma outra camada do sistema precisa ser alterada.

```python
class BaseFinalSegmentation(BaseNetwork):
    @abstractmethod
    def forward(self, markers): ...

# Nova rede final, sem alterar nenhuma outra camada:
class MyNewFinalNetwork(BaseFinalSegmentation):
    def forward(self, markers):
        ...
```

---

### `marker_unet.py`

**Objetivo:** implementar a MarkerNet — a primeira rede treinável do pipeline.

**O que deve fazer:** receber a imagem (e, quando aplicável, a segmentação inicial produzida pelo Cellpose) e produzir marcadores nebulosos (*fuzzy markers*) como saída.

**O que NÃO deve fazer:** não executa o Cellpose internamente (essa segmentação inicial já chega pronta, via dado persistido pelo `PreprocessingPipeline`); não calcula loss; não decide como seus marcadores serão usados pela rede final — essa orquestração é papel do `TrainingPipeline`.

**Como deve ser utilizado:**

```python
marker_net = MarkerUNet(**config)
markers = marker_net(image, segmentation)
```

---

### `frozen_segmentation_step.py`

**Objetivo:** encapsular, dentro do `TrainingPipeline`, a execução da rede final de segmentação.

**O que deve fazer:** receber os marcadores produzidos pela MarkerNet (via dicionário de dados) e executar o forward da rede final (uma implementação de `BaseFinalSegmentation`), adicionando o resultado (`segmentation`) ao dicionário.

**O que NÃO deve fazer:** não implementa a arquitetura da rede final (apenas a instancia/recebe e a executa); não realiza backward nem otimização.

**Como deve ser utilizado:**

```python
step = FrozenSegmentationStep(final_network=fmbs_network)
data = step.run(data)  # data["segmentation"] é adicionado
```

---

### `save_results_step.py`

**Objetivo:** persistir resultados do pipeline de pré-processamento em disco.

**O que deve fazer:** receber o dicionário de dados já enriquecido pelos Steps anteriores (ex. Cellpose, RGBA) e delegar ao `io/output_writer.py` a gravação em disco.

**O que NÃO deve fazer:** **nunca participa do treinamento.** Pertence exclusivamente ao `PreprocessingPipeline` e nunca deve ser adicionado à lista de Steps do `TrainingPipeline`. Não implementa a lógica de serialização em si — apenas invoca `io/`.

**Como deve ser utilizado:**

```python
pipeline = PreprocessingPipeline(
    steps=[
        CellposeStep(),
        RGBAStep(),
        DistanceMapStep(),
        SaveResultsStep(output_dir="data/preprocessed"),
    ]
)
```

---

### `distance_map_step.py`

**Objetivo:** calcular, durante o pré-processamento, o mapa de distância consumido pelas losses no treinamento (ex.: `DistanceMapLoss`).

**O que deve fazer:** ler a máscara na chave `mask_key` (padrão `ground_truth`) do dicionário de dados e adicionar o mapa de distância na chave `output_key` (padrão `distance_map`). O mapa é calculado com `scipy.ndimage.distance_transform_edt` sobre o primeiro plano da máscara, normalizado pelo valor máximo e invertido (`1.0 - dt`), resultando em valores em `[0, 1]` — próximo de `0` no interior das células e próximo de `1` nas fronteiras entre objetos e no fundo.

**O que NÃO deve fazer:** não participa do treinamento — nunca é adicionado ao `TrainingPipeline`; é um cálculo NumPy puro, executado uma única vez por imagem dentro do `PreprocessingPipeline`, sem criar grafo computacional diferenciável.

**Como deve ser utilizado:**

```python
pipeline = PreprocessingPipeline(
    steps=[
        CellposeStep(),
        RGBAStep(),
        DistanceMapStep(),
        SaveResultsStep(output_dir="data/preprocessed"),
    ]
)
# data["distance_map"] é consumido pelo Trainer via distance_map_key="distance_map"
```

> **Observação:** o passo também pode ser aplicado por amostra após redimensionamentos (como nos notebooks de treino), desde que a máscara em `mask_key` esteja na mesma resolução do batch. O mapa deve ser calculado sobre a máscara **já redimensionada** — redimensionar o mapa calculado na resolução original não produz os mesmos valores.

---

### `loss_composer.py`

**Objetivo:** centralizar a composição de múltiplas funções de perda em um único valor escalar utilizado pelo `Trainer`.

**O que deve fazer:** receber uma lista (ou dicionário) de losses individuais, com seus respectivos pesos, e calcular a soma ponderada a partir da segmentação predita e do ground truth.

**O que NÃO deve fazer:** não executa inferência de nenhum modelo; não conhece a arquitetura das redes; não decide quando o backward é chamado (isso é papel do `Trainer`).

**Como deve ser utilizado:**

```python
loss_composer = LossComposer(
    losses=[
        (SoftDiceLoss(), 1.0),
        (BorderLoss(), 0.5),
        (TotalVariationLoss(), 0.1),
    ]
)

loss_value = loss_composer.compute(prediction=data["segmentation"], target=data["ground_truth"])
```

> Para adicionar uma nova loss ao treinamento, basta criar uma nova classe em `losses/` e incluí-la na lista passada ao `LossComposer` — nenhuma outra camada precisa ser alterada.

---

## 5. Fluxo Completo de Execução

O fluxo completo, desde a imagem bruta até o valor de loss, ocorre em duas fases distintas: **preparação de dados** (executada uma vez, fora do laço de treinamento) e **treinamento** (executada repetidamente, dentro do laço de épocas).

### Fase 1 — Preparação de dados (execução única)

```text
Imagem bruta (disco)
    │
    ▼
MonusegDataset                       → { image, ground_truth }
    │
    ▼
PreprocessingPipeline
    ├── CellposeStep                 → adiciona { segmentation }
    ├── RGBAStep                     → adiciona { rgba }
    └── DistanceMapStep              → adiciona { distance_map }
    │
    ▼
SaveResultsStep                      → persiste em disco (io/output_writer.py)
```

Este fluxo roda uma única vez por imagem (ou sempre que o pré-processamento precisar ser refeito), e seu resultado fica disponível em disco para todas as épocas de treinamento seguintes, evitando recomputação do Cellpose a cada batch.

### Fase 2 — Treinamento (execução repetida, por batch/época)

```text
MonusegPreprocessedDataset            → { image, ground_truth, segmentation, rgba, distance_map }
    │
    ▼
TrainingPipeline
    ├── MarkerStep (MarkerNet)        → adiciona { markers }
    └── FrozenSegmentationStep        → adiciona { segmentation }
    │
    ▼
LossComposer                          → calcula loss(segmentation, ground_truth)
    │
    ▼
Trainer
    ├── backward()
    ├── optimizer.step()
    └── scheduler.step()
    │
    ▼
TrainingLoop                          → repete para todos os batches e épocas
```

O `TrainingPipeline` é sempre executado com gradientes habilitados durante o treinamento, pois `MarkerNet` e a rede final são treináveis. O `PreprocessingPipeline`, por sua vez, é executado sem necessidade de gradientes, já que o Cellpose não é treinável e seus resultados são consumidos como entrada fixa.

---

## 6. Contratos Entre as Camadas

Toda a comunicação entre Datasets, Pipelines e Trainer ocorre através de um **dicionário de dados compartilhado**, que cresce progressivamente à medida que passa pelos Steps:

```python
{
    "image": ...,                     # adicionado por MonusegDataset
    "ground_truth": ...,              # adicionado por MonusegDataset
    "segmentation": ...,              # adicionado por CellposeStep
    "rgba": ...,                      # adicionado por RGBAStep
    "distance_map": ...,              # adicionado por DistanceMapStep
    "markers": ...,                   # adicionado por MarkerStep
    "segmentation": ...,              # adicionado por FrozenSegmentationStep (substitui a máscara inicial do Cellpose)
}
```

### Regras do contrato

1. Cada Step **adiciona** novas chaves ao dicionário — nunca remove chaves existentes.
2. Cada Step deve documentar explicitamente quais chaves espera encontrar no dicionário de entrada, e quais chaves adiciona na saída.
3. Nenhum Step deve depender de chaves que ainda não foram produzidas por um Step anterior na mesma sequência.
4. O `PreprocessingPipeline` e o `TrainingPipeline` operam sobre o **mesmo formato de dicionário**, mas cada um é responsável por um subconjunto diferente de chaves (pré-processamento vs. inferência treinável).
5. Chaves persistidas em disco pelo `SaveResultsStep` devem ter nomes idênticos às chaves usadas em memória, para que `MonusegPreprocessedDataset` possa reconstruir o dicionário completo ao carregar os dados persistidos.

---

## 7. Regras Arquiteturais

1. `Dataset` nunca executa modelos de Deep Learning.
2. `Dataset` nunca realiza pré-processamento diretamente — delega sempre ao `PreprocessingPipeline`.
3. `models/` nunca conhece `data/` — uma rede não sabe de onde vêm seus dados de entrada.
4. `models/` nunca conhece `pipeline/` — uma rede não sabe que está sendo executada dentro de um Step.
5. `Trainer` nunca implementa redes neurais — ele apenas orquestra o `TrainingPipeline` já configurado.
6. `Trainer` nunca conhece detalhes de arquitetura de `MarkerNet` ou da rede final.
7. `Losses` nunca executam inferência de modelos — recebem apenas tensores já calculados (predição e ground truth).
8. `Steps` de `pipeline/steps/preprocessing/` nunca realizam backpropagation.
9. `Steps` de `pipeline/steps/inference/` nunca calculam loss nem chamam `optimizer.step()`.
10. `SaveResultsStep` nunca participa do `TrainingPipeline` — pertence exclusivamente ao `PreprocessingPipeline`.
11. `utils/` nunca contém regras de negócio específicas do domínio (segmentação, marcadores) — apenas funções genéricas e reutilizáveis.
12. `io/` apenas persiste e carrega dados — nunca decide *quando* ou *o quê* persistir; essa decisão pertence a um Step.
13. `PreprocessingPipeline` nunca executa Steps de inferência de redes treináveis.
14. `TrainingPipeline` nunca executa Steps de pré-processamento não treinável (ex. Cellpose).
15. `registry/` resolve nomes de configuração em classes concretas, mas nunca contém lógica de treinamento ou de inferência.
16. Toda nova rede final de segmentação deve implementar `base_final_segmentation.py` — nenhuma outra camada deve ser alterada para suportá-la.
17. Toda nova rede treinável deve herdar de `base_network.py`.
18. Toda nova loss deve ser adicionada como uma nova classe em `losses/` e registrada no `LossComposer` — nunca implementada dentro do `Trainer`.
19. Nenhuma camada pode importar de `tests/`.
20. `training/` nunca importa `models/` diretamente — a relação correta é `training/` → `pipeline/` → `models/`.
21. Novos Steps de pré-processamento ou de inferência devem ser adicionados às listas de Steps dos respectivos Pipelines, sem exigir modificação nas classes `PreprocessingPipeline` ou `TrainingPipeline`.
22. Nenhum Step remove chaves do dicionário de dados compartilhado — Steps apenas adicionam informação.
23. O `Trainer` é a única classe do sistema autorizada a chamar `backward()` e `optimizer.step()`.
24. O mapa de distância consumido pelas losses é produzido exclusivamente por um Step de pré-processamento (ex.: `DistanceMapStep`) — nunca dentro do `TrainingPipeline`, do `Trainer` ou nos notebooks de treinamento.

---

## 8. Exemplos de Uso

### Exemplo completo: da preparação dos dados ao treinamento

```python
# 1. Carregar os dados brutos da base MoNuSeg
dataset = MonusegDataset(root_dir="data/monuseg")

# 2. Definir o pipeline de pré-processamento (executado uma única vez)
preprocessing = PreprocessingPipeline(
    steps=[
        CellposeStep(),
        RGBAStep(),
        DistanceMapStep(),
        SaveResultsStep(output_dir="data/preprocessed"),
    ]
)

# 3. Combinar o Dataset bruto com o pipeline de pré-processamento
dataset = MonusegPreprocessedDataset(
    dataset,
    preprocessing,
)

# 4. Definir o pipeline de treinamento (executado a cada batch)
marker_net = MarkerUNet(**marker_config)
final_network = FMBSNetwork(**fmbs_config)

training_pipeline = TrainingPipeline(
    steps=[
        MarkerStep(marker_net),
        FrozenSegmentationStep(final_network),
    ]
)

# 5. Definir a composição de losses
loss_composer = LossComposer(
    losses=[
        (SoftDiceLoss(), 1.0),
        (BorderLoss(), 0.5),
    ]
)

# 6. Definir o Trainer e o TrainingLoop
trainer = Trainer(
    training_pipeline=training_pipeline,
    loss_composer=loss_composer,
    optimizer=torch.optim.Adam(marker_net.parameters()),
    scheduler=None,
    callbacks=[EarlyStoppingCallback(), CheckpointCallback()],
)

training_loop = TrainingLoop(
    trainer=trainer,
    train_loader=DataLoader(dataset, batch_size=8),
    val_loader=val_loader,
    num_epochs=100,
)

# 7. Executar o treinamento completo
training_loop.run()
```

**Explicação de cada etapa:**

1. **`MonusegDataset`** entrega apenas dados brutos — imagem e ground truth.
2. **`PreprocessingPipeline`** aplica Cellpose, conversão RGBA e o mapa de distância uma única vez, e `SaveResultsStep` persiste os resultados em disco.
3. **`MonusegPreprocessedDataset`** entrega, a partir daí, amostras já enriquecidas com os resultados persistidos, sem nunca reexecutar o Cellpose.
4. **`TrainingPipeline`** encadeia apenas as redes treináveis (`MarkerNet` e a rede final), mantendo o grafo diferenciável.
5. **`LossComposer`** combina as funções de perda relevantes para o problema.
6. **`Trainer`** orquestra forward, cálculo de loss, backward e otimização para um único batch.
7. **`TrainingLoop`** repete esse processo por todas as épocas, intercalando validação e checkpoints.

### Exemplo: adicionando uma nova rede final de segmentação

```python
# models/networks/final_segmentation/my_new_network.py
class MyNewFinalNetwork(BaseFinalSegmentation):
    def forward(self, markers):
        ...

# Uso — nenhuma outra camada do sistema precisa ser alterada:
final_network = MyNewFinalNetwork(**config)
training_pipeline = TrainingPipeline(
    steps=[
        MarkerStep(marker_net),
        FrozenSegmentationStep(final_network),
    ]
)
```

---

## 9. Princípios Arquiteturais

### Single Responsibility Principle (SRP)

Cada classe tem exatamente um motivo para mudar: `MonusegDataset` muda apenas se a forma de ler a base MoNuSeg mudar; `Trainer` muda apenas se a lógica de otimização mudar; `LossComposer` muda apenas se a forma de combinar losses mudar. Nenhuma classe mistura responsabilidades de camadas diferentes.

### Separation of Concerns (SoC)

O sistema separa explicitamente quatro preocupações: preparação de dados (`pipeline/preprocessing_pipeline.py`), inferência treinável (`pipeline/training_pipeline.py`), orquestração de treinamento (`training/`) e persistência (`io/`). Cada uma vive em seu próprio módulo, com dependências unidirecionais entre elas.

### Pipeline Pattern

Tanto o pré-processamento quanto a inferência durante o treinamento são modelados como sequências de Steps que operam sobre um dicionário de dados compartilhado. Isso permite adicionar, remover ou reordenar Steps sem modificar a classe `Pipeline` em si — apenas a lista de Steps fornecida.

### Strategy Pattern

A rede final de segmentação (`base_final_segmentation.py`) é o ponto de variação explícito do projeto. Qualquer nova implementação (FMBS, Scribble Prompting, ou uma rede futura) pode ser adicionada implementando a interface comum, sem exigir alterações no `TrainingPipeline`, no `Trainer` ou em qualquer outra camada.

### Composição em vez de Herança

O `Trainer` não herda de `TrainingPipeline`, nem `MonusegPreprocessedDataset` herda de `PreprocessingPipeline`: ambos **compõem** essas dependências, recebendo-as prontas via construtor. Isso permite trocar o comportamento de cada componente (por exemplo, usar um `PreprocessingPipeline` diferente) sem criar uma nova subclasse.

### Baixo Acoplamento

As dependências entre camadas seguem sempre uma única direção: `training/` depende de `pipeline/`, que depende de `models/`; nunca o inverso. `models/` não conhece `data/`, `pipeline/` nem `training/`. Essa direção única de dependência é o que permite substituir qualquer rede, Step ou pipeline sem efeitos colaterais em cascata.

### Alta Coesão

Cada pasta agrupa componentes que mudam pelos mesmos motivos: todos os arquivos em `losses/` mudam por razões relacionadas a funções de perda; todos os arquivos em `pipeline/steps/inference/` mudam por razões relacionadas à execução de redes treináveis dentro do fluxo de treinamento. Essa organização torna previsível onde encontrar (e onde adicionar) cada tipo de mudança.