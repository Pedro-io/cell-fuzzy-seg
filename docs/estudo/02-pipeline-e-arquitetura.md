# 02 — Pipeline e arquitetura

> Legenda: ✅ verificado no código ou na biblioteca · 📜 histórico/commits/docs antigos · 📖 artigo · ❓ a confirmar.
> Problemas ficam em [08-pendencias.md](08-pendencias.md) e são citados pelo ID.
>
> **Este documento substitui o antigo `docs/ARCHITECTURE.md`** (removido em 2026-09-27; recuperável com
> `git show f9b1e9b:docs/ARCHITECTURE.md`). A numeração das regras foi mantida, porque o código cita "regra 5", "regra 10"
> e "regra arquitetural 23".
>
> Última verificação: 2026-09-27 (commit `f9b1e9b`). A API do Cellpose foi conferida no código-fonte da versão fixada
> (`cellpose==4.1.1`, arquivo `cellpose/models.py` da tag `v4.1.1`).

---

## 1. Resumo

- Todo o processamento é uma sequência de **Steps**. Cada Step recebe um dicionário (`data`), acrescenta chaves e o devolve.
  Um **Pipeline** só executa uma lista de Steps em ordem. ✅
- Há dois pipelines com papéis diferentes:
  - o **de pré-processamento** (Cellpose → RGBA → mapa de distância → salvar), que roda **uma vez**, sem gradiente;
  - o **de treino** (MarkerUNet → ScribblePrompt), que roda **a cada batch**, com gradiente. ✅
- As dependências entre camadas estão limpas: nenhum import viola a direção `training → pipeline → models`. ✅ (§3)
- Há código morto, três classes de pipeline praticamente idênticas e uma chave do dicionário que é sobrescrita (PD-31,
  PD-32, PD-33).
- Três achados no `CellposeStep`: a proteção contra nome de modelo errado nunca executa (PD-29); o `diam_mean` é ignorado
  pelo Cellpose 4; e `diameter=30` significa "não redimensionar" (PD-30).

---

## 2. Por que essa arquitetura 📜

A motivação aparece em dois documentos antigos (hoje só no git):

- **Pipeline** (`docs/pipeline_pattern.md`, jul./2026): o pré-processamento e o forward das redes eram código espalhado nos
  notebooks. Transformar cada etapa num Step independente permite trocar, reordenar ou testar etapas isoladamente, e
  reaproveitar as mesmas etapas em notebooks diferentes.
- **Strategy** (`jornal.md`, 04/05/2026): a escolha das losses "estava presa no código-fonte"; para testar outra combinação
  era preciso editar o arquivo. Com `LossTerm` + `LossComposer`, o experimento **monta a lista de losses de fora**. O
  mesmo raciocínio vale para a rede final (`BaseFinalSegmentation`), que já foi trocada três vezes: watershed, U-Net congelada
  e ScribblePrompt.

O `ARCHITECTURE.md` descrevia isso como "arquitetura oficial" com princípios SOLID (responsabilidade única, baixo
acoplamento etc.). A §3 mostra o quanto o código segue de fato.

---

## 3. Camadas e dependências ✅

```
                 ┌──────────────┐
                 │  notebooks   │  montam tudo: datasets, steps, pipelines, losses, Trainer
                 └──────┬───────┘
        ┌───────────────┼────────────────┬───────────────┐
        ▼               ▼                ▼               ▼
  src/training/ ──► src/pipeline/ ──► src/models/    src/losses/     src/data/
  (Trainer,         (Pipelines,       (redes)         (termos,        (datasets)
   TrainingLoop,     Steps)                            composer)
   callbacks)            │
                         └──► src/io/ (OutputWriter)
                                   todos ──► src/utils/ (logger, conversões)
```

**Imports internos reais** (levantados com `grep` em `src/`):

| Pacote | Importa de | Direção esperada? |
|---|---|---|
| `training/` | só `pipeline.training_pipeline` e `utils` | ✅ nunca importa `models/` (regra 20) |
| `pipeline/steps/inference/` | `models.networks.final_segmentation.base_final_segmentation` (só a interface) e `utils` | ✅ |
| `pipeline/steps/persistence/` | `io.output_writer`, `utils` | ✅ |
| `pipeline/steps/preprocessing/` | `utils` | ✅ |
| `models/` | só `models/` e `utils` | ✅ não conhece `data/` nem `pipeline/` (regras 3 e 4) |
| `losses/` | só `losses/` | ✅ (regra 7) |
| `data/` | só `data/` e `utils` | ✅ |
| `io/`, `utils/` | nada interno | ✅ |

Os acoplamentos que existem são indiretos, por *duck typing* (sem import):
- o `Trainer` chama `step.set_training(...)` se o Step tiver esse método
  ([trainer.py:244-249](../../src/training/trainer.py#L244-L249));
- o `GradNormCallback` procura os atributos `model` e `final_network` dentro dos Steps
  ([grad_norm_callback.py:154-158](../../src/training/callbacks/grad_norm_callback.py#L154-L158));
- o `MarkerStep` chama `self.model.model`, ou seja, a `smp.Unet` **dentro** da `MarkerUNet`
  ([marker_step.py:71](../../src/pipeline/steps/inference/marker_step.py#L71),
  [:204](../../src/pipeline/steps/inference/marker_step.py#L204)).

Funciona, mas quem renomear esses atributos quebra o treino sem erro de import (PD-35).

---

## 4. O contrato: o dicionário `data` ✅

### 4.1 Chaves ao longo do fluxo

| Chave | Quem cria | Forma / tipo | Quem consome |
|---|---|---|---|
| `id` | `MonusegDataset` | str | `SaveResultsStep` (nome do arquivo) |
| `image` | `MonusegDataset` | (H,W,3) uint8 → no batch, (B,3,H,W) float [0,255] | `CellposeStep`, `RGBAStep`, `ScribblePromptingNetwork` |
| `ground_truth` | `MonusegDataset` | (H,W) uint8 0/1 → (B,1,H,W) | `DistanceMapStep`, losses |
| `ground_truth_instances` | `MonusegDataset` (desde 2026-09-29, só com XML) | (H,W) int32, um rótulo por núcleo | `DistanceMapStep` (desde 2026-10-06). Não é persistido (decisão do autor: sai dos XMLs em CPU). |
| `meta` | `MonusegDataset` | dict de caminhos | ninguém |
| `segmentation` ① | `CellposeStep` | (H,W) uint16, instâncias | `RGBAStep`, `SaveResultsStep` |
| `cellpose_prob` | `CellposeStep` (desde 2026-10-01) | (H,W) float16 [0,1]: sigmoide do logit `flows[2]` | `RGBAStep` com `alpha="prob"`, `SaveResultsStep` |
| ~~`flows`, `styles`~~ | `CellposeStep` (até 2026-10-01) | listas do Cellpose | ninguém; removidas (PD-31) |
| `rgba` | `RGBAStep` | (H,W,4) float32 [0,1]; alpha = máscara (`"mask"`, padrão) ou probabilidade (`"prob"`) | `MarkerStep` |
| `distance_map` | `DistanceMapStep` | (H,W) float32 [0,1]; 0 no centro de cada núcleo (desde 2026-10-06) | losses (`DMapTerm`) |
| `markers` | `MarkerStep` | (B,1,H,W) float [0,1], com grafo | `FrozenSegmentationStep`, losses |
| `segmentation` ② | `FrozenSegmentationStep` | (B,1,H,W) float [0,1], com grafo | `Trainer` (`prediction_key`), losses |

① e ② são **a mesma chave com significados diferentes**. Na fase 2, o `segmentation` do Cellpose nem é carregado (os
notebooks só leem `image`, `rgba`, `ground_truth` e `distance_map`). Assim, a ② não apaga nada em memória, mas o nome
engana e contraria o espírito da regra 22 (PD-33).

### 4.2 Regras do contrato (numeração original)

1. Cada Step **adiciona** chaves ao dicionário; nunca remove. ✅ Nenhum Step remove, mas um sobrescreve (PD-33).
2. Cada Step documenta as chaves que espera e as que adiciona. ✅ Todos os docstrings de `forward` fazem isso.
3. Nenhum Step depende de chave ainda não produzida. ✅ Cada Step verifica e lança `KeyError` com mensagem clara.
4. Os dois pipelines usam o mesmo formato de dicionário, cada um responsável por um subconjunto de chaves. ✅
5. Chaves persistidas pelo `SaveResultsStep` têm os mesmos nomes das chaves em memória. ✅
   ([save_results_step.py:47](../../src/pipeline/steps/persistence/save_results_step.py#L47)). ⚠️ O texto antigo dizia que
   isso permitia ao `MonusegPreprocessedDataset` "reconstruir o dicionário do disco", mas **ele não lê do disco**: quem
   reconstrói são os notebooks (PD-19).

---

## 5. Os Steps

### 5.1 A base: `PipelineStep` ✅ — [base_step.py](../../src/pipeline/steps/base_step.py)

```python
class PipelineStep(ABC):
    def __init__(self, name="PipelineStep"): self.name = name
    def __call__(self, data): return self.forward(data)   # step(data) == step.forward(data)
    @abstractmethod
    def forward(self, data) -> dict: ...
```

Não há `run()` no Step. O antigo `ARCHITECTURE.md` mostrava `step.run(data)`, o que estava errado.

### 5.2 `CellposeStep`: segmentação bruta — [cellpose_step.py](../../src/pipeline/steps/preprocessing/cellpose_step.py)

**O que é o Cellpose** 📖 (artigo de 2020, em `docs/papers/cellpose.pdf`): em vez de prever máscaras direto, a rede prevê
três mapas por pixel:
- o **fluxo horizontal** e o **fluxo vertical**: o gradiente de uma "difusão de calor" simulada a partir do centro de cada
  célula, ou seja, em que direção fica o centro;
- a **probabilidade de o pixel pertencer a alguma célula**.

Para recuperar as máscaras, cada pixel com probabilidade alta "segue o fluxo" por ~200 passos. Os pixels que convergem para
o mesmo ponto formam uma célula ("Mask recovery from vector flows"). Máscaras cujo fluxo previsto destoa muito do fluxo
recalculado a partir da própria forma são descartadas ("Mask quality threshold"). É isso que o `flow_threshold` controla.

**O que o projeto usa** ✅: `cellpose==4.1.1` com o modelo `cpsam` (**Cellpose-SAM**), o único modelo embutido nessa versão
(`MODEL_NAMES = ["cpsam"]`). A rede é um `Transformer` (encoder no estilo SAM), mas a saída continua a ser fluxos +
probabilidade. ❓ A descrição do Cellpose-SAM precisa do artigo próprio (PD-22).

**Parâmetros**, conferidos contra `cellpose/models.py` v4.1.1:

| Parâmetro no Step | Valor | Padrão da biblioteca | O que faz de verdade |
|---|---|---|---|
| `pretrained_model` | `"cpsam"` | `"cpsam"` | Um nome desconhecido faz o Cellpose **cair no `cpsam` com um aviso no log**. O `cpsam_v2` de julho, portanto, rodava `cpsam`. |
| `diam_mean` (construtor) | 30 → **não é mais passado** (2026-10-06) | — | **Ignorado** no Cellpose ≥ 4.0.1. A biblioteca loga "diam_mean argument are not used"; o aviso aparece na saída do notebook de pré-processamento (PD-30). |
| `diameter` (`eval`) | 30 → **`None`** (2026-10-06) | `None` | Reescala a imagem por `30/diameter`. **Com 30, o fator é 1: não reescala.** O núcleo mediano do MoNuSeg tem ~24 px (40×) e ~13 px nas imagens em 20× (PD-24, PD-30). |
| `flow_threshold` | 0,2 → **0,4** (2026-10-06) | 0,4 | Erro de fluxo máximo por máscara. O 0,2 (teste antigo do autor, sem números) descartava máscaras boas; no teste da PD-30, o padrão subiu o Dice no treino de 0,814 para 0,845. |
| `cellprob_threshold` | 0,0 | 0,0 | Limiar do logit de probabilidade; menor = mais e maiores máscaras. |
| `min_size` | 4 → **15** (2026-10-06) | 15 | Remove máscaras menores que `min_size` px. Mudou junto com o `flow_threshold` (PD-30). |
| `batch_size` | 8 | 8 | Nº de *tiles* de 256² processados juntos na GPU (só afeta velocidade). |

Outros comportamentos:
- **Exige GPU**: `core.use_gpu()` falso → `RuntimeError` ([L54-55](../../src/pipeline/steps/preprocessing/cellpose_step.py#L54-L55)).
- **Nome do modelo** (desde 2026-10-01, PD-29 ✅): o `_validate_model_name` aceita um arquivo existente ou um nome de
  `MODEL_NAMES + get_user_models()`, a mesma regra do Cellpose 4.1.1, e levanta `ValueError` antes de carregar o modelo. Antes,
  o `_warn_if_model_unavailable` importava `MODEL_LIST`, que não existe no 4.1.1, e a proteção (item P9 da investigação) nunca executava.
- `forward` grava `segmentation` e, desde 2026-10-01, `cellpose_prob`: a sigmoide do `flows[2]`, que é um logit (o Cellpose
  treina esse canal com `BCEWithLogitsLoss`, e o `cellprob_threshold=0` equivale a 0,5), em `float16`. O step confere que o mapa
  tem o formato da máscara. `flows` e `styles` deixaram de ser gravados (PD-31). Antes, a probabilidade era jogada fora (PD-34).
  Usá-la seria uma ideia a testar (PD-34).

### 5.3 `RGBAStep` — [rgba_step.py](../../src/pipeline/steps/preprocessing/rgba_step.py)

`rgba = concat( to_float32_rgb(image), (segmentation > 0) )` → (H,W,4) float32 em [0,1]
([L60-85](../../src/pipeline/steps/preprocessing/rgba_step.py#L60-L85)). O `to_float32_rgb`
([image_utils.py:29-49](../../src/utils/image_utils.py#L29-L49)) replica imagens cinza para 3 canais, corta em 3 canais e
divide por 255 se o máximo passar de 1. O alpha é **binário**: a identidade das instâncias do Cellpose se perde aqui.
**Desde 2026-10-01 (PD-34):** o parâmetro `alpha` escolhe a origem do 4º canal: `"mask"` (padrão, o comportamento acima) ou
`"prob"` (a `cellpose_prob`, em [0,1]). O step confere o formato e, com `"prob"`, a faixa.

**Por que RGBA?** 📖 É a forma de fusão precoce da tese (prova de conceito, 5.4.1: "RGB concatenado a uma máscara de cue,
entrada de 4 canais"). O "A" não é transparência de verdade; é só o 4º canal.

### 5.4 `DistanceMapStep` — [distance_map_step.py](../../src/pipeline/steps/preprocessing/distance_map_step.py)

**Desde 2026-10-06 (PD-06 ✅):** o mapa é **por núcleo**, a partir de `ground_truth_instances`
([compute_instance_distance_map](../../src/pipeline/steps/preprocessing/distance_map_step.py)):

```python
for cada núcleo (rótulo > 0):                       # recorte com 1 px de margem, limitado à imagem
    dt = distance_transform_edt(núcleo)             # distância ao pixel mais próximo fora do núcleo
    distance_map[núcleo] = 1 - dt / dt.max()        # máximo DAQUELE núcleo: o centro de todo núcleo vale 0
# fundo = 1; núcleos que se tocam têm a fronteira como borda dos dois; a borda da imagem não conta como fundo
```

📖 É a eq. 5.7 da tese, com a normalização por objeto. Sem a chave `ground_truth_instances`, o step levanta `KeyError`. A versão
anterior usava o máximo da **imagem** (só o maior núcleo tinha centro 0) e o GT binário (núcleos colados viravam um blob); ela
foi removida, por decisão do autor, em vez de virar opção (09 §16.5).

📜 O mapa era calculado dentro do notebook de treino e virou Step em 10/08 (regra 24). O `ARCHITECTURE.md` antigo avisava:
se a máscara for redimensionada, o mapa precisa ser **recalculado** na nova resolução, e não redimensionado. Hoje tudo roda
em 1000² e o aviso não se aplica. A aumentação (rotação de 90° e flips) mantém o mapa válido.

### 5.5 `SaveResultsStep` + `OutputWriter` — [save_results_step.py](../../src/pipeline/steps/persistence/save_results_step.py), [output_writer.py](../../src/io/output_writer.py)

- O Step exige `id` e delega a `OutputWriter.save_preprocessed(id, data, keys)`, que grava `<output_dir>/<chave>/<id>.npy`
  ([output_writer.py:29-57](../../src/io/output_writer.py#L29-L57)).
- Chaves padrão: `image`, `segmentation`, `cellpose_prob` (desde 2026-10-01), `rgba`, `ground_truth` e `distance_map`. A
  `cellpose_prob` é salva mesmo com `alpha="mask"`, para a outra variante poder ser montada sem rodar o Cellpose de novo.
- **Por que `.npy`** (docstring, 📜): `rgba` e `distance_map` são float32; salvar em PNG uint8 perderia precisão.
- Divisão de papéis (regra 12): o Step decide **o quê e quando** salvar; o `io/` só sabe **como**.
- O resto do `OutputWriter` (`save_all`, `save_segmentation`, `save_markers`, `save_overlay`, `save_rgba`, `_colorize`) não é
  usado por ninguém (PD-31).

### 5.6 Steps de inferência (detalhes no tema 3)

- **`MarkerStep`** ([marker_step.py](../../src/pipeline/steps/inference/marker_step.py)) tem dois modos.
  - `differentiable=True` (treino): `F.interpolate` para 256² → UNet → sigmoid → volta ao tamanho original, com o grafo
    preservado.
  - `differentiable=False` (inferência): `no_grad`, `cv2.resize` e binarização com limiar 0,5.
  - O Step também gerencia `train()`/`eval()` via `set_training` ([L90-93](../../src/pipeline/steps/inference/marker_step.py#L90-L93)).
  - Sem modelo, ele usa a máscara do Cellpose como marcador, só com um *warning* (PD-16).
- **`FrozenSegmentationStep`** ([frozen_segmentation_step.py](../../src/pipeline/steps/inference/frozen_segmentation_step.py))
  chama `final_network({"image", "scribbles": markers})` sem `no_grad` e grava `segmentation` ②.

---

## 6. Os pipelines ✅

| Classe | Arquivo | Método | Usada por |
|---|---|---|---|
| `PreprocessingPipeline` | [preprocessing_pipeline.py](../../src/pipeline/preprocessing_pipeline.py) | `run(data, verbose)`; `__call__` = `run` sem log | notebooks de pré-processamento |
| `TrainingPipeline` | [training_pipeline.py](../../src/pipeline/training_pipeline.py) | idem | `Trainer` e notebooks de experimento |
| `ModelPipeline` | [model_pipeline.py](../../src/pipeline/model_pipeline.py) | `forward(data, verbose)` | **ninguém** (resto da versão de maio) |

As três fazem **exatamente a mesma coisa**: percorrem os Steps, logam o nome, repassam exceções. A separação entre
pré-processamento e treino é **só de nome**: nada no código impede colocar um `CellposeStep` num `TrainingPipeline` (regras
13 e 14 dependem de disciplina). É uma escolha defensável, que deixa a intenção explícita e custa pouco, mas o `ModelPipeline`
pode sair (PD-31, PD-32).

**Modo treino/avaliação:** o `TrainingPipeline` **não** alterna os modos. Quem alterna é o `Trainer`, chamando `set_training`
nos Steps que têm esse método. O `ScribblePromptingNetwork` fica sempre em `eval()` (o `train()` dele é sobrescrito). Detalhes
nos temas 3 e 5.

---

## 7. Regras arquiteturais (numeração original) — situação em 2026-09-27

| # | Regra | Situação |
|---|---|---|
| 1 | `Dataset` nunca executa modelos | ⚠️ `MonusegPreprocessedDataset` roda o pipeline, que pode conter o Cellpose. Formalmente delega (regra 2), mas executa. |
| 2 | `Dataset` delega o pré-processamento ao `PreprocessingPipeline` | ✅ |
| 3 | `models/` não conhece `data/` | ✅ |
| 4 | `models/` não conhece `pipeline/` | ✅ |
| 5 | `Trainer` não implementa redes | ✅ |
| 6 | `Trainer` não conhece detalhes das redes | ✅ no `Trainer`; ⚠️ o `GradNormCallback` depende de atributos dos Steps (PD-35) |
| 7 | Losses não executam modelos | ✅ |
| 8 | Steps de pré-processamento não fazem backprop | ✅ |
| 9 | Steps de inferência não calculam loss nem chamam `optimizer.step()` | ✅ |
| 10 | `SaveResultsStep` só no `PreprocessingPipeline` | ✅ por uso (não é forçado) |
| 11 | `utils/` sem regra de negócio | ✅ |
| 12 | `io/` só persiste; quem decide é o Step | ✅ |
| 13 | `PreprocessingPipeline` não executa Steps de inferência | ✅ por uso (não é forçado) |
| 14 | `TrainingPipeline` não executa Steps de pré-processamento | ✅ por uso (não é forçado) |
| 15 | `registry/` resolve nomes em classes | ❌ **obsoleta**: a pasta não existe |
| 16 | Nova rede final implementa `BaseFinalSegmentation` | ✅ |
| 17 | Nova rede treinável herda de `BaseNetwork` (arquivo real: `base_networks.py`) | ✅ |
| 18 | Nova loss vira classe em `losses/`, usada via `LossComposer` | ✅ |
| 19 | Nada importa de `tests/` | ✅ |
| 20 | `training/` não importa `models/` | ✅ |
| 21 | Novos Steps entram na lista sem mudar as classes de pipeline | ✅ |
| 22 | Nenhum Step remove chaves | ⚠️ ninguém remove, mas `segmentation` é sobrescrita (PD-33) |
| 23 | Só o `Trainer` chama `backward()`/`optimizer.step()` | ✅ no `src/`; os notebooks têm uma célula de diagnóstico com `backward()` (sem `step()`) para conferir que o gradiente chega à MarkerUNet |
| 24 | O mapa de distância só é produzido pelo `DistanceMapStep` | ✅ (nenhum notebook calcula EDT por conta própria) |

---

## 8. Erros do antigo `ARCHITECTURE.md` (para não repeti-los)

- A árvore listava `registry/`, `fmbs_network.py` e `base_network.py`. Os dois primeiros não existem; o terceiro se chama `base_networks.py`.
- Chamava a rede final de "treinável". Ela é **congelada**.
- Exemplos com assinaturas que não existem: `MonusegDataset(root_dir=...)`, `MarkerUNet(**config)` seguido de
  `marker_net(image, segmentation)`, `step.run(data)`, `FMBSNetwork(...)`, `EarlyStoppingCallback`/`CheckpointCallback`
  (não implementados).
- Dizia que o `MonusegPreprocessedDataset` entrega os dados persistidos "sem nunca reexecutar o Cellpose". É o contrário.
- A interface `BaseFinalSegmentation.forward(markers)` hoje é `forward(data)`, com `data = {"image", "scribbles"}`.

---

## 9. Decisões confirmadas (2026-09-27)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | Origem de `flow_threshold=0.2` e `min_size=4` | **Testes do autor**: pelo que ele lembra, esses valores melhoraram a segmentação inicial e encontravam melhor os núcleos. O teste não ficou registrado. | PD-30: remedir junto com o diâmetro, registrando os números |
| 2 | Testar `diameter` ≈ 24 (≈ 13 nas imagens em 20×) | Sim | PD-30: decidido; a executar no Colab, depois regerar `MoNuSegPreprocessed/` |
| 3 | Código morto (`ModelPipeline`, métodos do `OutputWriter`…) | Marcar para remoção; se precisar para figuras, reimplementar depois | PD-31: decidido remover |
| 4 | Renomear a saída final para `final_segmentation` | Sim | PD-33: decidido renomear |

Na §5.2, os valores 0,2 e 4 deixam de ser "sem registro do motivo": foram escolhidos por teste do autor.

### Perguntas originais (histórico)

1. **Parâmetros do Cellpose:** `flow_threshold=0.2` e `min_size=4` são diferentes do padrão (0,4 e 15). Foi ajuste seu,
   veio de algum tutorial ou do orientador? (PD-30)foi por teste meu, pelo oque me lembro este valores melhoraram a segmentação inicial, ajudando a encontra melhor os nucleos 
2. **Diâmetro:** quer testar `diameter≈24` (a mediana medida) e, nas três imagens em 20×, ≈13? Isso mudaria a entrada da
   MarkerUNet, então exigiria regerar `MoNuSegPreprocessed/`. (PD-30, PD-24) podemos fazer isso 
3. **`ModelPipeline` e métodos não usados do `OutputWriter`:** posso marcar para remoção? Ou você pretende usar
   `save_overlay`/`save_markers` para gerar figuras do TCC? (PD-31) ainda não pensei sobre. podemos marcar como remoção e depois se precisar implementamos 
4. **Chave `segmentation`:** topa renomear a saída final para algo como `final_segmentation`? Exige mudar o `prediction_key`
   do `Trainer`, os testes e os notebooks. (PD-33) justo, podemos seguir com isso 
