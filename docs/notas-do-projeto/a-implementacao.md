# Apêndice A. Implementação

Este apêndice descreve o código: como o repositório está organizado, os padrões que ele segue, o dicionário que circula
entre as etapas, as regras de arquitetura, como rodar e como testar. Os capítulos 3 a 9 explicam o que cada parte faz; aqui
está como as partes se encaixam.

## A.1 Estrutura do repositório

| Pasta ou arquivo | Papel |
|---|---|
| [src/data/load/](../../src/data/load/) | leitura do MoNuSeg (`MonusegDataset`, com a rasterização dos XMLs) e dos dados preparados (`load_preprocessed`, `PreprocessedDataset`) |
| [src/pipeline/](../../src/pipeline/) | pipelines e passos: preparação (Cellpose, RGBA, mapa de distância, gravação), inferência (MarkerUNet, ScribblePrompt) e execução em blocos |
| [src/models/](../../src/models/) | a MarkerUNet, o wrapper do ScribblePrompt e as interfaces das redes |
| [src/losses/](../../src/losses/) | os termos de perda e o compositor |
| [src/training/](../../src/training/) | laço de treino, *callbacks*, validação cruzada e gravação de modelos |
| [src/evaluation/](../../src/evaluation/) | métricas por imagem, resumo estatístico, registro dos resultados e figuras |
| [src/io/](../../src/io/), [src/utils/](../../src/utils/) | gravação em disco, *logger*, conversões de imagem |
| [configs/datasets.yml](../../configs/datasets.yml) | caminhos do MoNuSeg |
| [notebooks/](../../notebooks/) | preparação dos dados, experimentos e explorações; ver [notebooks/README.md](../../notebooks/README.md) |
| [tests/](../../tests/) | testes automatizados (seção A.9) |
| [data_source/](../../data_source/) | MoNuSeg original e dados preparados, versionados no repositório |
| [docs/estudo/](../estudo/) | registros de trabalho: pendências (08), correções e experimentos (09), oráculos (11), resultados |

## A.2 Padrões

**Pipeline.** Todo processamento é uma sequência de passos (*Steps*). Cada passo recebe um dicionário, acrescenta chaves e o
devolve; um pipeline só executa a lista de passos em ordem ([base_step.py](../../src/pipeline/steps/base_step.py)). Isso
permite trocar, reordenar ou testar passos isoladamente e reaproveitá-los em notebooks diferentes.

**Strategy.** Duas partes do método são trocáveis sem editar o código:

- as **perdas**: cada termo implementa `LossTerm`, e o experimento monta a lista de termos que o `LossComposer` soma (capítulo 8);
- a **camada de segmentação**: qualquer rede que implemente `BaseFinalSegmentation`
  ([base_final_segmentation.py](../../src/models/networks/final_segmentation/base_final_segmentation.py)) pode substituir o
  ScribblePrompt. O contrato é `forward({"image", "scribbles"})`, que devolve um tensor `(N, 1, H, W)` em $[0,1]$, e a chave
  opcional `"negative"` para o canal de fundo.

**Montagem nos notebooks.** Cada notebook de experimento define uma função que monta, do zero, a MarkerUNet, o pipeline e as
perdas. A validação cruzada chama essa função uma vez por *fold*, o que garante um modelo novo em cada um.

## A.3 O dicionário de dados

| Chave | Quem cria | Formato | Quem usa |
|---|---|---|---|
| `id` | `MonusegDataset` | texto | gravação, relatórios |
| `image` | `MonusegDataset` | `(H, W, 3)` uint8; no lote, `(B, 3, H, W)` float em $[0, 255]$ | Cellpose, RGBA, ScribblePrompt |
| `ground_truth` | `MonusegDataset` | `(H, W)` 0 ou 1; no lote, `(B, 1, H, W)` | mapa de distância, perdas, métricas |
| `ground_truth_instances` | `MonusegDataset` | `(H, W)` int32, um rótulo por núcleo; não é gravado | mapa de distância |
| `segmentation` (na preparação) | `CellposeStep` | `(H, W)` instâncias do Cellpose | RGBA, gravação |
| `cellpose_prob` | `CellposeStep` | `(H, W)` float16 em $[0,1]$ | RGBA com `alpha="prob"`, gravação |
| `rgba` | `RGBAStep` | `(H, W, 4)` float32 em $[0,1]$ | `MarkerStep` |
| `distance_map` | `DistanceMapStep` | `(H, W)` float32 em $[0,1]$ | perda $L_{Dmap}$ |
| `cellpose_segmentation` | `load_preprocessed` | a segmentação do Cellpose, renomeada na leitura | linha de base, canal de fundo |
| `cellpose_negative` | `load_preprocessed` (opcional) | `(H, W)` 0 ou 1 | ScribblePrompt, como canal de fundo |
| `markers` | `MarkerStep` | `(B, 1, H, W)` em $[0,1]$, com gradiente | ScribblePrompt, perdas, métricas do marcador |
| `segmentation` (no treino) | `FrozenSegmentationStep` | `(B, 1, H, W)` em $[0,1]$, com gradiente | perdas, métricas |

A chave `segmentation` tem dois significados: a máscara do Cellpose na preparação e a saída final no treino. Na leitura dos dados
preparados, a máscara do Cellpose é renomeada para `cellpose_segmentation`, então as duas nunca convivem no mesmo dicionário.
Renomear a saída final para `final_segmentation` está decidido e ainda não aplicado (capítulo 12, D-17).

## A.4 Pipelines e passos

| Classe | Passos | Quando roda |
|---|---|---|
| [PreprocessingPipeline](../../src/pipeline/preprocessing_pipeline.py) | `CellposeStep`, `RGBAStep`, `DistanceMapStep`, `SaveResultsStep` | uma vez, no notebook de preparação, com GPU |
| [TrainingPipeline](../../src/pipeline/training_pipeline.py) | `MarkerStep`, `FrozenSegmentationStep` | a cada lote, no treino e na avaliação |
| [TiledPipeline](../../src/pipeline/tiled_pipeline.py) | envolve um `TrainingPipeline` | no modo por recortes: passa os recortes direto e divide imagens maiores em blocos |

As duas primeiras classes fazem a mesma coisa (percorrem os passos em ordem); a separação é de intenção, e nada no código impede
misturar passos. Existe ainda um `ModelPipeline` sem uso.

O pipeline não alterna os modos de treino e avaliação das redes. Quem faz isso é o `Trainer` (e a função `evaluate`), chamando
`set_training` nos passos que têm esse método. O ScribblePrompt fica sempre em modo de avaliação.

## A.5 Camadas e dependências

```
notebooks           montam tudo: dados, passos, pipelines, perdas, Trainer
   |
src/training  --->  src/pipeline  --->  src/models
                         |
                         +--> src/io
src/losses, src/data, src/evaluation    independentes entre si
todos         --->  src/utils
```

Os imports seguem essa direção: `training` não importa `models`, `models` não conhece `data` nem `pipeline`, e `losses` não
importa nada de fora. Três acoplamentos existem sem import, por nome de atributo: o `Trainer` chama `set_training` nos passos que o
têm; o `GradNormCallback` procura os atributos `model` e `final_network` nos passos; e o `MarkerStep` acessa a `smp.Unet` dentro
da MarkerUNet por `model.model`. Renomear esses atributos quebra o treino sem erro de import
([08-pendencias.md, PD-35](../estudo/08-pendencias.md#pd-35)).

## A.6 Regras de arquitetura

As regras abaixo são numeradas porque o código ainda as cita pelo número ("regra 23"). A numeração é mantida enquanto essas
citações não forem removidas ([08-pendencias.md, PD-50](../estudo/08-pendencias.md#pd-50)).

| Nº | Regra | Situação |
|---|---|---|
| 1 | Datasets nunca executam modelos | cumprida |
| 2 | O dataset delega a preparação ao `PreprocessingPipeline` | cumprida |
| 3 | `models` não conhece `data` | cumprida |
| 4 | `models` não conhece `pipeline` | cumprida |
| 5 | O `Trainer` não implementa redes | cumprida |
| 6 | O `Trainer` não conhece detalhes das redes | cumprida no `Trainer`; o `GradNormCallback` depende de atributos dos passos |
| 7 | Perdas não executam modelos | cumprida |
| 8 | Passos de preparação não fazem retropropagação | cumprida |
| 9 | Passos de inferência não calculam perda nem chamam `optimizer.step()` | cumprida |
| 10 | O `SaveResultsStep` só entra no `PreprocessingPipeline` | cumprida por uso, não forçada |
| 11 | `utils` não tem regra de negócio | cumprida |
| 12 | `io` só grava; quem decide é o passo | cumprida |
| 13 | O `PreprocessingPipeline` não executa passos de inferência | cumprida por uso |
| 14 | O `TrainingPipeline` não executa passos de preparação | cumprida por uso |
| 15 | Uma pasta `registry/` resolve nomes em classes | obsoleta: a pasta não existe |
| 16 | Uma nova camada de segmentação implementa `BaseFinalSegmentation` | cumprida |
| 17 | Uma nova rede treinável herda de `BaseNetwork` | cumprida |
| 18 | Uma nova perda vira classe em `losses` e é usada pelo `LossComposer` | cumprida |
| 19 | Nada importa de `tests` | cumprida |
| 20 | `training` não importa `models` | cumprida |
| 21 | Novos passos entram na lista sem mudar as classes de pipeline | cumprida |
| 22 | Nenhum passo remove chaves | cumprida, mas a chave `segmentation` é sobrescrita (seção A.3) |
| 23 | Só o `Trainer` chama `backward()` e `optimizer.step()` | cumprida (conferido em `src` e nos notebooks de experimento atuais) |
| 24 | O mapa de distância só é produzido pelo `DistanceMapStep` | cumprida |

## A.7 Módulo de treino e avaliação

| Arquivo | O que faz |
|---|---|
| [trainer.py](../../src/training/trainer.py) | um passo de treino (pipeline, perda, retropropagação, recorte do gradiente, passo do otimizador e da agenda) e um passo de avaliação sem gradiente |
| [training_loop.py](../../src/training/training_loop.py) | épocas, validação e *callbacks* |
| [best_model_callback.py](../../src/training/callbacks/best_model_callback.py) | guarda o melhor estado pelo Dice de validação e para o treino sem melhora |
| [grad_norm_callback.py](../../src/training/callbacks/grad_norm_callback.py) | registra a norma do gradiente; mede depois do recorte, então o valor 1,0 só indica que o recorte agiu ([PD-14](../estudo/08-pendencias.md#pd-14)) |
| [kfold.py](../../src/training/kfold.py) | `TrainConfig`, divisão dos *folds*, `run_kfold`, `final_epochs`, `run_final` |
| [checkpoints.py](../../src/training/checkpoints.py) | grava e lê modelos com metadados; monta o pipeline a partir de um modelo gravado |
| [metrics.py](../../src/evaluation/metrics.py) | métricas por imagem, com a linha de base do Cellpose e as medidas do marcador |
| [report.py](../../src/evaluation/report.py) | resumo estatístico pareado, procedência (commit, versões, GPU) e gravação dos resultados |
| [figures.py](../../src/evaluation/figures.py) | escolha das imagens (pior, mediana e melhor contra o Cellpose) e figura das predições |

## A.8 Como rodar

**Preparação dos dados** (uma vez, ou quando o Cellpose ou o GT mudarem): `notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb`
no Google Colab com GPU. O `CellposeStep` exige CUDA.

**Um experimento:** abrir o notebook em `notebooks/experiments/` no Colab com GPU (a GPU usada até aqui é uma T4). O notebook clona
o repositório na branch de trabalho, instala as dependências, roda a validação cruzada, grava os modelos no Google Drive e os
resultados em `docs/estudo/resultados/<experimento>/`. Ele recusa gravar por cima de resultados existentes.

**O teste:** `notebooks/experiments/experimento_final.ipynb`, com a configuração escolhida copiada do experimento.

**Localmente** não há GPU, e o ambiente do repositório não tem PyTorch. Para rodar os testes ou o ScribblePrompt em CPU, usar um
ambiente virtual à parte com PyTorch para CPU, `segmentation-models-pytorch` 0.5.0, o pacote `scribbleprompt` instalado do
repositório dele sem dependências, `segment-anything` 1.0, `scikit-image`, `matplotlib`, `pandas`, `loguru` e `pytest`.

## A.9 Testes

A pasta `tests/` tem 23 arquivos e 182 funções de teste; com os casos parametrizados, a suíte tem 184 casos. Comando:

```
PYTHONPATH=. python -m pytest tests -q
```

Na execução de 2026-10-17: 180 passam, 3 falham e 1 é pulado, em cerca de 7 segundos. As 3 falhas são conhecidas: os testes do
$L_{size}$ ainda esperam a versão simétrica, revertida de propósito (capítulo 12, D-28;
[PD-11](../estudo/08-pendencias.md#pd-11)).

Os testes não usam GPU nem o Cellpose. Eles cobrem a "tubulação": formatos, chaves, fluxo do gradiente, modos de treino e
avaliação, perdas em casos simples, validação cruzada, gravação e leitura de modelos, blocos. Usam redes substitutas pequenas no
lugar das reais. Não cobrem o comportamento das redes reais, os valores das perdas em dados reais nem a execução automática a cada
mudança ([PD-48](../estudo/08-pendencias.md#pd-48)).

## A.10 Convenções

- Imports absolutos a partir de `src.`; *lint* com `ruff` (configuração no `pyproject.toml`, linhas de até 100 caracteres).
- Docstrings e comentários em português, no estilo Google (`Args:`, `Returns:`, `Raises:`).
- O código descreve o comportamento, não a história: identificadores de pendências, datas e "antes era" ficam nos documentos.
- *Logs* em nível INFO só para o que importa a quem roda (progresso, *folds*, parada, download); detalhes em DEBUG, ativados com a
  variável `CELL_FUZZY_LOG_LEVEL=DEBUG`.
- Branch de trabalho `homolog`; *pull requests* para `main`.

## A.11 Problemas conhecidos no código

| Problema | Pendência |
|---|---|
| `pipeline_test.py`, na raiz, importa módulos que não existem mais | [PD-17](../estudo/08-pendencias.md#pd-17) |
| Partes de `configs/datasets.yml` (`loader_config`, `preprocessing`) não são usadas | [PD-18](../estudo/08-pendencias.md#pd-18) |
| Arquivos `.pyc` versionados, apesar do `.gitignore` | [PD-20](../estudo/08-pendencias.md#pd-20) |
| Código sem uso: `ModelPipeline`, métodos do `OutputWriter`, `to_uint8_rgb`, `RMSEAccuracy` | [PD-31](../estudo/08-pendencias.md#pd-31) |
| O `MarkerStep` sem modelo usa a segmentação como marcador, em silêncio | [PD-16](../estudo/08-pendencias.md#pd-16) |
| Comentários e testes que citam a história do projeto | [PD-50](../estudo/08-pendencias.md#pd-50) |

## Para o artigo

- **Disponibilidade do código:** o repositório é público; citar a URL e a licença, quando escolhida (PD-27).
- **Detalhes de implementação ou material suplementar:** bibliotecas e versões (PyTorch, `segmentation-models-pytorch` 0.5.0,
  Cellpose 4.1.1, ScribblePrompt v1), GPU usada e tempo de treino.

## Referências

Sem referências externas.
