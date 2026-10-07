# Notebooks

Os notebooks do projeto são organizados em três categorias:

```
notebooks/
├── experiments/      # Experimentos de treino da MarkerNet (um por experimento)
├── preprocessing/    # Pré-processamento dos dados (Cellpose + RGBA + mapa de distância)
└── exploration/      # Análises exploratórias e testes pontuais
```

## `experiments/`

Notebooks de treino da MarkerNet com a rede final congelada (ScribblePrompt). Cada experimento usa o módulo de treino de `src/`
(`load_preprocessed`, `run_kfold`, `run_final` e `save_run`): validação cruzada em 5 *folds* nas 37 imagens de treino, treino
final com todas elas e uma única avaliação nas 14 de teste, sempre ao lado do Cellpose. Os resultados vão para
`docs/estudo/resultados/<experimento>/`.

> Os experimentos 1 a 6, anteriores a esse módulo, estão no histórico do git (por exemplo,
> `git show df0a30f:notebooks/experiments/experiment_4.ipynb`); os resultados e a análise deles estão em
> `docs/estudo/06-experimentos.md`.

Os experimentos dependem dos dados gerados por `preprocessing/preprocessamento_monuseg_persistido.ipynb`.

## `preprocessing/`

Pré-processamento executado **uma única vez** e persistido em disco, para que os experimentos não reprocessem os dados.

| Notebook | Descrição |
|---|---|
| `preprocessamento_monuseg.ipynb` | Tutorial didático do pipeline (Cellpose → RGBA), passo a passo. |
| `preprocessamento_monuseg_persistido.ipynb` | **Fluxo canônico:** pré-processa treino (37) + teste (14) em 1000×1000 e persiste em `data_source/MoNuSegPreprocessed/{train,test}/` (via `SaveResultsStep`), gravando `meta.json` com a procedência. |

## `exploration/`

Análises exploratórias, inspeção de dados e validações pontuais (não fazem parte do fluxo de experimentos).

| Notebook | Descrição |
|---|---|
| `analise_image.ipynb` | Análise de imagens do MoNuSeg. |
| `groud_truth_monuseg.ipynb` | Inspeção das anotações (XML → máscara) do ground truth. |
| `inspecao_pipeline.ipynb` | **Conferência antes de um treino longo:** lê uma imagem do disco, mostra as entradas, monta o pipeline do experimento-base (MarkerUNet + ScribblePrompt no modo `positive`, 256²), confere gradiente, congelamento e scribbles, e faz um teste de sobreajuste nessa imagem, com o Cellpose ao lado. Roda em CPU (~3 min). |
| `oraculo_scribbleprompt.ipynb` | **Oráculo da rede final** (sem treino): entrega ao ScribblePrompt marcadores tirados do GT e mede o Dice por modo do canal negativo e por resolução. Roda em CPU. |
| `cellpose_parametros.ipynb` | Teste do diâmetro e dos limiares do Cellpose nas 37 imagens de treino; resultados em `docs/estudo/resultados/`. Precisa de GPU. |

## Fluxo recomendado

1. **Pré-processar uma vez:** `preprocessing/preprocessamento_monuseg_persistido.ipynb` (no Colab, com GPU).
2. **Rodar experimentos:** um notebook em `experiments/` por experimento, usando o módulo de treino.
3. **Explorar/validar:** `exploration/` quando precisar inspecionar dados ou testar o pipeline isoladamente.
