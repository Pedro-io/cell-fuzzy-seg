# Notebooks

Os notebooks do projeto são organizados em três categorias:

```
notebooks/
├── experiments/      # Treinamento/avaliação da MarkerNet (um por experimento)
├── preprocessing/    # Pré-processamento dos dados (Cellpose + RGBA + mapa de distância)
└── exploration/      # Análises exploratórias e testes pontuais
```

## `experiments/`

Notebooks de treinamento da MarkerNet com a rede final congelada (ScribblePrompt) como camada de loss.

| Notebook | Descrição |
|---|---|
| `experiment_1.ipynb` | Primeira versão corrigida (split sem vazamento, resolução 256², correções P1–P12). |
| `experiment_2.ipynb` | Variação do experimento 1 (resolução de trabalho 1000×1000). |
| `experiment_3.ipynb` | Versão atual: **carrega os dados pré-processados do disco** (`data_source/MoNuSegPreprocessed/`), sem reprocessar. |

> Os notebooks de experimento dependem dos dados gerados por `preprocessing/preprocessamento_monuseg_persistido.ipynb` — rode-o antes na primeira execução.

## `preprocessing/`

Pré-processamento executado **uma única vez** e persistido em disco, para que os experimentos não reprocessem os dados a cada execução.

| Notebook | Descrição |
|---|---|
| `preprocessamento_monuseg.ipynb` | Tutorial didático do pipeline (Cellpose → RGBA), passo a passo. |
| `preprocessamento_monuseg_persistido.ipynb` | **Fluxo canônico:** pré-processa treino (30) + teste (14) em 1000×1000 e persiste em `data_source/MoNuSegPreprocessed/{train,test}/` (via `SaveResultsStep`), gravando `meta.json`. |

## `exploration/`

Análises exploratórias, inspeção de dados e validações pontuais (não fazem parte do fluxo de experimentos).

| Notebook | Descrição |
|---|---|
| `analise_image.ipynb` | Análise de imagens do MoNuSeg. |
| `groud_truth_monuseg.ipynb` | Inspeção das anotações (XML → máscara) do ground truth. |
| `test_e2e_pipeline.ipynb` | Teste ponta a ponta do pipeline com uma imagem. |

## Fluxo recomendado

1. **Pré-processar uma vez:** `preprocessing/preprocessamento_monuseg_persistido.ipynb` (no Colab, com GPU).
2. **Rodar experimentos:** `experiments/experiment_3.ipynb` (carrega os dados persistidos do disco).
3. **Explorar/validar:** `exploration/` quando precisar inspecionar dados ou testar o pipeline isoladamente.
