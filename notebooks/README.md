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

| Notebook | Descrição | Resultado na validação (14 imagens de teste) |
|---|---|---|
| `experiment_1.ipynb` | Primeira versão pós-investigação. MoNuSeg bruto redimensionado para 256² (Cellpose roda em 256²). Loss Dice + RMSE + Size + DMap, 50 épocas. | val_loss 1,1369 (sem métricas binárias) |
| `experiment_2.ipynb` | Igual ao 1, com resolução de trabalho 1000². | val_loss 1,1081 (sem métricas binárias) |
| `experiment_3.ipynb` | Primeiro a **carregar os dados pré-processados do disco** (`data_source/MoNuSegPreprocessed/`). Acrescenta TV + Border. | val_loss 1,1779 (sem métricas binárias) |
| `experiment_4.ipynb` | Dice + RMSE + Size(0,02) + DMap(0,2). Primeiro com métricas binárias. ⚠️ As células finais têm saídas de outro run. | **Dice 0,648 / IoU 0,481** |
| `experiment_5.ipynb` | Ablação `scribble_mode="dense_soft"`, 500 épocas. | Dice 0,598 / IoU 0,428 |
| `experiment_6.ipynb` | Volta ao `sharpened` + TV + Border, 200 épocas. | Dice 0,578 / IoU 0,408 |

Referência: o **Cellpose sozinho** (a máscara que entra na MarkerNet) tem Dice 0,810 / IoU 0,682 no mesmo conjunto.
Os valores de val_loss não são comparáveis entre experimentos, porque a composição da loss muda. Detalhes em `docs/estudo/06-experimentos.md`.

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
| `oraculo_scribbleprompt.ipynb` | **Oráculo da rede final** (sem treino): entrega ao ScribblePrompt marcadores tirados do GT (GT inteiro, miolo, centros, máscara do Cellpose, vazio) e mede o Dice por modo do canal negativo e por resolução. Diz qual é o teto do ScribblePrompt neste pipeline (PD-03). Roda em CPU. |

## Fluxo recomendado

1. **Pré-processar uma vez:** `preprocessing/preprocessamento_monuseg_persistido.ipynb` (no Colab, com GPU).
2. **Rodar experimentos:** copie o experimento mais recente (`experiments/experiment_6.ipynb`), que carrega os dados persistidos do disco.
3. **Explorar/validar:** `exploration/` quando precisar inspecionar dados ou testar o pipeline isoladamente.
