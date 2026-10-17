# Notebooks

Os notebooks do projeto são organizados em três categorias:

```
notebooks/
├── experiments/      # Experimentos de treino da MarkerNet (um por experimento)
├── preprocessing/    # Pré-processamento dos dados (Cellpose + RGBA + mapa de distância)
└── exploration/      # Análises exploratórias e testes pontuais
```

## `experiments/`

Notebooks de treino da MarkerNet com a rede final congelada (ScribblePrompt). Todos seguem a mesma estrutura e usam o módulo de
`src/` (`load_preprocessed`, `run_kfold`, `save_run`):

1. **Notebooks de experimento** (um por configuração): só a **validação cruzada** em 5 *folds* nas 37 imagens de treino, sempre
   ao lado do Cellpose. Cada *fold* treina um modelo novo, guarda a melhor época e grava o modelo (no Colab, em
   `MyDrive/cell-fuzzy-models/<experimento>/fold<k>.pt`). A figura mostra imagens da validação, cada uma prevista pelo modelo
   do *fold* em que ficou de fora. **O teste não é carregado.** Resultados em `docs/estudo/resultados/<experimento>/`.
2. **`experimento_final.ipynb`:** depois de escolher a configuração **só pela validação cruzada**, treina-a com as 37 imagens
   (pelo número de épocas registrado na validação dela) e avalia as 14 de teste **uma única vez**. Confere que a configuração
   copiada é a mesma do experimento escolhido. Resultados em `docs/estudo/resultados/final_<experimento>/`.

| Notebook | Experimento |
|---|---|
| `base_dice.ipynb` | experimento-base **sem o TV**: só soft Dice, ScribblePrompt em 256² no modo `positive` (τ = 0,5), MarkerUNet com alpha = máscara do Cellpose. |
| `negativo_sem_tv.ipynb` | só Dice, com o **negativo do Cellpose** (pixels a mais de 3 px de qualquer máscara do Cellpose) como canal negativo fixo do ScribblePrompt. |
| `dice_dmap.ipynb` | Dice + DMap (0,1; mapa por núcleo). |
| `rmse.ipynb` | RMSE no lugar do Dice. |
| `tese.ipynb` | combinação da tese: Dice + Size (0,1) + DMap (1/3000) + TV (1/3000). |
| `experimento_final.ipynb` | treino final da configuração escolhida e avaliação única no teste. |

Todos usam a mesma configuração de treino (só a perda e o negativo mudam) e precisam de GPU.

### `experiments/legado/`

Experimentos já rodados que **não seguem a estrutura atual**: avaliam o teste no fim e não gravam os modelos. A primeira célula
de cada um diz o que falta, o resultado e o que se aprendeu. Os resultados continuam em `docs/estudo/resultados/<nome>/`.

| Notebook | Experimento | Validação / teste (Cellpose 0,845 / 0,837) |
|---|---|---|
| `base_dice_tv.ipynb` | experimento-base: soft Dice + TV (0,001) | 0,775 / 0,777 |
| `dice_size.ipynb` | Dice + Size (0,1) | 0,774 / 0,789 |
| `base_negativo.ipynb` | Dice + TV com o negativo do Cellpose | 0,808 / 0,803 |

> Os experimentos 1 a 6, anteriores ao módulo de treino, estão no histórico do git (por exemplo,
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
| `oraculo_negativo.ipynb` | **Oráculo com negativo do Cellpose** (sem treino): marcadores do GT e do Cellpose × canais negativos (zero, complemento, fundo do Cellpose denso e esparso, fundo do GT) × 128²/256² × limiar de saída × recorte pelo Cellpose, com as imagens em 20× separadas; resultados em `docs/estudo/resultados/oraculo_negativo/`. Roda em CPU (~20 min). |
| `oraculo_resolucao.ipynb` | **Oráculo de resolução** (sem treino): marcadores do GT (GT inteiro, miolo, centros) × negativo (zero, fundo do Cellpose) × entrada do ScribblePrompt — imagem inteira em 256², 512² e 1024² ou **blocos sobrepostos** (128 → 128², 256 → 256², 256 → 128²) combinados com janela de Hann; imagens em 20× separadas. Mostra tudo no notebook; salvar é opcional. Com GPU, poucos minutos. |
| `diagnostico_marcadores.ipynb` | **Diagnóstico dos marcadores aprendidos:** treina o experimento-base nos 5 *folds*, guarda o melhor modelo de cada um e, sem retreinar, testa nas 37 imagens de treino o limiar de saída, o marcador binarizado e o negativo do Cellpose; mede onde cai o marcador, onde ficam os falsos positivos e quantos núcleos a saída funde. Mostra tudo no notebook; salvar é opcional. Precisa de GPU (~40 min numa T4). |
| `cellpose_parametros.ipynb` | Teste do diâmetro e dos limiares do Cellpose nas 37 imagens de treino; resultados em `docs/estudo/resultados/`. Precisa de GPU. |

### `exploration/legado/`

| Notebook | Descrição |
|---|---|
| `oraculo_scribbleprompt.ipynb` | Primeiro oráculo da rede final, com dados que já foram corrigidos (GT antigo, 30 imagens, parâmetros antigos do Cellpose). Substituído por `oraculo_negativo.ipynb` e `oraculo_resolucao.ipynb`; a primeira célula resume o resultado e o que se manteve. |

## Fluxo recomendado

1. **Pré-processar uma vez:** `preprocessing/preprocessamento_monuseg_persistido.ipynb` (no Colab, com GPU).
2. **Rodar experimentos:** um notebook em `experiments/` por configuração (só validação cruzada; modelos gravados no Drive).
3. **Escolher** a configuração **só pela validação cruzada** e rodar `experiments/experimento_final.ipynb` uma única vez.
4. **Explorar/validar:** `exploration/` quando precisar inspecionar dados ou testar o pipeline isoladamente.
