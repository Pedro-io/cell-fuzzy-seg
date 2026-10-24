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
| `recortes.ipynb` | o `base_dice` com **recortes**: treino com 4 recortes de 256² por imagem por época (MarkerUNet sem redução, ScribblePrompt em 128²); avaliação da imagem inteira em 25 blocos de 256² com sobreposição de 1/4 e janela de Hann. |
| `recortes_negativo.ipynb` | os **recortes** com o **negativo do Cellpose** (só Dice): junta as duas mudanças que mais renderam. |
| `recortes_negativo_dmap.ipynb` | o `recortes_negativo` com o termo **DMap** da tese (0,1) sobre o marcador: empurra o marcador para o centro dos núcleos e para fora do fundo. |
| `recortes_negativo_bce.ipynb` | o `recortes_negativo` com **Dice + entropia cruzada binária** (1,0) sobre a segmentação: pune cada pixel de fundo com probabilidade alta. |
| `recortes_negativo_tese.ipynb` | o `recortes_negativo` com as **perdas da tese sem o TV**: Dice + Size (0,1) + DMap (1/3000), os pesos da tese divididos por λ_seg (o DMap quase não age com esse peso). |
| `recortes_negativo_tese_bce.ipynb` | o mesmo conjunto da tese sem o TV **+ BCE** (1,0). |
| `recortes_negativo_bce_dmap.ipynb` | o `recortes_negativo` com **BCE (1,0) + DMap (0,1)**: as duas perdas que atacam a precisão, juntas. |
| `recortes_negativo_r2_fechamento.ipynb` | o `recortes_negativo` com **faixa livre do negativo de 2 px** (era 3 px) e **fechamento de 3 px do marcador no treino e na avaliação** (`MarkerClosingStep`), as duas mudanças externas que funcionaram sem retreinar; a seção 8b avalia os modelos dos *folds* com a melhor inferência (sobreposição 1/2 e TTA) contra o Cellpose e mede a editabilidade do marcador cru e do fechado. |
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
| `testes_pos_treino.ipynb` | **Testes depois do treino**, sem retreinar: carrega os modelos dos *folds* de um experimento (padrão `recortes_negativo`) e compara, nas 37 imagens, a base com a interação iterativa do ScribblePrompt (2 e 3 passadas, `mask_input`) e com um negativo entre núcleos vizinhos do Cellpose. Mostra tudo no notebook; salvar é opcional. Precisa de GPU. |
| `testes_pos_treino_2.ipynb` | **Configurações externas à MarkerUNet**, sem retreinar, com os modelos dos *folds* de `recortes_negativo`: faixa livre do negativo do Cellpose (1, 2, 5 px), τ do canal positivo (0,3, 0,4, 0,6), sobreposição dos blocos (1/2), entrada do ScribblePrompt (160², 192²) e TTA (8 rotações e espelhamentos), um fator por vez contra a base, e depois a combinação dos que vencerem (Δ > 0, maioria das imagens melhorando, Wilcoxon p < 0,05). Mostra tudo no notebook; salvar é opcional. Precisa de GPU. |
| `testes_pos_treino_3.ipynb` | **Fechamento morfológico dos marcadores** (dilatação seguida de erosão por um disco de 1, 2, 3 e 5 px, `MarkerClosingStep`) antes do ScribblePrompt, sem retreinar, com os modelos de `recortes_negativo` e a melhor inferência (faixa do negativo 2 px, sobreposição 1/2, TTA). Mede a segmentação (pareada contra o raio 0) e a editabilidade do marcador (componentes, fragmentos, buracos, regiões do GT partidas ou unidas), com uma figura do marcador cru e fechado. Mostra tudo no notebook; salvar é opcional. Precisa de GPU. |
| `editabilidade_experimentos.ipynb` | **Editabilidade dos marcadores** de `recortes_negativo`, `recortes_negativo_dmap` e `recortes_negativo_bce`, sem retreinar: Dice (como foi treinado, melhor inferência e melhor inferência com fechamento de 3 px, pareado contra o `recortes_negativo`) e editabilidade do marcador (área, componentes, fragmentos, buracos, regiões do GT partidas ou unidas; `src/evaluation/editability.py`) com fechamento de 0, 2 e 3 px, e figura dos três marcadores na mesma região. Mostra tudo no notebook; salvar é opcional. Precisa de GPU (~20 min). |
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
