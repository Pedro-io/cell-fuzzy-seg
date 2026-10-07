# 05 — Treinamento

> Legenda: ✅ verificado no código, nos notebooks ou medido · 📜 histórico/commits/investigação · 📖 tese · ❓ a confirmar.
> Problemas ficam em [08-pendencias.md](08-pendencias.md) e são citados pelo ID.
>
> Última verificação: 2026-09-27 (commit `f9b1e9b`).

---

## 1. Resumo

- O treino tem três camadas:
  - o **`TrainingLoop`** percorre épocas e batches;
  - o **`Trainer`** faz um passo de otimização (forward → loss → backward → clip → step → scheduler);
  - os **callbacks** observam o processo: o `GradNormCallback` e, desde 2026-10-07, o `BestModelCallback` (melhor Dice de
    validação + parada antecipada, PD-46). ✅
- Só a **MarkerUNet** é otimizada; o ScribblePrompt está congelado. Todos os experimentos usam **Adam, lr 1e-4, cosseno por
  passo, clip de gradiente 1,0, batch 4 e seed 42**, e mudam só as losses e o número de épocas. ✅
- **O treino é barato: ~1,3–1,7 s por época**, ou ~1,5 min para 50 épocas no Colab. Isso torna viáveis a validação cruzada
  (PD-02) e as ablações (PD-38, PD-44). ✅
- Pontos fracos, quase todos já registrados:
  - a validação é feita no teste oficial (PD-02);
  - a aumentação é estática (PD-08);
  - os runs provavelmente validaram com a BatchNorm em modo treino (PD-09);
  - não há checkpoints desde o exp. 3 (PD-10);
  - não há escolha do "melhor modelo" (PD-46, nova).

---

## 2. O laço de treino ✅

```
TrainingLoop.run()                                         training_loop.py
└── para cada época:
    ├── para cada batch de treino:  Trainer.train_step(batch)
    │     1. _to_device(batch)                     move só os tensores (o "id" continua lista)
    │     2. set_training(True) nos Steps          MarkerUNet em train()
    │     3. optimizer.zero_grad()
    │     4. TrainingPipeline.run(data)            MarkerStep → FrozenSegmentationStep (com grafo)
    │     5. LossComposer(segmentation, distance_map, ground_truth, markers)
    │     6. loss.backward()
    │     7. clip_grad_norm_(params do otimizador, grad_clip)
    │     8. optimizer.step();  scheduler.step()   (o scheduler anda a cada PASSO, não a cada época)
    │     9. callbacks.on_train_step_end(...)
    ├── se época % validate_every == 0:
    │     para cada batch de validação: Trainer.eval_step(batch)   @torch.no_grad, set_training(False)
    ├── callbacks.on_epoch_end(...)
    └── log a cada log_every épocas
→ devolve history = {"train_loss", "val_loss", "train_terms", "val_terms"}   (médias por época)
```

### 2.1 `Trainer` — [trainer.py](../../src/training/trainer.py)

| Parte | Onde | Detalhe |
|---|---|---|
| construtor | [L119-150](../../src/training/trainer.py#L119-L150) | recebe o pipeline, o composer, o otimizador, o scheduler, os callbacks, o `grad_clip` e as chaves (`prediction_key="segmentation"`, `distance_map_key`, `ground_truth_key`) |
| `train_step` | [L152-190](../../src/training/trainer.py#L152-L190) | os 9 passos acima. O clip usa **os parâmetros do otimizador**, ou seja, só os da MarkerUNet |
| `eval_step` | [L192-218](../../src/training/trainer.py#L192-L218) | `@torch.no_grad()`, modo avaliação; chama `on_validation_step_end` |
| `_compute_loss` | [L220-242](../../src/training/trainer.py#L220-L242) | tira as chaves do `data` e chama o composer; os `markers` são opcionais |
| `_set_pipeline_training_mode` | [L244-249](../../src/training/trainer.py#L244-L249) | chama `step.set_training(bool)` se o Step tiver esse método (*duck typing*) |

📜 **Por que o `Trainer` é o único a chamar `backward()`** (regra 23): na versão de maio, `MarkerNet.train_step()` fazia tudo por
dentro, e o notebook fazia o backward por fora ("para fins didáticos", segundo o antigo jornal). Havia dois caminhos para a mesma
coisa. Centralizar evita, por exemplo, esquecer o `zero_grad`.

📜 **Por que o `set_training`** (P3 da investigação): a MarkerUNet ficava em `eval()` o tempo todo, e as BatchNorms do decoder
novo não se adaptavam. Desde 22/09 (`ef7d731`), o `Trainer` coloca em `train()` no treino e em `eval()` na validação. Os runs
dos exp. 1 a 6 são **anteriores** a isso (PD-09).

### 2.2 `TrainingLoop` — [training_loop.py](../../src/training/training_loop.py)

- `run()` ([L78-123](../../src/training/training_loop.py#L78-L123)) acumula as médias por época; `validate_every` e `log_every`
  controlam a frequência. Nos notebooks, `validate_every=1`, e o log a cada 5 épocas.
- **A média é por batch, não por imagem** ([L139-160](../../src/training/training_loop.py#L139-L160)). Com 30 imagens de treino
  e batch 4, o último batch tem 2 imagens e pesa como os de 4; o mesmo vale para a validação (14 = 4+4+4+2). O efeito é pequeno,
  mas soma-se às normalizações por batch das losses (PD-12, PD-45).
- **Não há early stopping nem "guardar o melhor modelo"**: o modelo avaliado no fim é o da **última época** (PD-46).
- Os iteráveis são **listas de batches prontas** (`train_batches`), não `DataLoader`s. Não há shuffle, e a aumentação foi
  sorteada uma vez só (PD-08).

### 2.3 Callbacks — [grad_norm_callback.py](../../src/training/callbacks/grad_norm_callback.py)

- A interface `TrainerCallback` ([trainer.py:22-70](../../src/training/trainer.py#L22-L70)) exige três métodos:
  `on_train_step_end`, `on_validation_step_end` e `on_epoch_end`.
- O `GradNormCallback` registra, a cada passo, a norma L2 total, a média e o máximo absolutos, e a fração de parâmetros com
  gradiente.
  - 📜 Nasceu para diagnosticar o P1 da investigação: a loss "perfeitamente plana" no exp. 1 antigo, com gradiente ~0 em float32.
  - ⚠️ Ele roda **depois** do clip, então o valor 1,0 só mostra que o clip atuou (PD-14).
- Ainda não existem os callbacks citados no antigo `ARCHITECTURE.md` (`EarlyStoppingCallback`, `CheckpointCallback`).

---

## 3. Configuração usada nos experimentos ✅ (lida dos notebooks)

| | Exp. 1–4 | Exp. 5 | Exp. 6 | Tese (modelo de objeto, 6.1.1) 📖 |
|---|---|---|---|---|
| Otimizador | Adam | Adam | Adam | Adam |
| lr | 1e-4, cosseno até 0 | idem | idem | **5e-4 fixo** |
| Clip de gradiente | 1,0 | 1,0 | 1,0 | — |
| Batch | 4 | 4 | 4 | **32** |
| Passos por época | 8 | 8 | 8 | — |
| Épocas / passos totais | 50 / 400 | **500 / 4.000** | 200 / 1.600 | — / **2.250** |
| Amostras vistas (passos × batch) | 1.600 | 16.000 | 6.400 | **72.000** |
| Seeds | `torch` e `numpy` = 42 | idem | idem | — |
| Dados | 30 imagens | 30 | 30 | 9.000 imagens (COCO) |

- 📜 **De onde vêm os valores:** o experimento E6 da investigação do exp. 1 recomendou "batch 2–4, `train()` na MarkerNet, grad
  clip 1.0, scheduler (CosineAnnealing), lr menor (1e-4)". Antes era lr 1e-3 fixo, batch 1 e sem clip (P4, P5).
- **O cosseno leva o lr a zero no último passo** (`T_max = épocas × 8`). Mudar o número de épocas muda também a curva do lr,
  então o exp. 5 não é "o exp. 4 treinado por mais tempo": ele tem um lr alto por muito mais passos.
- O otimizador recebe só `marker_net.parameters()`. Os 24,4 M parâmetros têm o mesmo lr (PD-38).

### 3.1 Tempo e custo ✅

Medido pelos timestamps dos logs do `TrainingLoop` (8 batches de treino + 4 de validação por época):

| Exp. | s/época | Total |
|---|---|---|
| 1 | 1,3 | ~1 min |
| 2–4 | 1,6 | ~1,5 min |
| 5 | 1,6 | ~13 min |
| 6 | 1,7 | ~6 min |

**Consequências:**
- Uma validação cruzada de 5 *folds* × 50 épocas custa ~7 minutos por configuração. A ablação de 5 variantes da PD-38 fica em
  ~35 minutos. O receio de "gastar imagens" com validação (PD-02) não tem custo computacional relevante.
- Com o ScribblePrompt em 256² (PD-44), a rede final faz ~4× mais contas. ❓ O tempo total deve subir menos que isso, porque a
  MarkerUNet e as operações em 1000² não mudam.
- A GPU não aparece nos notebooks; segundo o autor, é uma **T4** do Colab. 500 épocas levam ~13 min (confirmado pelo autor).

---

## 4. Dados no treino (ligação com o tema 1) ✅

- **Batches:** `build_batch` empilha 4 amostras (imagem (B,3,1000,1000) em [0,255]; rgba, GT e Dmap em [0,1]). `make_batches`
  roda **uma vez**, antes do treino ([01-dados.md §7.2](01-dados.md)).
- **Aumentação:** rot90 × k, flip horizontal e flip vertical, sorteados **uma vez** com `np.random.seed(42)`. Todas as épocas
  veem as mesmas 8 versões aumentadas, na mesma ordem (PD-08). É como se a aumentação não existisse para a generalização: são
  sempre as mesmas 30 imagens.
- **Validação:** as 14 imagens de teste, sem aumentação (PD-02).
- **Determinismo:** as seeds são fixas, mas não há `torch.backends.cudnn.deterministic`. ❓ Dois runs iguais podem diferir um
  pouco na GPU.

---

## 5. Como os resultados são medidos ✅

- **Durante o treino:** a `val_loss` (a mesma loss composta) em toda época. Ela **não é comparável entre experimentos**, porque os
  termos e pesos mudam ([04-losses.md §2.3](04-losses.md)).
- **No fim (exp. 4 a 6):** `evaluate_val_quality` roda o pipeline sobre a validação e calcula, **por imagem**, com limiar 0,5:
  Dice, precisão, recall, IoU, `mass_ratio` e `marker_frac` (a fração da imagem com marcador ≥ 0,5). Depois tira a média. É o
  mesmo cálculo que o notebook do oráculo usa.
- **O modelo avaliado é o da última época**, não o de melhor `val_loss` (PD-46). Com o cosseno terminando em lr 0, o fim do treino
  costuma ser estável, mas o número "melhor val_loss (época X)" que os notebooks imprimem **não é** o do modelo avaliado.
- Nenhuma métrica por instância (AJI, PQ): a avaliação é semântica, por decisão do coordenador.

---

## 6. Reprodutibilidade 📜✅

| Item | Situação |
|---|---|
| Código usado em cada run | o Colab clona a branch no momento do run; o commit exato não é registrado (é por isso que PD-09 e P10 ficam ❓) |
| Checkpoint da MarkerUNet | salvo nos exp. 1–2; **comentado** nos exp. 3–6 (PD-10) |
| Configuração completa | espalhada pelas células; não é salva junto com o resultado |
| Saídas dos notebooks | o exp. 4 tem células com saída de outro run; o exp. 5 foi rodado num kernel reaproveitado (PD-10) |
| Seeds | fixas; o cuDNN não é determinístico |

📜 A investigação já recomendava (E8): "descomentar o checkpoint e salvar também os markers, a segmentação, o grad-norm médio e a
config completa, para que qualquer run futuro seja auditável". **Sugestão prática:** no começo de cada run, imprimir e salvar o
`git rev-parse HEAD` junto com um JSON de configuração e o checkpoint.

---

## 7. Decisões (2026-09-27)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | GPU | **T4** (pela lembrança do autor). 500 épocas levam **13 min** (logs do exp. 5: 00:54:37 → 01:07:52, em 11/09); o autor confirmou esse número | §3.1 |
| 2 | Guardar o melhor modelo | Sim: checkpoint de **melhor Dice na validação do *fold*** | PD-46 |
| 3 | *Early stopping* | Sim, pela validação do *fold* | PD-46 |
| 4 | Módulo em `src/` para dados, aumentação e k-fold | Sim | PD-19, PD-08 |

### Perguntas originais (histórico)

1. **GPU:** qual GPU o Colab costuma te dar (T4, L4, A100)? Vale anotar nos notebooks, porque o tempo por época depende dela. T4 se não me engano, mas costuma levar 30 min para 500 épocas
2. **Melhor modelo × último modelo (PD-46):** quer que o treino guarde o checkpoint de melhor Dice na validação (dentro do *fold*,
   não no teste) e avalie esse? sim, melhor 
3. **Épocas:** o exp. 5 (500 épocas) teve *overfitting* claro. Com o treino tão barato, topa usar *early stopping* pela validação
   do *fold* em vez de fixar as épocas? Sim, vamos fazer isso 
4. **Código repetido nos notebooks (PD-19, PD-08):** `load_preprocessed`, `build_batch`, `make_batches` e as métricas estão
   copiados em cada notebook. Quer que isso vire um módulo em `src/` (dataset que lê do disco, aumentação por época, laço de
   *k-fold*), para os próximos experimentos importarem? Sim, acho válido
   
