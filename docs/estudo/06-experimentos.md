# 06 — Experimentos

> Legenda: ✅ verificado nas saídas dos notebooks, no código ou medido · 📜 histórico/commits/investigação · 📖 tese · ❓ a confirmar.
> Problemas ficam em [08-pendencias.md](08-pendencias.md) e são citados pelo ID.
>
> **Este documento substitui o antigo `docs/investigacao_experimento_1.md`** (removido em 2026-09-27; recuperável com
> `git show f9b1e9b:docs/investigacao_experimento_1.md`). O código cita os IDs daquele documento ("investigação, P3", "C2"...).
> Eles estão preservados, com a situação atual de cada um, na [§2.3](#23-catálogo-da-investigação-p-c-e--situação-em-2026-09-27).
>
> Os números da §3 foram levantados das saídas salvas nos notebooks. Última verificação: 2026-09-27.
>
> **Os notebooks `experiment_1` a `experiment_6` foram removidos em 2026-10-07** (decisão do autor, PD-10): foram substituídos
> pelo módulo de treino (PD-19). Continuam recuperáveis com `git show df0a30f:notebooks/experiments/experiment_N.ipynb`. Os números
> deste documento usam o GT antigo, 30 imagens de treino e os limiares antigos do Cellpose.

---

## 1. Resumo

| | Dice | IoU | Observação |
|---|---|---|---|
| **Cellpose sozinho** (linha de base, teste) | **0,810** | **0,682** | a máscara que **entra** na MarkerUNet |
| Exp. 1 (256², Dice+RMSE+Size+DMap) | — | — | só loss; aprende (P1 resolvido) |
| Exp. 2 (= 1, em 1000²) | — | — | só loss |
| Exp. 3 (dados do disco; + TV + Border) | — | — | só loss; configuração mudou depois do run |
| **Exp. 4** (Size 0,02, DMap 0,2) | **0,648** | **0,481** | melhor resultado |
| Exp. 5 (`dense_soft`, 500 épocas) | 0,598 | 0,428 | *overfitting*; segmenta demais |
| Exp. 6 (`sharpened` + TV + Border, 200 épocas) | 0,578 | 0,408 | mais equilibrado |
| Oráculo, marcador = miolo perfeito (sem treino) | até 0,805 | — | o teto do ScribblePrompt com marcadores pequenos (PD-40) |

- **Todos os seis rodaram até o fim**, sem NaN e com a loss caindo. Nenhum passou do Cellpose (PD-01).
- As métricas são por pixel, média das 14 imagens de teste, limiar 0,5. As 14 imagens de teste também foram usadas para
  escolher configurações (PD-02).
- A `val_loss` **não é comparável** entre experimentos, porque os termos e pesos mudam ([04-losses.md §2.3](04-losses.md)).
- O oráculo ([03-modelos.md §8](03-modelos.md)) explica boa parte do quadro: mesmo com marcadores **perfeitos**, o ScribblePrompt
  só passa do Cellpose se o marcador for o núcleo inteiro.

---

## 2. Pré-história: o experimento 1 original e a investigação (ago./2026) 📜

### 2.1 O que aconteceu

A primeira versão do `experiment_1` (antes de 13/08) **não aprendia nada**: a loss ficou **idêntica em 1,6960 da época 2 à 100**
(validação 1,8406), e a melhor validação foi a da época 1. A decomposição dos termos mostrava uma saída **degenerada**: o
ScribblePrompt devolvia ≈0,98 em praticamente todos os pixels ("tudo é núcleo"). Com isso, RMSE ≈ 0,86 (quase o máximo), Dice
≈ 0,39 e massa ≈ 4,5× a do GT.

Configuração daquele run:
- batch 1, lr 1e-3 fixo, sem clip e sem scheduler;
- MarkerNet em `eval()`;
- scribbles `[s, 1−s]` densos, com s entre 0,43 e 0,79 em **todos** os pixels;
- sem `DMapTerm`, com o Cellpose caindo no modelo padrão;
- as 14 imagens de teste **misturadas ao treino**.

### 2.2 A explicação encontrada (hipótese principal da investigação)

1. A MarkerNet recém-inicializada produzia marcadores suaves em toda a imagem.
2. O ScribblePrompt, treinado com rabiscos esparsos e binários, recebia os dois canais altos em todo lugar e respondia com
   "tudo primeiro plano" saturado (~0,98).
3. Com o sigmoid saturado e a cadeia longa através da rede congelada, o gradiente que chegava à MarkerNet era ~0 em float32.
4. O Adam não mexia nos pesos, a saída não mudava e a loss ficava plana.

A investigação catalogou todas as suspeitas (P = engenharia, C = conceito, E = experimento de isolamento). As correções foram
aplicadas em 13/08 e entraram no PR #2. O exp. 1 atual é o **primeiro run depois delas**.

### 2.3 Catálogo da investigação (P, C, E) — situação em 2026-09-27

É por esses IDs que o código se refere às decisões ("investigação, P3" etc.).

| ID | O que era | O que foi feito em 13/08 | Situação hoje | Pendência |
|---|---|---|---|---|
| **P1** | gradiente efetivamente nulo até a MarkerNet (crítico) | `GradNormCallback`; checagem da magnitude do gradiente | ✅ resolvido: nos exp. 1–2 o clip atua em ~90% dos passos, então há gradiente. O callback mede depois do clip | PD-14 |
| **P2 / C2** | scribbles densos fora da distribuição do ScribblePrompt | modo `sharpened` (quase binário) | ⚠️ mitigado. O oráculo mostrou que o problema restante é o **canal negativo denso**; decidido usar só positivos | PD-04 |
| **P3** | MarkerNet em `eval()` durante o treino | `train()` quando `differentiable=True` | ✅ no código. `eval()` na validação só desde 22/09, ou seja, depois dos runs | PD-09 |
| **P4** | batch 1 | batch 4 | ✅ | — |
| **P5** | sem scheduler nem clip; lr 1e-3 | cosseno + clip 1,0 + lr 1e-4 (via E6) | ✅ | — |
| **P6** | sem aumentação | rot90 + flips | ⚠️ parcial: sorteada uma vez só | PD-08 |
| **P7** | resolução 1000→256→128→1000 | exp. 1 passou a trabalhar em 256² | ⚠️ reaberto: os exp. 2–6 voltaram a 1000², e o ScribblePrompt continua em 128² | PD-05 |
| **P8** | desbalanceamento (~77% fundo) | não tratado | aberto | PD-13 |
| **P9** | Cellpose caindo no modelo padrão | corrigido o typo `pretreined_model`; `cpsam_v2` → `cpsam`; aviso | ✅ na prática (`cpsam` é o único modelo do Cellpose 4), mas o aviso nunca executa | PD-29 |
| **P10** | `SizeTerm` = `peso·ratio` "puxando a massa para zero" | trocado por `|ratio − 1|` (simétrico) | ↩️ **revertido** em 30/08 para `peso·ratio`, que é o L_size da tese; os testes ficaram desatualizados | PD-11 |
| **P11 / C3 / C4** | mapa de distância calculado e não usado; supervisão só indireta | `DMapTerm` sobre os **marcadores**; composer com `prediction` e `markers` separados | ✅ feito, mas o mapa é normalizado por imagem, não por núcleo | PD-06 |
| **P12** | higiene: checkpoint comentado, épocas inconsistentes, teste no treino | checkpoint salvo; épocas coerentes | ⚠️ regrediu: checkpoint comentado de novo desde o exp. 3 | PD-10 |
| **C1** | ScribblePrompt como "camada de loss" é um uso fora do propósito | — (decisão de método) | 🔴 **confirmado como limitante** pelo oráculo | PD-40 |
| **C5** | GT binário funde núcleos | — | aberto (−25,5% de componentes no treino) | PD-07 |
| **C6** | teste oficial misturado ao treino | treino = 30 oficiais; validação = 14 oficiais de teste | ⚠️ não há mais vazamento no treino, mas o teste virou conjunto de **seleção** | PD-02 |
| **E1** | medir grad-norm | feito (callback) | ✅ | PD-14 |
| **E2** | sensibilidade do ScribblePrompt aos marcadores | — | ⚠️ parcial: o oráculo varia tipo de marcador, canal negativo e resolução | PD-03 |
| **E3** | oráculo: marcador = GT | — | ✅ **feito em 27/09** | PD-40 |
| **E4** | ablação da loss | — | não feito formalmente: os exp. 3–6 mudaram várias coisas por vez. Planejado | PD-44 |
| **E5** | rede final "dummy" | — | não feito | PD-03 |
| **E6** | hiperparâmetros (batch 2–4, `train()`, clip, cosseno, lr 1e-4) | aplicado | ✅ | — |
| **E7** | treinar em 512² ou 256² | exp. 1 em 256², exp. 2 em 1000² (só a resolução de trabalho) | ⚠️ o ScribblePrompt nunca saiu de 128² nos treinos; o oráculo mediu 256² e 512² | PD-05 |
| **E8** | reprodutibilidade (checkpoint + config + grad-norm) | parcial nos exp. 1–2 | aberto | PD-10 |

---

## 3. Cada experimento ✅

Configuração comum a todos (detalhes em [05-treinamento.md §3](05-treinamento.md)):
- MarkerUNet ResNet34 em 256²; ScribblePrompt congelado em 128²;
- Adam lr 1e-4, cosseno, clip 1,0, batch 4, seed 42;
- treino = 30 imagens oficiais; validação = 14 oficiais de teste; aumentação estática.

### Exp. 1 — primeiro run depois das correções
- **Git:** criado em 13/08 (`a0bb5c4`/`a0bd661`, "add experiment_1"), ajustado em 14/08 (`096dad6`), movido para `experiments/` em
  16/08 (`0c10713`). Run de 14/08.
- **O que mudou:** as correções da §2.3. Os dados vêm do `MonusegDataset` bruto, **reduzido para 256²** (bilinear na imagem,
  nearest no GT), e o Cellpose roda na imagem já reduzida. Scribbles `sharpened` (T = 10).
- **Loss:** Dice + RMSE + Size (0,05) + DMap (0,1), 50 épocas. O Size era o simétrico (P10), aplicado à segmentação ❓ (a data é
  anterior à reversão).
- **Resultado:** treino 1,4005 → 0,7876; validação 1,4154 → 1,1381 (melhor 1,1369, época 34). O clip atuou em 369 de 400 passos,
  ou seja, **o gradiente estava vivo** e o P1 não se repetiu. Checkpoint salvo. Sem métricas binárias.

### Exp. 2 — o mesmo, em 1000²
- **Git:** `0c10713` (16/08). Run de 15/08.
- **O que mudou:** só `WORK_SIZE = 1000`, e o Cellpose passou a rodar em 1000². O markdown chama isso de "P7", o que contradiz a
  correção P7 (que era *reduzir* para 256²).
- **Resultado:** validação 1,3934 → 1,1110 (melhor 1,1081, época 37), um pouco melhor que o exp. 1. O clip atuou em 362 de 400
  passos. Checkpoint salvo.

### Exp. 3 — dados do disco, + TV + Border
- **Git:** `0c10713` (16/08; nessa versão as saídas eram cópia das do exp. 2), `bfa3a2a` (16/08) e `4c4db98` (30/08, "adjust loss
  configs"). Run da versão atual: 23/08.
- **O que mudou:** passa a ler `data_source/MoNuSegPreprocessed/` (1000², pré-processado em 16/08) e acrescenta TV e Border.
  - Versão `bfa3a2a`: DMap 0,1, TV 0,01, Border 0,01. Validação melhor: 1,2742.
  - Versão atual: Size 0,05, **DMap 0,3, TV 0,001, Border 0,5**.
- **Resultado (versão atual):** validação 1,4907 → 1,1844 (melhor 1,1779, época 32). ⚠️ O **TV subiu** durante o treino
  (0,012 → 0,045): os marcadores ficaram menos suaves. O clip atuou só em 52 de 400 passos. Checkpoint comentado; markdown
  desatualizado. ❓ O run (23/08) é anterior à reversão do Size (30/08), então provavelmente usou a versão simétrica.

### Exp. 4 — o melhor resultado
- **Git:** `ef7d731` (22/09). Run de 04/09.
- **O que mudou:** tira TV e Border, reduz o Size para 0,02 e o DMap para 0,2; passa a clonar a branch `homolog`. Primeiras
  **métricas binárias**. O Size (agora `peso·ratio`) passa a olhar os marcadores: o valor inicial 0,049 = 0,02 × 2,45 é compatível.
- **Resultado:** validação 1,6752 → 1,0115 (melhor na época 48).
  - **Dice 0,648 · precisão 0,589 · recall 0,735 · IoU 0,481 · massa 1,28× o GT**.
  - Os marcadores cobrem 20% da imagem, perto do tamanho do GT (~22%), bem maiores que um "miolo".
- ⚠️ A célula 24 dá `NameError` (`epochs`). As células 25–32 têm saídas de **outro run** (Dice 0,480, "melhor val 0,8356 época 7"),
  então as figuras do exp. 4 não são confiáveis (PD-10).

### Exp. 5 — ablação `dense_soft`, 500 épocas
- **Git:** `ef7d731`. Run de 11/09. Kernel reaproveitado (contadores de execução de 36 a 55).
- **O que mudou:** `scribble_mode="dense_soft"` (`[m, 1−m]`) e 500 épocas. `scribble_temperature=5.0` foi declarada, mas é
  **ignorada** nesse modo; o markdown chama o modo de "quase binário", o que está errado.
- **Resultado:** **Dice 0,598 · precisão 0,495 · recall 0,783 · IoU 0,428 · massa 1,65×**: segmenta demais. No treino, o termo de
  Dice termina em 0,149 (soft Dice ≈ 0,85): **overfitting** claro (PD-38). A validação chegou a 1,0412, mas na época 50 estava em
  1,0986, pior que o exp. 4 na mesma época.

### Exp. 6 — `sharpened` + TV + Border, 200 épocas
- **Git:** `ef7d731`. Run de 12/09.
- **O que mudou:** volta ao `sharpened`, com 200 épocas, e acrescenta TV (0,001) e Border (0,5) à loss do exp. 4. Nova célula de
  diagnóstico dos marcadores.
- **Resultado:** **Dice 0,578 · precisão 0,638 · recall 0,544 · IoU 0,408 · massa 0,88×**; os marcadores cobrem 13% da imagem.
  Diagnóstico: marcador médio 0,446 dentro do GT e 0,111 fora. ⚠️ Esse diagnóstico foi calculado num batch de **treino**, não
  de validação.

---

## 4. O que os seis experimentos ensinam

1. **O pipeline aprende, mas não passa da própria entrada.** Desde o exp. 1 o gradiente existe e a loss cai, mas o melhor Dice
   (0,648) fica 16 pontos abaixo do Cellpose. O oráculo diz por quê: com o ScribblePrompt, só marcadores do tamanho do núcleo
   dão Dice alto (PD-40).
2. **Os melhores resultados vêm de marcadores grandes.** O exp. 4 (marcadores = 20% da imagem, ~ o GT) é o melhor. O exp. 6
   (13%, com mais regularização) cai. Coerente com o conflito entre Size/DMap/Border e o Dice ([04-losses.md §4](04-losses.md)).
3. **Mais épocas pioram** (exp. 5: overfitting). Com 30 imagens e 24 M de parâmetros, a generalização é o gargalo (PD-38, PD-46).
4. **`dense_soft` é pior que `sharpened`**, mas o oráculo sugere que o problema de fundo é o **canal negativo denso**, comum aos
   dois modos (PD-04).
5. **Os experimentos mudaram várias coisas de cada vez** (loss, épocas, modo de scribble), e a validação foi feita no teste. Não
   dá para atribuir com segurança o efeito de cada mudança; é isso que a PD-44 corrige.

---

## 5. Próximos experimentos planejados

Todos já decididos; ordem sugerida (detalhes nas pendências):

| # | Experimento | Pendência | Custo estimado |
|---|---|---|---|
| 0 | Infraestrutura: módulo em `src/` (dataset do disco, aumentação por época, k-fold, melhor modelo, early stopping) | PD-19, PD-08, PD-46, PD-02 | código |
| 1 | **Cellpose com o diâmetro real** (e por imagem nas três em 20×), antes de regerar os dados | PD-30, PD-24 | minutos (sem treino) |
| 2 | **Mapa de distância por núcleo** (regerar `MoNuSegPreprocessed/`) | PD-06 | minutos |
| 3 | **Experimento-base:** só Dice + TV, ScribblePrompt em 256², só positivos, Cellpose como linha de base | PD-44 | ~7 min por configuração (5 *folds*) |
| 4 | Reintroduzir DMap (mapa corrigido), depois Size, um de cada vez | PD-44 | idem |
| 5 | Ablação de normalização ImageNet + encoder congelado (5 variantes) | PD-36, PD-38 | ~35 min |

**Decisão (2026-09-27):** seguir com os passos 0 a 5 antes de decidir o rumo da PD-40. A decisão é do autor. O oráculo passa a ser
**material de publicação**: ele e os resultados dos passos 3 a 5 formam o conjunto de evidências do trabalho (PD-47).

---

## 6. Decisões (2026-09-27)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | Ordem do §5 (infraestrutura e dados antes do experimento-base) | Sim | §5 |
| 2 | Levar o oráculo ao orientador antes ou depois | **Rodar os novos experimentos primeiro**; a decisão de rumo é do autor. O trabalho deve ser **publicado**, e o oráculo entra como **resultado** do trabalho | PD-40, PD-47 |
| 3 | O que fazer com os notebooks antigos | Depois das correções, **remover o que não fizer sentido**. O foco é manter o material que embasa o trabalho; o histórico fica neste documento e no git | PD-10 |

### Perguntas originais (histórico)

1. **Ordem do §5:** faz sentido começar pela infraestrutura (módulo + k-fold) e pelos dois ajustes de dados (diâmetro do Cellpose e
   mapa de distância) antes do experimento-base? SIm 
2. **Conversa com o orientador (PD-40):** prefere levar o resultado do oráculo antes de rodar os próximos experimentos, ou rodar o
   experimento-base primeiro e levar os dois resultados juntos? Vamos testar os novos experimentos, em suma eu que vou tomar a decisão. 
   Pensando que este trabalho possui o objetivo de ser publicado, a analise do oraculo pode ser uma boa para termos mais resultados de testes
3. **Notebooks antigos:** os exp. 1–6 ficam como estão (histórico), ou prefere corrigir os markdowns desatualizados (Size
   "simétrico", `dense_soft` "quase binário") e marcar as células do exp. 4 com saída de outro run? Vamos remover oque não fizer sentido depois das correções. o foco é ter oque faz sentido e entrega material que vai ajudar a embasar o trabalho 
   
