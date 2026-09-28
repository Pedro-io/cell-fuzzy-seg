# 00 — Visão geral do TCC

> Documento de estudo pessoal. Legenda de origem de cada afirmação:
> ✅ verificado no código · 📜 histórico de commits / docs antigos · 📖 tese ou artigo · ❓ inferência minha, **precisa da sua confirmação**.
>
> Última verificação contra o código: 2026-09-27 (branch `homolog`, commit `f9b1e9b`).

---

## 1. Em uma frase

Treinar uma rede (**MarkerUNet**) que, a partir da imagem histológica e de uma segmentação bruta feita pelo **Cellpose**,
gera **marcadores difusos** que, entregues a uma rede de segmentação interativa congelada (**ScribblePrompt**), produzem uma
segmentação de núcleos melhor. A MarkerUNet é treinada **através** do ScribblePrompt, sem precisar de "marcadores
corretos" como gabarito. ✅📖

---

## 2. O problema de onde tudo parte

### 2.1 Segmentação por marcadores 📖

Na segmentação interativa, o usuário indica **onde está o objeto** com *marcadores* (ou *seeds*): cliques, rabiscos
(*scribbles*), caixas. Um algoritmo propaga essa informação e devolve a segmentação. Isso é preciso, mas **custa esforço
humano**, e o custo explode em imagens com centenas de objetos, como núcleos em histologia (tese, seção 1.1 e fig. 1.1).

### 2.2 Por que não dá para aprender marcadores de forma supervisionada 📖

Não existe "o marcador correto" de uma imagem: pessoas diferentes marcariam de jeitos diferentes, e vários marcadores
distintos recuperam a mesma segmentação boa. Por isso **não existem datasets com gabarito de marcador**, e aprender
marcadores não pode ser um problema supervisionado comum (tese, 1.1 e questão 2 da 1.2).

### 2.3 A solução da tese do seu orientador 📖

Gabriel Barbosa da Fonseca, *Hierarchy-Based Fuzzy Segmentation and Marker Learning* (PUC Minas / Univ. Gustave Eiffel,
2026). O truque é **não supervisionar o marcador diretamente**, e sim o **resultado** que ele produz:

```
imagem ──► rede de marcadores f(X;θ) ──► marcador difuso μ ∈ [0,1] ──► camada de segmentação S(μ) ──► segmentação
                                              │                                                      │
                                              └──► regularizações L_i(μ)            L_seg(S(μ), GT) ◄─┘
```

A loss total (tese, eq. 5.1) é:

```
L_marker = L_seg( S(f(X;θ)), GT )  +  Σ_i λ_i · L_i( f(X;θ) )
```

- **L_seg** garante a 1ª propriedade de um bom marcador: *recuperar a segmentação correta*.
- **L_i** garantem a 2ª propriedade: *ser fácil de editar por um humano*. Por isso ele deve ser pequeno (L_size), longe
  da borda do objeto (L_Dmap), ter poucos componentes conexos (L_comp) e ser suave (L_TV) (tese, 5.3).

Para o gradiente de L_seg chegar à rede, **S precisa ser diferenciável em relação a μ**. A contribuição central da tese é
justamente o **FMBS** (*Fuzzy Marker-Based Segmentation*), uma versão diferenciável da segmentação por marcadores sobre
hierarquias (árvore binária de partições). O "fuzzy" do nome do repositório vem daqui: **os marcadores são difusos**
(valores contínuos em [0,1]), não binários. ✅ Não há outra lógica fuzzy no código.

### 2.4 Onde o TCC se encaixa na tese

**A pergunta do TCC:** o mesmo tipo de pipeline da tese (segmentação intermediária → rede de marcadores → camada de
segmentação, treinado de ponta a ponta) consegue bons resultados **sem o FMBS**, usando outra camada de segmentação no
lugar? E, sem ele, os resultados podem ser melhores? (sua resposta, 2026-09-27)

**Sugestão para discutir com o orientador** (vocês ainda não falaram disso): a tese trabalha com imagens naturais
(MSRA10K, MS COCO) e precisa de uma *cue* do usuário para saber qual objeto marcar. Nos trabalhos futuros, a seção
**7.1.2 ("Application to task-specific segmentation")** propõe aplicar o método quando o objeto de interesse já é conhecido,
citando **tipos de células em microscopia**. Nesse caso a cue do usuário é desnecessária, exatamente como no TCC, em que o
Cellpose faz esse papel. 📖 Pode ser um bom gancho para o texto do TCC.

---

## 3. O que o TCC muda em relação à tese

A arquitetura mais próxima na tese é o **modelo de marcador de objeto** (seção 5.4.2, fig. 5.8): uma rede de segmentação
gera uma segmentação intermediária, e a rede de marcadores transforma essa segmentação em marcador.

| Componente | Tese | TCC | Origem |
|---|---|---|---|
| Segmentação intermediária | U-Net interativa treinada, entrada = imagem + cues | **Cellpose** (modelo `cpsam`, congelado, **pré-computado uma vez**) | ✅ [cellpose_step.py:28](../../src/pipeline/steps/preprocessing/cellpose_step.py#L28) |
| Rede de marcadores | U-Net com **3 encoders** ResNet34 (imagem; seg. objeto+fundo; saliency map) | **MarkerUNet**: U-Net `smp` com **1 encoder** ResNet34 ImageNet e entrada de **4 canais** (RGB + máscara Cellpose), ou seja, fusão precoce, como a prova de conceito da tese (5.4.1) | ✅ [marker_unet_config.py](../../src/models/configs/marker_unet_config.py) |
| Camada de segmentação | **FMBS** sobre hierarquia (sem pesos aprendíveis) | **ScribblePrompt-UNet congelado** (rede neural de segmentação interativa biomédica, entrada 128×128) | ✅ [scribble_prompting_network.py](../../src/models/networks/final_segmentation/scribble_prompting_network.py) |
| Marcador de fundo | fixo (4 cliques nos cantos) ou aprendido por inversão (5.4.3) | **complemento do marcador de objeto**: canal negativo = `sigmoid(T·(0.5−s))`, denso na imagem toda. ⚠️ Você não lembra o motivo e acha que **provavelmente foi um erro**: [PD-04](08-pendencias.md#pd-04) | ✅ [scribble_prompting_network.py:348-352](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L348-L352) |
| Regularizações | L_size, L_Dmap, L_comp, L_TV | L_size, L_Dmap e L_TV implementados. **L_comp removido de propósito** (§6 e §8). Extras que não estão na tese: Border, NotTooThin e RMSE, **sugeridos pelo orientador** | ✅ [terms.py](../../src/losses/terms.py) |
| Dados | MSRA10K / MS COCO + 4 datasets não vistos | **MoNuSeg**: 30 imagens de treino e 14 de teste, 1000×1000 | ✅ [meta.json](../../data_source/MoNuSegPreprocessed/meta.json) |

**Por que tirar o FMBS?** Porque essa é a própria pergunta do TCC (§2.4): ver se o pipeline funciona, e talvez funcione
melhor, sem ele. A camada de segmentação passou por três versões. O **watershed** e uma **U-Net congelada** deram
resultados muito ruins e foram removidos. O **ScribblePrompt** é a versão atual (sua resposta; datas na §6). Se ele também
não der resultado, outras redes podem ser testadas, e o código foi feito para isso: a rede final é trocável via
[base_final_segmentation.py](../../src/models/networks/final_segmentation/base_final_segmentation.py). ✅

📖 Argumento a favor do ScribblePrompt para o texto: ele é uma rede de segmentação interativa feita para imagens
**biomédicas**, e o treino dele inclui datasets de células (artigo, seção 3.3).

### 3.1 Uma diferença conceitual importante 📖✅❓

No FMBS, a segmentação é **inteiramente determinada** pelos marcadores e pela hierarquia. É isso que torna o marcador uma
"explicação" editável da segmentação (tese, 1.3). O ScribblePrompt é uma rede neural que usa a imagem **e** os scribbles.
Ela pode segmentar coisas que o marcador não "explica", e foi treinada com **traços esparsos 0/1**, não com mapas densos.
❓ Com isso a propriedade "marcador editável e interpretável" da tese fica mais fraca, e vale discutir isso no texto do TCC.
O código já tenta aproximar a entrada da distribuição de treino do ScribblePrompt com o modo `sharpened`
([scribble_prompting_network.py:55-60](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L55-L60)).

---

## 4. Fluxo de ponta a ponta ✅

### Fase 1: pré-processamento (roda uma vez, no Colab com GPU)

Notebook: `notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb`.

```
data_source/MoNuSegTrainingData/{Tissue_Images/*.tif, Annotations/*.xml}
        │  MonusegDataset: lê o .tif; desenha os polígonos do XML numa máscara BINÁRIA (cv2.fillPoly, valor 1)
        ▼
{id, image (1000×1000×3), ground_truth (1000×1000, 0/1)}
        │  CellposeStep  — cpsam, diâmetro 30, flow_threshold 0.2, cellprob 0.0, min_size 4
        ▼  + segmentation  (máscara de INSTÂNCIAS do Cellpose: 0 = fundo, 1..K = núcleos)
        │  RGBAStep      — RGB normalizado em [0,1] + alpha = (segmentation > 0)
        ▼  + rgba (1000×1000×4, float32)
        │  DistanceMapStep — 1 − EDT(gt>0) / max(EDT), calculado sobre o GT
        ▼  + distance_map (1000×1000, float32; ~0 no centro dos núcleos, 1 na borda e no fundo)
        │  SaveResultsStep
        ▼
data_source/MoNuSegPreprocessed/{train,test}/<chave>/<id>.npy   (+ meta.json)
```

### Fase 2: treino (um notebook por experimento)

```
dados persistidos (.npy)
        │
        ▼  MarkerStep (differentiable=True)
rgba 1000² ──resize──► 256² ──► MarkerUNet ──► sigmoid ──resize──► markers 1000² ∈ [0,1]
        │
        ▼  FrozenSegmentationStep → ScribblePromptingNetwork (pesos congelados, mas sem no_grad)
image → tons de cinza ; markers → [pos, neg] (sharpened, T=10) ; tudo → 128² ; UNet ; → 1000² ; sigmoid
        ▼
segmentation 1000² ∈ [0,1]          ⚠️ essa chave SOBRESCREVE a máscara do Cellpose da fase 1
        │
        ▼  LossComposer (termos escolhidos no notebook)
Dice(segmentation, gt) + Size(markers) + DMap(markers, distance_map) + TV(markers) + ...
        │
        ▼  Trainer: backward → clip de gradiente (opcional) → optimizer.step → scheduler.step
```

A resolução passa por **1000 → 256 → 1000 → 128 → 1000**. O gargalo de detalhe é o ScribblePrompt, que trabalha em
128×128: cada pixel dele equivale a ~7,8 px da imagem original, e um núcleo do MoNuSeg tem poucas dezenas de pixels de
diâmetro. ✅ (os tamanhos estão no código) · ❓ (o impacto disso nos resultados é inferência minha)

---

## 5. Mapa do repositório

| Pasta / arquivo | Papel | Documento de estudo |
|---|---|---|
| [configs/datasets.yml](../../configs/datasets.yml) | caminhos do MoNuSeg. ⚠️ `loader_config` e `preprocessing` não são usados pelo treino atual | 01 |
| [src/data/load/](../../src/data/load/) | datasets (`BaseDataset`, `MonusegDataset`, `MonusegPreprocessedDataset`) | 01 |
| [src/pipeline/](../../src/pipeline/) | pipelines e Steps (pré-processamento, inferência, persistência) | 02 |
| [src/io/](../../src/io/), [src/utils/](../../src/utils/) | gravação em disco, logger, conversões de imagem | 02 |
| [src/models/](../../src/models/) | MarkerUNet e ScribblePromptingNetwork | 03 |
| [src/losses/](../../src/losses/) | termos de perda e `LossComposer` | 04 |
| [src/training/](../../src/training/) | `Trainer`, `TrainingLoop`, `GradNormCallback` | 05 |
| [notebooks/experiments/](../../notebooks/experiments/) | experimentos 1 a 6 | 06 |
| [tests/](../../tests/) | 74 testes pytest | 07 |
| [data_source/](../../data_source/) | MoNuSeg bruto e pré-processado (versionado no git) | 01 |
| [docs/papers/](../papers/) | PDFs do Cellpose (2020) e do ScribblePrompt | — |
| [pipeline_test.py](../../pipeline_test.py) | ⚠️ script **quebrado** (importa módulos que não existem mais) | 08 |

---

## 6. Linha do tempo (dos commits) 📜

Serve para lembrar **em que ordem** as decisões foram tomadas. Os "porquês" que faltam estão na §8.

| Data | O que aconteceu |
|---|---|
| 15–26/04 | Primeiro commit; notebooks de teste do Cellpose; MoNuSeg adicionado; visualização GT × máscara gerada |
| 28–30/04 | Estrutura `src/`; pasta `data` → `data_source`; `datasets.yml`; classes base; máscaras a partir do XML |
| 03/05 | `CellposeStep`, `RGBAStep`, `OutputWriter`, primeiro `ModelPipeline`; entra `segmentation-models-pytorch` |
| 04/05 | Funções de perda com padrão Strategy, **incluindo uma perda topológica com component trees (Higra)**, que é o L_comp da tese |
| 11–17/05 | Primeira MarkerNet; notebook-template de experimento |
| **19/05** | Commit "Refactor code structure…" **remove a pasta `src/losses/topology/`** (L_comp): complexa demais e ligada ao FMBS (sua resposta; ver §8) |
| 02/06 | Rede final = **watershed por marcadores** (`SegmentationStep`); resultados muito ruins |
| 09/07 | Watershed trocado por uma **U-Net congelada** (`FrozenUnetSegmentationStep`), também com resultados muito ruins; PR #1 (`feat/marker-net`) |
| 15/07 | Cellpose passa a usar o modelo `cpsam_v2` (hoje o código usa `cpsam`) |
| 22/07 | Steps reorganizados em `preprocessing/` e `inference/`; `MonusegPreprocessedDataset` |
| **01/08** | **ScribblePrompt vira a rede final** |
| 10/08 | `DistanceMapStep` (o mapa de distância sai do notebook e vai para o pré-processamento) |
| 13–16/08 | `experiment_1`; pré-processamento persistido em disco |
| 30/08–03/09 | Ajustes das losses; PR #2 (`feat/experiment_1`), com as correções da investigação do exp. 1 |
| 22/09 | `MarkerStep`/`Trainer` passam a gerenciar `train()`/`eval()`; novos notebooks de experimento |
| 27/09 | Peso configurável nas losses Dice e RMSE (último commit) |

**Evolução da rede final:** watershed → U-Net congelada → ScribblePrompt. 📜

---

## 7. Estado atual dos experimentos

Resumo. O detalhamento de cada experimento vai para o documento 06. Todos os seis rodaram até o fim, sem NaN e com a
loss caindo. ✅ (saídas salvas nos notebooks)

| Exp. | O que mudou | Dice | IoU |
|---|---|---|---|
| 1 | 256², Dice + RMSE + Size + DMap, 50 épocas | — | — |
| 2 | igual ao 1, em 1000² | — | — |
| 3 | dados do disco; + TV + Border | — | — |
| **4** | Size(0,02) + DMap(0,2), sem TV/Border | **0,648** | **0,481** |
| 5 | `dense_soft`, 500 épocas | 0,598 | 0,428 |
| 6 | `sharpened` + TV + Border, 200 épocas | 0,578 | 0,408 |
| **Cellpose sozinho** | a máscara que **entra** na MarkerUNet | **0,810** | **0,682** |

Métricas por pixel, média das 14 imagens de teste, limiar 0,5. A linha do Cellpose não está nos notebooks: foi calculada
nesta revisão direto dos `.npy` de `data_source/MoNuSegPreprocessed/test/`. ✅

**Leitura principal: hoje o pipeline piora a segmentação que recebe.** O melhor experimento fica ~16 pontos de Dice abaixo
do próprio Cellpose. Enquanto isso não se inverter, o TCC não mostra ganho. ❓ Não sei a causa. As suspeitas que já
estão documentadas são o ScribblePrompt como camada de loss (C1 da investigação), a resolução de 128² e a entrada fora de
distribuição. **Atualização (tema 3):** a resolução sozinha não explica, porque uma predição perfeita em 128² ainda daria
Dice 0,88–0,92 (PD-05). O canal negativo denso sobre os núcleos (PD-04) e o mapa de distância por imagem (PD-06) ficaram mais
suspeitos. Ver [08-pendencias.md](08-pendencias.md).

**Oráculo (2026-09-27, [PD-40](08-pendencias.md#pd-40)): o achado mais importante até agora para o rumo do TCC.**
Para saber até onde a rede final consegue chegar, o ScribblePrompt recebeu **marcadores perfeitos**, tirados do próprio GT
(algo que nenhuma MarkerUNet treinada alcança):

| Marcador perfeito dado ao ScribblePrompt | Melhor Dice | Comparado com o Cellpose sozinho (≈0,80) |
|---|---|---|
| o núcleo inteiro | 0,93 | melhor |
| o miolo do núcleo (1/3 da área) | 0,78–0,80 | empata, sem passar |
| um ponto no centro de cada núcleo | 0,57–0,61 | bem pior |

- **Hoje, o Cellpose sozinho é melhor** do que o ScribblePrompt alimentado com marcadores pequenos, mesmo perfeitos. A ideia da
  tese (marcador pequeno que a camada de segmentação expande até o núcleo) não se realiza com o ScribblePrompt neste dado: ele
  expande pouco. O problema, portanto, não é só o treino da MarkerUNet; **a rede final tem um teto baixo**.
- **Usar 256×256 na entrada do ScribblePrompt** (hoje 128×128) melhorou todos os casos, mas não resolve o ponto anterior.

Explicação completa, em linguagem simples: [03-modelos.md §8.1](03-modelos.md).

Ressalvas que afetam a comparação entre experimentos (✅ verificadas no código ou nos notebooks):
- O conjunto de **teste oficial** é usado como validação **e** para escolher a configuração. Na prática, não há teste cego.
- A aumentação de dados é feita **uma vez**, antes do loop: todas as épocas veem os mesmos 8 batches, na mesma ordem.
- Os runs (14/08 a 12/09) são anteriores ao commit de 22/09 que coloca a MarkerUNet em `eval()` na validação. ❓ Se o
  Colab usou o código da época, a validação rodou com a BatchNorm em modo treino.
- O `GradNormCallback` mede a norma **depois** do clipping; o valor 1,0 só indica que o clip atuou.
- A **correção P10** da investigação (Size simétrico `|ratio−1|` sobre a segmentação) foi **revertida** em 30/08 (`4c4db98`)
  para `peso·Σmarcador/ΣGT` sobre os **marcadores**, para corrigir a loss e alinhá-la à tese (sua resposta). Isso coincide
  com o L_size da tese (eq. 5.5). Os testes em [test_object_size_loss.py](../../tests/test_object_size_loss.py) ainda
  esperam a versão simétrica e devem falhar (pendência P-B).
- Nenhum checkpoint foi salvo nos exp. 3 a 6. No exp. 4, as figuras finais são de outro run.
- A investigação do exp. 1 recomenda experimentos de isolamento (E2: sensibilidade do ScribblePrompt aos scribbles; E3:
  "oráculo" com marcador = GT), que **nunca foram feitos**. O E3 responderia direto se o ScribblePrompt consegue superar o
  Cellpose quando recebe o marcador ideal.

---

## 8. Decisões confirmadas e pendências

### 8.1 Respostas incorporadas (2026-09-27)

| # | Pergunta | Resposta | Observação |
|---|---|---|---|
| 1 | Encaixe na tese | Ainda não conversaram. A diferença principal para a tese é **não usar o FMBS** | §2.4: a seção 7.1.2 fica como sugestão para discutir com o orientador |
| 2 | Por que tirar o FMBS | É a pergunta do TCC: dá para ter resultados bons, ou melhores, sem ele? Watershed e U-Net congelada deram resultados muito ruins | — |
| 3 | Por que o L_comp saiu | Sem o FMBS ele não faz sentido e deixava tudo mais complexo | 📖 Atenção ao argumento: na tese o L_comp é calculado sobre a **max-tree do próprio marcador** (seção 5.3), não sobre a hierarquia do FMBS. Tecnicamente ele funcionaria sem o FMBS. O argumento mais sólido para a banca: o objetivo do L_comp é deixar o marcador **editável por um humano** (poucos componentes), e no TCC o marcador vai para uma rede, não para um usuário. Somado ao custo de complexidade, isso justifica a remoção. |
| 4 | Modelo do Cellpose | `cpsam` | ✅ Confirmado no código ([cellpose_step.py:28](../../src/pipeline/steps/preprocessing/cellpose_step.py#L28)) e nos logs dos notebooks. Falta a referência certa (pendência P-D) |
| 5 | Marcador de fundo = complemento | Não lembra o motivo; provavelmente foi um erro | Pendência P-A |
| 6 | Métrica | Semântica (Dice/IoU), por sugestão do coordenador | — |
| 7 | Border, NotTooThin, RMSE | Vieram do orientador | — |
| 8 | Reversão do P10 | Foi para corrigir a loss conforme a tese | 📖 **Sim, é o que a tese usa.** Eq. 5.5: `L_size = Σ μ_O(x) / Σ Ŝ_O(x)`, calculado **sobre o marcador**. É mínimo quando o marcador é pequeno; não é "igualar a massa ao GT". Única diferença: a tese calcula por objeto/imagem e o código soma o batch inteiro. Os testes ficaram desatualizados (P-B) |
| 9 | Baseline e validação separada | Medo de perder imagens de treino | Proposta na pendência P-C |
| 10 | PDFs no git | Colocar no `.gitignore` | ✅ Feito |

### 8.2 Pendências abertas

> **Movidas para o registro central [08-pendencias.md](08-pendencias.md).** Correspondência: P-A → PD-04 · P-B → PD-11 ·
> P-C → PD-02 · P-D → PD-22 · P-E → PD-15. O texto abaixo fica só como histórico; a versão atualizada é a do 08.

- **P-A — Marcador de fundo.** Hoje o scribble negativo é o complemento denso do marcador de objeto
  ([scribble_prompting_network.py:348-352](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L348-L352)).
  O ScribblePrompt foi treinado com negativos esparsos, e na tese o fundo é outro marcador. Alternativas para testar:
  (a) canal negativo zerado; (b) negativo fixo e esparso, como os cantos da tese ou regiões longe de qualquer núcleo do
  Cellpose; (c) negativo aprendido a partir do complemento da máscara do Cellpose, como a inversão da tese (5.4.3). O
  modo atual deve virar só uma das opções de uma ablação.
- **P-B — Testes do Size.** Atualizar [test_object_size_loss.py](../../tests/test_object_size_loss.py) para a fórmula da
  tese e trocar a docstring de `ObjectSizeLoss` ("penaliza o desvio de tamanho…"), que também ficou errada.
- **P-C — Validação sem perder imagens de treino.** Proposta:
  1. Usar **validação cruzada k-fold** nas 30 imagens de treino (por exemplo, 5 folds de 24 treino / 6 validação) para
     comparar configurações.
  2. Escolhida a configuração, treinar com as **30** e avaliar **uma única vez** nas 14 de teste.

  Assim nenhuma imagem sai do treino final; o custo é rodar cada configuração k vezes. Em qualquer caso, colocar o
  **Cellpose sozinho como linha de base** em todo notebook (não custa imagem nenhuma).
- **P-D — Referência do Cellpose.** Baixar o artigo do Cellpose-SAM para `docs/papers/` e citá-lo. O PDF atual (2020)
  continua útil para explicar a ideia dos *flows*, mas não descreve o modelo usado.
- **P-E — Checkpoint do ScribblePrompt sem hash.** Ver a nota de segurança no CLAUDE.md.

### 8.3 Perguntas originais (histórico)

1. **Encaixe na tese.** O TCC é a aplicação da seção 7.1.2 da tese (segmentação de tarefa específica, sem cue)? Ou o
   orientador enxerga de outro jeito? não chegamos a falar sobre isso, a diferença maior do meu pipeline para o dele é que eu não uso a fmbs
2. **Por que ScribblePrompt no lugar do FMBS?** Foi por falta de implementação pública do FMBS, por o ScribblePrompt ser
   biomédico, ou por outro motivo? E por que o watershed e a U-Net congelada foram abandonados? A proposta era tentar conseguir bons resultados com o mesmo tipo de pipeline e remover a fmbs para ver se sem ela podemos ter resultados melhores, eu testei vom o watershed e a unet, pelos resultados muito ruins eu preferi remover 
3. **L_comp.** Por que a perda topológica (component trees / Higra) foi removida em 19/05? Pretende voltar com ela? removemos pq n faz sentido aqui, faria com a fmbs, além de deixar mais complexo
4. **Cellpose.** A referência certa é o Cellpose-SAM (Pachitariu, Rariden, Stringer, 2025, o modelo `cpsam` do Cellpose 4)?
   O PDF em `docs/papers/cellpose.pdf` é o Cellpose original (2020). acredito que usamos a cpsam como padrão 
5. **Marcador de fundo.** Usar o complemento do marcador de objeto como scribble negativo foi uma decisão consciente?
   Na tese o fundo é fixo (cantos) ou aprendido por inversão. Não me lembro pq fiz isso, provavelmente foi um erro
6. **Métrica de avaliação.** O MoNuSeg costuma ser avaliado por instância (AJI, PQ), mas o pipeline produz uma máscara
   **binária**. A avaliação do TCC vai ser semântica (Dice/IoU) mesmo? acredito que sim, foi oque meu cordenador sujeriu
7. **Referências extras.** As losses Border, NotTooThin e RMSE não estão na tese. Vieram de algum código do orientador
   ou de outro artigo? foi do orientador mesmo
8. **Reversão do P10.** A volta do `SizeTerm` para `Σmarcador/ΣGT` (30/08) foi para alinhar com a tese? Se foi, os
   testes de `test_object_size_loss.py` precisam ser atualizados. foi para tentar corrigir a loss, isso é oque foi aplicado na tese, correto? 
9. **Baseline.** Faz sentido colocar o Cellpose sozinho como linha de base fixa em todos os notebooks e separar uma
   validação dentro das 30 imagens de treino, deixando as 14 de teste só para o resultado final? meu medo aqui é perder muitas imagens para o treino
10. **PDFs no git.** `docs/papers/` ainda não está versionado (24 MB, e o Cellpose é CC-BY-NC). Prefere versionar ou
   colocar no `.gitignore`? vamos deixar no gitignore 
