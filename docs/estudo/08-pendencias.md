# 08 — Pendências (registro central)

> Lugar único para tudo o que está errado, incompleto, arriscado ou em aberto. Os outros documentos citam os itens pelo ID
> (`PD-nn`) e não repetem o detalhe.
>
> **Como usar:** todo problema novo entra aqui com o próximo número livre. Quando for resolvido, muda para ✅ **Resolvida**,
> com data e commit, e **não é apagado** (o histórico serve para o texto do TCC). IDs nunca são reaproveitados.
>
> **Severidade:** 🔴 crítica (invalida ou bloqueia resultados) · 🟠 alta (afeta resultados ou conclusões) ·
> 🟡 média (reprodutibilidade, risco latente) · ⚪ baixa (higiene, documentação).
>
> **Origem:** ✅ verificado · 📖 tese/artigo · ❓ hipótese a confirmar.

Última atualização: 2026-09-29 (PD-23/PD-26: novo download; PD-25/PD-07: medição da etapa 1).

---

## Índice

| ID | Sev. | Área | Resumo | Status |
|---|---|---|---|---|
| [PD-01](#pd-01) | 🔴 | Resultados | O pipeline é pior que o Cellpose sozinho (Dice 0,648 × 0,810) | Aberta |
| [PD-02](#pd-02) | 🟠 | Metodologia | O teste oficial é usado como validação e para escolher configurações | Aberta (proposta: k-fold) |
| [PD-03](#pd-03) | 🟠 | Metodologia | Os experimentos de isolamento E2/E3/E5 nunca foram executados | Parcial: E3 (oráculo) executado em 2026-09-27 |
| [PD-04](#pd-04) | 🟠 | Modelo | O scribble negativo é o complemento denso do marcador (provável erro) | Decidida: só positivo (confirmado pelo oráculo) |
| [PD-05](#pd-05) | 🟡 | Resolução | O ScribblePrompt trabalha em 128²: o núcleo mediano vira ~3 px | Aberta; o oráculo recomenda 256² |
| [PD-06](#pd-06) | 🟠 | Dados/Loss | O mapa de distância é normalizado pelo máximo da imagem, não por núcleo | Aberta |
| [PD-07](#pd-07) | 🟠 | Dados | O GT binário funde núcleos (−25% de componentes no treino) | Parcial: máscara por instância gerada (2026-09-29); falta usar e persistir |
| [PD-08](#pd-08) | 🟡 | Treino | A aumentação é estática e não há shuffle | Decidida: módulo em `src/` com aumentação por época |
| [PD-09](#pd-09) | 🟡 | Treino | Os runs provavelmente validaram com a BatchNorm em `train()` | Aberta |
| [PD-10](#pd-10) | 🟡 | Reprodutibilidade | Checkpoints não salvos, saídas velhas, markdowns desatualizados | Aberta |
| [PD-11](#pd-11) | ⚪ | Testes | Os testes e a docstring do `ObjectSizeLoss` descrevem a fórmula antiga | Aberta |
| [PD-12](#pd-12) | 🟡 | Loss | Size/DMap/TV são normalizados pela soma do batch inteiro | Aberta |
| [PD-13](#pd-13) | 🟡 | Loss | Desbalanceamento de classes não tratado (P8) | Aberta |
| [PD-14](#pd-14) | ⚪ | Diagnóstico | `GradNormCallback` mede a norma depois do clipping | Aberta |
| [PD-15](#pd-15) | 🟡 | Segurança | O checkpoint do ScribblePrompt é baixado sem verificação de hash | Aberta (hash de referência registrado) |
| [PD-16](#pd-16) | 🟡 | Robustez | Fallbacks silenciosos (GT como entrada; rede "dummy") | Parcial (2026-10-01): notebook de pré-processamento corrigido; falta o `MarkerStep` |
| [PD-17](#pd-17) | ⚪ | Código morto | `pipeline_test.py` está quebrado | Aberta |
| [PD-18](#pd-18) | ⚪ | Config | Partes de `datasets.yml` não são usadas | Aberta |
| [PD-19](#pd-19) | 🟡 | Dados | Dois caminhos de carregamento; `MonusegPreprocessedDataset` não lê do disco | Decidida: módulo em `src/` |
| [PD-20](#pd-20) | ⚪ | Git | `.pyc` e `egg-info` versionados | Aberta |
| [PD-21](#pd-21) | 🟡 | Git | 1,6 GB de dados no git, incluindo 1,1 GB de `.npy` derivados | Adiada (decisão do autor) |
| [PD-22](#pd-22) | ⚪ | Referências | Falta o artigo do Cellpose-SAM | Aberta |
| [PD-23](#pd-23) | 🟠 | Dados | 16.966 núcleos anotados no treino × "~22.000" oficiais | Parcial (2026-09-29): faltavam 7 das 37 imagens de treino; dados reorganizados; falta regerar os `.npy` do treino |
| [PD-24](#pd-24) | 🟡 | Dados | 3 imagens de treino em 20× (0,5 µm/px): núcleos com metade do tamanho | Aberta |
| [PD-25](#pd-25) | 🟡 | Dados | Rasterização: o `fillPoly` engorda o GT em ~10% (o `int()` não é o problema); 5 polígonos com 2 vértices e área 0 | ✅ Resolvida (2026-09-29) |
| [PD-26](#pd-26) | ⚪ | Dados | `Binary_masks`/`Binary_masks_instance` não vêm do download oficial e não são usadas | ✅ Resolvida (2026-09-29): removidas |
| [PD-27](#pd-27) | 🟡 | Licença | Repositório público redistribui o MoNuSeg sem atribuição; `LICENSE` vazio | Parcial: atribuição feita; falta a licença do código |
| [PD-28](#pd-28) | ⚪ | Docs | `ARCHITECTURE.md` tem trechos errados (`registry/`, rede final "treinável") | ✅ Resolvida (2026-09-27) |
| [PD-29](#pd-29) | 🟡 | Pré-proc. | A proteção contra nome de modelo do Cellpose nunca executa (`MODEL_LIST` não existe) | ✅ Resolvida (2026-10-01): nome desconhecido → erro |
| [PD-30](#pd-30) | 🟡 | Pré-proc. | Parâmetros do Cellpose: `diam_mean` ignorado, `diameter=30` sem reescala, `flow_threshold`/`min_size` fora do padrão | Decidida: testar no Colab |
| [PD-31](#pd-31) | ⚪ | Código morto | `ModelPipeline`, métodos do `OutputWriter`, `to_uint8_rgb`, `RMSEAccuracy`, `flows`/`styles` | Parcial (2026-10-01): `flows`/`styles` removidas; o resto continua decidido |
| [PD-32](#pd-32) | ⚪ | Arquitetura | Três classes de pipeline idênticas; separação só por convenção | Aberta |
| [PD-33](#pd-33) | 🟡 | Contrato | A chave `segmentation` significa duas coisas (Cellpose e saída final) | Decidida: renomear |
| [PD-34](#pd-34) | ⚪ | Ideia | Usar a probabilidade contínua do Cellpose em vez do alpha binário | Implementada como opção (2026-10-01); falta a ablação |
| [PD-35](#pd-35) | ⚪ | Arquitetura | Acoplamento implícito a atributos internos (`step.model`, `self.model.model`) | Aberta |
| [PD-36](#pd-36) | 🟡 | Modelo | A entrada da MarkerUNet não tem a normalização ImageNet que o encoder espera | Decidida: ablação |
| [PD-37](#pd-37) | ⚪ | Resolução | Reduções bilineares sem antialias: o ScribblePrompt lê 6,6% dos pixels | ✅ Encerrada: o antialias piora (oráculo) |
| [PD-38](#pd-38) | 🟡 | Modelo/Treino | 24,4 M parâmetros treinados, sem congelar nada, com 30 imagens | Decidida: ablação |
| [PD-39](#pd-39) | ⚪ | Ideia | ScribblePrompt-SAM como rede final (via `mask_input`) | Adiada (decisão do autor) |
| [PD-40](#pd-40) | 🔴 | Resultados | Com marcadores do tipo da tese, o ScribblePrompt não alcança o Cellpose nem com o marcador ideal | Aberta |
| [PD-41](#pd-41) | 🟠 | Loss | `BorderTerm` pune marcadores sobre 16–17% dos pixels de núcleo real | Aberta |
| [PD-42](#pd-42) | ⚪ | Docs | Docstring do `LossComposer` diz que o Size olha a predição (olha os marcadores) | Aberta |
| [PD-43](#pd-43) | 🟡 | Loss | Com GT vazio: DMap e TV dividem por zero; Size perde a normalização | Aberta (latente) |
| [PD-44](#pd-44) | 🟠 | Experimento | Plano do experimento-base "só Dice + TV" (256², só positivos) | Decidida: a executar |
| [PD-45](#pd-45) | ⚪ | Treino | Médias por época são por batch, não por imagem (o último batch tem 2 imagens) | Aberta |
| [PD-46](#pd-46) | 🟡 | Treino | Não há seleção do melhor modelo nem early stopping; avalia-se a última época | Decidida: melhor Dice + early stopping |
| [PD-47](#pd-47) | 🟡 | Publicação | Resultados (oráculo e próximos experimentos) precisam ser salvos e reproduzíveis para o artigo | Aberta |
| [PD-48](#pd-48) | 🟡 | Testes | Testes cobrem a "tubulação" com dummies; nada de redes reais, dados, métricas ou valores das losses; sem execução automática | Aberta |
| [PD-49](#pd-49) | 🟠 | Modelo/Experimento | Só positivo: um fundo ≥ 1e-3 no canal positivo faz o ScribblePrompt marcar a imagem inteira; a sigmoid da MarkerUNet nunca dá 0 | Aberta (decisão do autor, 09 §9) |
| [PD-50](#pd-50) | ⚪ | Código | Código, testes e notebooks citam a história do projeto (PD-nn, "investigação, P5", "C2", "regra 23") | Aberta (regra do autor, 2026-10-01) |

---

## Detalhes

### PD-01
**🔴 O pipeline é pior que o Cellpose sozinho.** ✅
O melhor experimento (exp. 4) tem Dice 0,648 / IoU 0,481 nas 14 imagens de teste. A máscara do Cellpose, que é a
**entrada** da MarkerUNet, tem Dice 0,810 / IoU 0,682 no mesmo conjunto (média por imagem, limiar 0,5, calculada dos `.npy`
persistidos). Enquanto isso não se inverter, o TCC não mostra ganho.
**Próximos passos:** colocar o Cellpose como linha de base em todo notebook; rodar o E3 (PD-03), que diz qual é o melhor
resultado possível com o ScribblePrompt; atacar PD-04, PD-05 e PD-06, que são as causas prováveis. ❓
**Onde:** [00-visao-geral.md §7](00-visao-geral.md#7-estado-atual-dos-experimentos).

### PD-02
**🟠 O teste oficial é usado como validação e para escolher configurações.** ✅
Todos os notebooks usam as 14 imagens de teste como `val_batches`, e os experimentos são comparados por elas: não há teste
cego. O autor tem receio de tirar imagens do treino.
**Proposta:** validação cruzada k-fold nas 30 imagens de treino (por exemplo, 5 × 24/6) para escolher a configuração; depois
treinar com as 30 e avaliar **uma vez** nas 14. A divisão tem que ser **por imagem**, nunca por recorte.
**Custo (tema 5, medido)** ✅: uma época leva ~1,3–1,7 s no Colab; 5 *folds* × 50 épocas ≈ 7 min por configuração. O k-fold
não tem custo computacional relevante.

### PD-03
**🟠 Os experimentos de isolamento nunca foram executados.** ✅
A investigação do exp. 1 (hoje em [06-experimentos.md §2.3](06-experimentos.md)) recomenda E2 (sensibilidade do ScribblePrompt aos scribbles), E3 (oráculo: marcador =
GT, ou uma erosão do GT) e E5 (rede final dummy). Nenhum notebook faz isso. **O E3 é o mais importante:** se o ScribblePrompt
não passar do Cellpose nem com o marcador ideal, nenhuma MarkerUNet vai passar.
**E3 executado (2026-09-27):** notebook [oraculo_scribbleprompt.ipynb](../../notebooks/exploration/oraculo_scribbleprompt.ipynb),
rodado em CPU fora do Colab. Os resultados estão em [03-modelos.md §8](03-modelos.md#8-resultado-do-oráculo-2026-09-27), e a
conclusão virou a [PD-40](#pd-40). E2 e E5 continuam não executados. O E2 ficou parcialmente respondido, porque o oráculo varia
o tipo de marcador, o canal negativo e a resolução.

### PD-04
**🟠 O scribble negativo é o complemento denso do marcador.** ✅ (código) · autor: "provavelmente foi um erro"
[scribble_prompting_network.py:348-352](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L348-L352):
`neg = sigmoid(T·(0.5 − s))` cobre praticamente toda a imagem fora dos marcadores. O ScribblePrompt foi treinado com
negativos esparsos (artigo, seção 3.2). Na tese, o fundo é um marcador próprio: fixo nos cantos ou aprendido por inversão (5.4.3).
**Alternativas para uma ablação:** (a) canal negativo zerado; (b) negativo esparso e fixo, longe de qualquer núcleo do Cellpose;
(c) negativo gerado a partir do complemento da máscara do Cellpose, à maneira da inversão da tese.
**Evidência adicional (tema 3, 2026-09-27)** ✅: com `m ≈ 0`, o canal negativo vale **0,993**. Todo pixel não marcado,
inclusive o **interior dos núcleos** que o marcador não cobre, é informado ao ScribblePrompt como "fundo". Isso entra em
conflito direto com o L_size, que quer marcadores pequenos: marcador menor → mais negativo sobre os núcleos → segmentação
menor → o Dice pede marcador maior. Ver [03-modelos.md §3.4](03-modelos.md#34-de-marcador-1-canal-para-scribbles-2-canais-).
**Decisão (2026-09-27): usar só scribbles positivos** (canal negativo zerado).
**Confirmação pelo oráculo** ✅: com o negativo atual, a saída é praticamente o próprio marcador. Com o miolo dos núcleos como
marcador (33% da área do GT), a precisão é 0,99 e o recall 0,30 (Dice 0,452 em 128²). Com só positivos, a rede "cresce" a partir
do marcador (recall 0,52, Dice 0,668). **Contraponto:** quando o marcador já é o GT inteiro, tirar o negativo faz a rede
segmentar demais (massa 1,47× o GT, Dice 0,796 contra 0,873). O ganho do "só positivo" é para marcadores pequenos, que são
justamente o que a tese quer.

### PD-05
**🟡 (era 🟠) Resolução: o núcleo mediano vira ~3 px na entrada do ScribblePrompt.** ✅ (medido)
O núcleo mediano tem ~438 px de área, ou ~24 px de diâmetro equivalente, em 1000² ([01-dados.md §4](01-dados.md#4-números-medidos)).
A MarkerUNet trabalha em 256², onde o núcleo tem ~6 px. O ScribblePrompt trabalha em 128², onde tem **~3 px**. Os 5% menores
(~170 px, ~15 px de diâmetro) viram ~2 px. Esse é o item P7 da investigação, reaberto nos exp. 2 a 6.
**Ideia:** processar **recortes** (ex.: 128² ou 256² na resolução nativa) em vez da imagem inteira reduzida. Isso também
multiplica o número de amostras de treino.
**Atualização (tema 3, 2026-09-27)** ✅ medido: o GT reduzido a 128² e ampliado de volta ainda tem **Dice 0,921** (redução com
antialias) ou **0,875** (bilinear sem antialias, como o código faz). Em 256², 0,976 / 0,950. Portanto a resolução **não impede
sozinha** de superar o Cellpose (0,810): ela reduz a margem, mas não explica o PD-01 por si só. Severidade rebaixada para 🟡.
A parte do antialias virou a PD-37.
**Oráculo (2026-09-27)** ✅: a entrada do ScribblePrompt em **256²** melhora todos os marcadores (Dice no treino, só positivo:
GT 0,796 → 0,851; miolo 0,668 → 0,731; centros 0,342 → 0,482). Com o negativo atual e o GT inteiro, sobe de 0,873 para 0,931.
Mudar exige só `net.input_size = (256, 256)` (a rede é totalmente convolucional; as dimensões precisam ser divisíveis por 16,
e 1000 não é). **Recomendação:** passar o ScribblePrompt para 256² nos próximos experimentos. O custo de memória e de tempo sobe ~4×.
Em **512²** (só treino): o miolo com só positivo sobe para 0,784, mas o GT inteiro com só positivo cai para 0,690, porque vaza.
A melhor resolução depende do tamanho do marcador; 256² é o meio-termo mais seguro. Detalhe na PD-40.

### PD-06
**🟠 O mapa de distância é normalizado pelo máximo da imagem inteira.** ✅ (medido) · 📖
[distance_map_step.py:40-43](../../src/pipeline/steps/preprocessing/distance_map_step.py#L40-L43) calcula
`1 − EDT/max(EDT)` com o **máximo global** da imagem. Na tese (eq. 5.7) a normalização é **por objeto**, e o centro de todo
objeto vale 0. Medido nas 30 imagens de treino: o valor **no ponto mais central** de cada núcleo tem mediana **0,48**, e em
**45,5%** dos núcleos até o centro passa de 0,5. O DMapTerm penaliza marcadores até no centro da maioria dos núcleos e empurra
a massa para os poucos blobs grandes.
**Correção sugerida:** calcular a EDT por componente, ou por instância (ver PD-07), e normalizar pelo máximo daquele componente.
**Medição das três versões (2026-09-27, 30 imagens de treino)** ✅ — explicação passo a passo e figura em
[04-losses.md §7](04-losses.md#7-entendendo-o-problema-do-mapa-de-distância-pd-06):

| Mapa | Centro do núcleo (mediana) | Núcleos com centro > 0,5 | Pixels de núcleo < 0,5 |
|---|---|---|---|
| atual (máx. da imagem) | 0,488 | 46,1% | 6,9% |
| por componente | 0,000 | 0,6% | 31,8% |
| por núcleo (XML) | 0,000 | 0,0% | 33,1% |

**Decisão (2026-09-27):** corrigir o mapa **antes** de ajustar o peso do DMap. Depois da correção, o peso precisa ser
recalibrado, porque o mapa passa a ter ~5× mais pixels baratos. Por componente já resolve quase tudo com o GT atual; por
núcleo é o ideal.

### PD-07
**🟠 O GT binário funde núcleos que se tocam.** ✅ (medido)
[monuseg_dataset.py:71](../../src/data/load/monuseg_dataset.py#L71) pinta todos os polígonos com valor 1. Resultado: no
treino, 16.966 núcleos anotados viram **12.648 componentes conexos (−25,5%)**; no teste, 6.697 viram 6.086 (−9,1%). Há ainda
128.007 px no treino cobertos por mais de um polígono. Isso afeta o mapa de distância (PD-06) e qualquer métrica ou loss por
objeto. Existe uma máscara **por instância** do treino em `Binary_masks_instance/` (PD-26) que não é usada.
**Com as 37 imagens (2026-09-29)** ✅: no treino, 24.140 regiões viram 18.160 componentes (**−24,8%**); no teste, −9,1% (sem
mudança). Com a rasterização pelo centro do pixel, a sobreposição entre polígonos é 1,00% do primeiro plano no treino. Numa máscara
por instância, "o último desenhado vence" apaga 2 núcleos inteiros, e "o menor vence" não apaga nenhum. **Proposta:** E1-c do
[09 §13.3](09-correcoes-pontuais.md): gerar a máscara por instância a partir dos XMLs (as `Binary_masks_instance` foram removidas, PD-26).
**Parcial (2026-09-29, `3dc9d75`):** o `MonusegDataset` devolve `ground_truth_instances` (`int32`, rótulos 1..N na ordem do
XML, "o menor vence" na sobreposição): 24.133 instâncias no treino e 6.697 no teste. O `ground_truth` binário continua fundindo
núcleos vizinhos, o que é intencional (a avaliação é por pixel). **Falta:** usar as instâncias no Dmap (PD-06, etapa 4) e
persisti-las (etapa 5). ⚠️ O `resize_sample` do notebook de pré-processamento monta um dicionário novo só com `id`, `image` e
`ground_truth`, e **descarta** a chave nova; precisa ser ajustado na etapa 4.

### PD-08
**🟡 A aumentação é estática e não há shuffle.** ✅
Nos notebooks, `make_batches(train_preprocessed, 4, augment=True)` roda **uma vez**, antes do treino. Todas as épocas veem os
mesmos 8 batches, com as mesmas rotações e flips e na mesma ordem.
**Correção:** sortear a aumentação dentro do loop (por época) e embaralhar as amostras.
**Decisão (2026-09-27):** fica no módulo de dados/treino em `src/` da PD-19, e não copiado em cada notebook.

### PD-09
**🟡 Os runs provavelmente validaram com a BatchNorm em modo treino.** ❓
O controle `train()`/`eval()` por modo entrou em `ef7d731` (22/09). Os runs são de 14/08 a 12/09. Se o Colab usou o código da
época, a validação usou as estatísticas do lote e ainda atualizou as *running stats* com imagens de teste.
**Ação:** reavaliar em `eval()` o melhor modelo, o que depende de PD-10 (checkpoints).

### PD-10
**🟡 Reprodutibilidade dos experimentos.** ✅
- Nenhum checkpoint salvo nos exp. 3 a 6 (a célula está comentada).
- No exp. 4, as células finais têm saídas de **outro run**; a célula 24 dá `NameError`.
- Os markdowns dos exp. 3 a 6 dizem "Size simétrico", mas o código atual não é simétrico.
- No exp. 5, `dense_soft` é descrito como "quase binário", e `scribble_temperature` é ignorada nesse modo.
- A configuração da loss do exp. 3 mudou depois do run e o markdown não foi atualizado.

**Decisão (2026-09-27):** não corrigir os notebooks antigos agora. **Depois das correções** (PD-44 e seguintes), remover os
notebooks e células que não fizerem mais sentido e manter só o material que embasa o trabalho. O histórico dos exp. 1–6 fica em
[06-experimentos.md](06-experimentos.md) e no git.

### PD-11
**⚪ Os testes e a docstring do Size descrevem a fórmula antiga.** ✅ (executado em 2026-09-27: os 3 testes falham, como previsto)
A fórmula atual (`peso·Σmarcador/ΣGT`) é o L_size da tese (eq. 5.5) e é **intencional**. Mas
[test_object_size_loss.py](../../tests/test_object_size_loss.py) espera `|ratio−1|` e deve falhar, e a docstring de
[object_size_loss.py:8](../../src/losses/object_size_loss.py#L8) diz "penaliza o desvio de tamanho".

### PD-12
**🟡 Size, DMap e TV são normalizados pela soma do batch inteiro.** ✅ · 📖
Na tese, a normalização é por objeto. Aqui, `y_true.sum()` soma o batch todo
([object_size_loss.py:32](../../src/losses/object_size_loss.py#L32), [distance_map_loss.py:49](../../src/losses/distance_map_loss.py#L49),
[total_variation_loss.py:38](../../src/losses/total_variation_loss.py#L38)). O peso efetivo de cada imagem depende das outras
do mesmo batch. Detalhar no tema 4.

### PD-13
**🟡 Desbalanceamento de classes não tratado.** ✅
O primeiro plano ocupa em média 24,6% dos pixels no treino e 21,7% no teste. Não há `pos_weight` nem focal loss. É o item P8 da investigação.
(O ScribblePrompt usa Dice + Focal no próprio treino, segundo o artigo, seção 3.4.) 📖

### PD-14
**⚪ `GradNormCallback` mede a norma depois do clipping.** ✅
[trainer.py:179-188](../../src/training/trainer.py#L179-L188): o callback roda depois de `clip_grad_norm_`, então o valor
1,0 só indica que o clip atuou. O diagnóstico precisa da norma antes do clip.

### PD-15
**🟡 O checkpoint do ScribblePrompt é baixado sem verificar hash.** ✅
[scribble_prompting_network.py:381-384](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L381-L384)
baixa um `.pt` de um link do Dropbox e o pacote o desserializa (pickle). Se o arquivo for trocado, ele pode executar código.
**Correção:** fixar o SHA-256 do arquivo em que você confia e recusar o arquivo se não bater:

```python
# atual                                   # proposto
urllib.request.urlretrieve(url, dest)     urllib.request.urlretrieve(url, dest)
                                          digest = hashlib.sha256(dest.read_bytes()).hexdigest()
                                          if digest != cls.CHECKPOINT_SHA256[version]:
                                              dest.unlink()
                                              raise RuntimeError("Checkpoint com hash inesperado.")
```

**Hash de referência (2026-09-27):** o arquivo baixado pelo link oficial (o mesmo do README do ScribblePrompt) nesta data tem
SHA-256 `43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe`. Ele é o candidato para `CHECKPOINT_SHA256["v1"]`.
Vale conferir contra o arquivo que o Colab usou nos experimentos: o notebook do oráculo imprime o hash.
Mitigação parcial: com torch ≥ 2.6, `torch.load` usa `weights_only=True` por padrão e recusa objetos arbitrários. O Colab
atual já usa essa versão, e o `requirements.txt` só exige `torch>=2.1.0`.

### PD-16
**🟡 Fallbacks silenciosos.** ✅
- `preprocessamento_monuseg_persistido.ipynb` (célula 9): se o Cellpose não carrega, usa o **próprio GT** como
  `segmentation`. O canal alpha passa a conter a resposta, ou seja, **vazamento total**.
- Os notebooks de experimento trocam o ScribblePrompt por uma `DummyFinalNetwork` se ele falhar.
- `MarkerStep` sem modelo usa a máscara do Cellpose como marcador, só com um *warning*
  ([marker_step.py:115-124](../../src/pipeline/steps/inference/marker_step.py#L115-L124)).

Nenhum run atual foi afetado: as saídas mostram "Cellpose disponível" e "Rede final: ScribblePrompt", e o `meta.json` diz
`segmentation_source: cellpose`. Mas é um risco. **Correção:** falhar com erro, ou exigir uma flag explícita para o fallback.
**Parcial (2026-10-01, `9395df2`, C6 parte do notebook; base em [09 §7](09-correcoes-pontuais.md)):** a célula 9 do
`preprocessamento_monuseg_persistido.ipynb` não tem mais o `try/except`: se o `CellposeStep()` falhar (sem GPU ou, desde a
PD-29, com nome de modelo desconhecido), a execução para. Saíram também o `segmentation = GT` do `run_preprocess`, o pipeline
condicional e o `segmentation_source: ground_truth_fallback` do `meta.json`. O markdown das células 0 e 16 descreve o
comportamento novo ("Cellpose obrigatório"), sem citar a mudança (regra da PD-50). As saídas guardadas do run de 16/08 ficaram
como estão: são o registro do run que gerou os `.npy` atuais. ❓ A célula ainda não foi executada depois da mudança; isso
acontece no Colab, na PD-30. **Falta:** o fallback do `MarkerStep` (etapa 7). O `DummyFinalNetwork` dos notebooks de
experimento fica como está (PD-10).

### PD-17
**⚪ `pipeline_test.py` está quebrado.** ✅
Ele importa `src.pipeline.steps.cellpose_step`, `marker_step` e `segmentation_step`, que foram movidos ou removidos. Remover
o arquivo ou reescrevê-lo.

### PD-18
**⚪ Partes de `datasets.yml` não são usadas.** ✅
[datasets.yml:11](../../configs/datasets.yml#L11) (`train_split: 0.8`), `loader_config` (L23) e `preprocessing` (L31: 256²,
normalização, augmentation) não são lidos por nenhum código de treino. Só um notebook-tutorial imprime `loader_config`.
Isso sugere comportamentos que não existem.

### PD-19
**🟡 Dois caminhos de carregamento de dados.** ✅
[MonusegPreprocessedDataset](../../src/data/load/monuseg_preprocessed_dataset.py#L66-L78) **roda o pipeline na hora** sobre o
dataset bruto; **não lê** os `.npy`. Os experimentos não o usam: cada notebook tem seu `load_preprocessed` e seu
`build_batch`, copiados de um para outro. O antigo `ARCHITECTURE.md` (regra 5 e fase 2) descrevia o contrário.
**Ideia:** um dataset que leia `data_source/MoNuSegPreprocessed/` e faça a aumentação por amostra, usado por todos os notebooks.
**Decisão (2026-09-27): criar o módulo em `src/`.** Escopo combinado, para os próximos experimentos importarem em vez de copiar:
1. um dataset que **lê os `.npy` do disco** (o `load_preprocessed` de hoje), com opção de subconjunto de ids (para os *folds*);
2. **aumentação por amostra, sorteada a cada época** (rot90 + flips, aplicada juntos a imagem, rgba, GT e Dmap), com shuffle (PD-08);
3. o `collate` que monta o batch no formato de hoje (`build_batch`);
4. as **métricas binárias por imagem** (`compute_binary_metrics` / `evaluate_val_quality`), com o Cellpose como linha de base (PD-01);
5. um **laço de k-fold** sobre as 30 imagens de treino (PD-02), com seleção do melhor modelo e early stopping (PD-46).

Vale decidir, na hora de implementar, se o `MonusegPreprocessedDataset` atual (que roda o pipeline na hora) é substituído ou
mantido para o tutorial.

### PD-20
**⚪ `.pyc` versionados.** ✅ Há arquivos em `src/**/__pycache__/` no git, apesar do `__pycache__/` no `.gitignore`.
O mesmo vale para `src/cell_fuzzy_seg.egg-info/`, que é gerado pelo `pip install -e .`.
**Correção:** `git rm -r --cached` nesses caminhos e acrescentar `*.egg-info/` ao `.gitignore`.

### PD-21
**🟡 1,6 GB de dados no git.** ✅
`data_source/` tem 1,6 GB (o pack do repositório tem 614 MB). Desse total, 1,1 GB são `.npy` **derivados**
(`MoNuSegPreprocessed/`), que podem ser regenerados. ❓ Talvez esteja no git para o Colab clonar tudo de uma vez.
**Alternativas:** Git LFS; guardar no Google Drive e montar no Colab; ou gerar no Colab e manter fora do git.
**Decisão (2026-09-27):** o autor não teve problemas com isso; **fica como está por enquanto**. Reabrir se o clone ou o push
começarem a falhar, ou se o GitHub reclamar do tamanho.

### PD-22
**⚪ Falta o artigo do Cellpose-SAM.** O código usa `cpsam`; o PDF em `docs/papers/` é o Cellpose de 2020.

### PD-23
**🟡 Contagem de núcleos diferente da oficial.** ✅ (contagem) · ❓ (causa)
Os XMLs do treino têm **16.966** regiões e os do teste **6.697**. O site do desafio fala em **~22.000** no treino e ~7.000 no
teste. O artigo original (Kumar et al., 2017) fala em "mais de 21.000". A diferença (~23% a menos no treino) pode vir de outra
versão do release ou de anotações incompletas nos XMLs baixados. Confirmar antes de citar números no texto.
**Ação (autor):** o download foi, provavelmente, do site do desafio. Baixar de novo **direto da fonte oficial** e comparar a
contagem de `<Region ` por XML com a tabela da §4 de [01-dados.md](01-dados.md#4-números-medidos). Um comando para comparar:
`grep -c '<Region ' Annotations/*.xml`.

**Novo download (2026-09-29, feito pelo autor e comparado com o commit `b865652`)** ✅
- **Teste:** os 14 `.tif` e os 14 `.xml` são **idênticos, byte a byte** (SHA-256), aos que estão no git. Continuam com 6.697 regiões.
  Todos os resultados no teste (Cellpose 0,810, exp. 4 0,648, oráculo) continuam valendo.
- **Treino:** o download tem **37 imagens, e não 30**. As 30 antigas são idênticas às do git (`.tif` e `.xml`), e as 7 que faltavam
  somam **7.174 regiões**:

  | Imagem nova | Regiões | µm/px |
  |---|---|---|
  | TCGA-BC-A217-01Z-00-DX1 | 757 | 0,2472 |
  | TCGA-F9-A8NY-01Z-00-DX1 | 1.361 | 0,2525 |
  | TCGA-FG-A87N-01Z-00-DX1 | 742 | 0,2527 |
  | TCGA-MH-A561-01Z-00-DX1 | 720 | 0,2527 |
  | TCGA-UZ-A9PJ-01Z-00-DX1 | 1.078 | 0,2527 |
  | TCGA-UZ-A9PN-01Z-00-DX1 | 1.207 | 0,2525 |
  | TCGA-XS-A8TJ-01Z-00-DX1 | 1.309 | 0,2525 |

  Total do treino: **24.140 regiões** (16.966 + 7.174). As 7 são 1000×1000 e estão em 40×; ao contrário das outras 30, não usam
  compressão LZW. Nenhum id coincide com o teste (a `FG-A87N` é do mesmo centro que a `FG-A4MU` do teste, mas de outro paciente).
- **Causa da diferença:** o download antigo tinha só 30 das 37 imagens de treino. ❓ Os 24.140 ainda não batem com os "~22.000"
  do site, mas o número do site é aproximado. Falta registrar a URL exata de onde o download foi feito (PD-27).
- **O que muda:** tudo o que foi medido no **treino** com 30 imagens: as contagens do [01-dados.md §4](01-dados.md#4-números-medidos),
  o Cellpose no treino (0,802), o oráculo no treino, o k-fold planejado (PD-02) e o `MoNuSegPreprocessed/train`, que só tem as 30.
- **A organização das pastas mudou** (não mexi em nada; decisão pendente):
  - treino em `MoNuSegTrainingData/MoNuSeg 2018 Training Data/{Annotations,Tissue Images}/`, com espaços no nome, além de
    `__MACOSX/` e `.DS_Store` (arquivos do macOS, 77 no total);
  - teste solto em `MoNuSegTestData/`, sem as subpastas;
  - o [datasets.yml](../../configs/datasets.yml) espera `Tissue_Images/` e `Annotations/` nos dois, então o `MonusegDataset`
    não acha mais os arquivos;
  - o `data_source/README.md` (atribuição do MoNuSeg, PD-27) foi **apagado** na substituição, e `Binary_masks/` e
    `Binary_masks_instance/` também sumiram (ver PD-26).

**Decisões do autor (2026-09-29), todas conforme a recomendação:**
1. treinar com as **37** imagens;
2. voltar à organização antiga (`Tissue_Images/` + `Annotations/` no treino e no teste) e descartar `__MACOSX/` e `.DS_Store`;
3. restaurar o `data_source/README.md` e registrar a origem do novo download;
4. deixar `Binary_masks*/` apagadas (não são oficiais; continuam no histórico do git).

**Feito (2026-09-29, `f263379`)** ✅: arquivos movidos para as pastas antigas, sem mudar o conteúdo. A permissão de execução que veio do zip
foi tirada, para o git não registrar mudanças falsas nos 30 arquivos antigos. Os arquivos do macOS foram apagados, e o README foi
restaurado e atualizado (37 imagens, 24.140 regiões, alterações em relação ao original). O `MonusegDataset` real, rodado com o
`datasets.yml` sem mudanças (com dublês de `torch` e `cellpose.io`), acha **37 pares no treino e 14 no teste**. Nenhum código mudou.
**Origem do download (informada pelo autor, 2026-09-29):** página oficial de dados do desafio,
<https://monuseg.grand-challenge.org/Data/>. Registrada no `data_source/README.md`.
**Continua aberto:** regerar o `MoNuSegPreprocessed/train` com as 37 imagens (etapa 5 do plano) e refazer os números do treino
(01-dados §4) na etapa 1.

### PD-24
**🟡 Três imagens de treino têm outra escala.** ✅
`TCGA-HE-7128`, `-7129` e `-7130` têm `MicronsPerPixel = 0,5005` (≈ 20×). Das outras 41, 37 têm entre 0,2456 e 0,2527 (≈ 40×)
e 4 não informam a escala (`TCGA-AY-A8YK`, `-KB-A93J`, `-NH-A8F7`, `-RD-A8N9`). Nessas três, os núcleos têm área mediana de 116 a 152 px (contra ~438 px no geral) e há 1.076
a 1.863 núcleos por imagem. Juntas, elas são **27% de todos os núcleos do treino**. O Cellpose roda com `diam_mean=30` fixo
para todas as imagens. O autor não sabia disso (2026-09-27).

**Impacto medido no Cellpose** ✅: nessas três imagens, o Dice fica entre 0,750 e 0,792, contra média de 0,805 nas outras 27
(a pior das 27 tem 0,706), e a massa segmentada é só 68–74% da do GT (0,79 nas outras). O impacto é moderado. O maior
problema está mais adiante: com área de ~116–152 px (≈ 12–14 px de diâmetro), esses núcleos ficam com **~1,5 px** em 128²
(PD-05).
**Opções:** (a) reamostrar as três para ~0,25 µm/px, ou seja, ampliar 2× e recortar em blocos de 1000², o que as deixa na
mesma escala do resto; (b) passar ao Cellpose o diâmetro proporcional à escala de cada imagem; (c) no mínimo, registrar no
texto e mostrar os resultados dessas três separadamente.

### PD-25
**⚪ Detalhes da rasterização.** ✅
[monuseg_dataset.py:69](../../src/data/load/monuseg_dataset.py#L69) usa `int(x)`, que **trunca** coordenadas fracionárias
(viés de até 1 px para cima e para a esquerda). 5 polígonos do treino têm menos de 3 vértices. O impacto é pequeno; registrado
para o texto.
**Revisão com medição (2026-09-29, 37 + 14 imagens; severidade ⚪ → 🟡)** ✅ — base completa em
[09-correcoes-pontuais.md §13](09-correcoes-pontuais.md):
- **O diagnóstico acima estava errado em parte.** O XML usa, ao que tudo indica, a convenção de canto do pixel (❓ inferido dos
  dados). Nela, `int()` é o certo, e `round()` piora o alinhamento com o Cellpose no treino (0,7991 contra 0,8016).
- **O problema real é o `cv2.fillPoly`, que pinta todo pixel tocado pela borda:** o GT tem ~10% mais pixels que a área dos
  polígonos (1,096 no treino e 1,099 no teste; o `Area` do XML é exatamente a área do polígono). Rasterizar pelo centro do pixel em
  `(c + 0,5, r + 0,5)` acerta a área (0,995 e 1,011) e alinha melhor com o Cellpose (Dice 0,8099 contra 0,8016 no treino e 0,8218
  contra 0,8103 no teste).
- Os 5 polígonos degenerados têm 2 vértices e `Area = 0`: são cliques soltos, não núcleos.
- **Efeito:** mudar a rasterização muda o GT de treino e de teste, e todas as linhas de base precisam ser refeitas (etapa 5).
**Proposta:** E1-a e E1-b do 09 §13.3.
**✅ Resolvida (2026-09-29, `3dc9d75`):** decisão do autor, conforme a recomendação. O `MonusegDataset` rasteriza pelo centro
do pixel (`skimage.draw.polygon` sobre `X − 0,5`, `Y − 0,5`) e descarta as regiões com < 3 vértices ou área 0, com log. Conferido
nos dados reais: o GT binário novo é idêntico à rasterização medida na base em 37/37 e 14/14 imagens; a área fica em 0,995 e
1,011 da anotada; o Dice do Cellpose contra o GT novo é 0,8099 (treino, 30) e 0,8218 (teste). `scikit-image==0.24.0` entrou no
`requirements.txt`. **Atenção:** o `MoNuSegPreprocessed/` ainda tem o GT antigo, até a etapa 5; os números dos exp. 1 a 6 e do
oráculo são do GT antigo.

### PD-26
**⚪ Máscaras extras sem origem conhecida.** ✅ · ❓
- `MoNuSegTrainingData/Binary_masks/*.png` está no repositório desde o primeiro commit que moveu os dados (28/04).
- `Binary_masks_instance/*.npy` (int64) foi adicionada em 10/08 (`6a4801d`).
- As duas coincidem entre si (`instância > 0 == png`). O maior rótulo de cada máscara de instância é igual ao número de
  regiões do XML correspondente.
- Elas diferem da máscara gerada pelo código em ~2,2% dos pixels da imagem, quase sempre porque são **menores**: há
  662 mil px só na máscara gerada contra 9 mil só no PNG. Também separam núcleos vizinhos: 13.797 componentes contra 12.648.
- Só existem para o treino, e nenhum código as usa. Podem resolver o PD-07 no treino.

**Origem (resposta do autor, 2026-09-27):** vieram do download original. ❓ Dois pontos a conferir no novo download (PD-23):
a página oficial menciona só XML + código MATLAB de conversão, e `Binary_masks_instance/` só entrou no git em 10/08 (`6a4801d`).
**O que falta:** decidir se as máscaras por instância serão usadas (por exemplo, para o mapa de distância por núcleo, PD-06).
Como não existem para o teste, gerá-las a partir dos XMLs pode ser mais consistente.
**Novo download (2026-09-29)** ✅: o pacote baixado de novo **não tem** `Binary_masks/` nem `Binary_masks_instance/`, só
`Annotations/` e `Tissue Images/`. Então essas máscaras **não vêm do download oficial**, ao contrário do que se lembrava. ❓ A origem
continua desconhecida, e elas cobrem só 30 das 37 imagens de treino. Reforça a proposta de gerar as instâncias a partir dos XMLs (PD-07).
**✅ Resolvida (2026-09-29, `f263379`):** por decisão do autor, as duas pastas foram **removidas** do repositório (continuam no histórico do
git, por exemplo `git show b865652:data_source/MoNuSegTrainingData/Binary_masks_instance/<id>.npy`). As máscaras por instância,
se forem necessárias, serão geradas dos XMLs (PD-07).
**Elas afetaram algum treino? Não** ✅ (conferido em 2026-09-29, a pedido do autor). Nenhuma célula de código de nenhum notebook,
nem nada em `src/`, `tests/` ou `configs/`, lê `Binary_masks`. No histórico, o nome só aparece no markdown de um tutorial
(`45bdc4b`). Nos dados, o `ground_truth` persistido que os exp. 3 a 6 usaram é **idêntico** à máscara gerada dos XMLs em 30/30
imagens, e difere do PNG em 2,24% dos pixels em média (máx. 4,81%). Portanto o GT de treino sempre veio dos XMLs oficiais.

### PD-27
**🟡 Licença e atribuição do MoNuSeg.** ✅
O repositório `github.com/Pedro-io/cell-fuzzy-seg` é **público** e contém o MoNuSeg, cuja licença é **CC BY-NC-SA 4.0**.
Redistribuir é permitido, desde que haja **atribuição** (citar Kumar et al., 2017), uso **não comercial** e a **mesma
licença** para os dados. Hoje `LICENSE`, `README.md` e `CITATION.cff` estão vazios.
**Ação:** colocar a atribuição e a licença dos dados, por exemplo num `data_source/README.md`, e escolher uma licença para o código.
**Feito (2026-09-27, não commitado):** [data_source/README.md](../../data_source/README.md) com fonte, citação, licença e o
aviso de que `MoNuSegPreprocessed/` é derivado e herda a CC BY-NC-SA. **Falta:** escolher a licença do **código** e preencher
`LICENSE`, `README.md` e `CITATION.cff`, que continuam vazios.
**Atualização (2026-09-29, `f263379` e seguinte):** o `data_source/README.md`, apagado sem querer na troca dos dados, foi restaurado
e agora registra a origem do novo download (<https://monuseg.grand-challenge.org/Data/>), o conteúdo (37 + 14 imagens) e as
alterações em relação ao original. Continua faltando a licença do **código**.

### PD-28
**⚪ Trechos errados no `ARCHITECTURE.md`.** ✅
Ele cita a pasta `registry/` (regra 15), que não existe; chama a rede final de "treinável" (ela é congelada); os exemplos de
uso têm assinaturas antigas (`MonusegDataset(root_dir=...)`); e descreve o `MonusegPreprocessedDataset` como leitor do disco
(PD-19). Será substituído pelo tema 2.
**Resolvida (2026-09-27, não commitado):** o arquivo foi removido e substituído por
[02-pipeline-e-arquitetura.md](02-pipeline-e-arquitetura.md), que mantém a numeração do contrato e das regras e registra
esses erros na §8.

### PD-29
**🟡 A proteção contra nome de modelo errado do Cellpose nunca executa.** ✅
[cellpose_step.py:77-80](../../src/pipeline/steps/preprocessing/cellpose_step.py#L77-L80) faz
`from cellpose.models import MODEL_LIST`. No `cellpose==4.1.1` esse nome **não existe** (há `MODEL_NAMES = ["cpsam"]`). O
`except (ImportError, AttributeError): return` engole o erro, e o aviso do item P9 da investigação nunca aparece. A
consequência é pequena, porque o próprio Cellpose loga "pretrained model … not found, using default model" e o padrão é o
`cpsam`. Mas o código dá uma falsa sensação de proteção.
**Correção:** usar `MODEL_NAMES` + `get_user_models()` do Cellpose 4, ou remover a função e confiar no aviso da biblioteca.
**Base adicional (2026-09-28)** ✅: o aviso da biblioteca mostra o caminho do modelo **padrão**, e não o nome pedido, porque a
variável é trocada antes do log (`cellpose/models.py` v4.1.1, L130-140). Detalhe em [09 §5](09-correcoes-pontuais.md).
**✅ Resolvida (2026-10-01, `f4fcf3a`):** o autor decidiu que um nome errado deve **levantar erro**. O `_warn_if_model_unavailable`
virou `_validate_model_name`, que aplica a mesma regra da biblioteca (arquivo existente, ou nome em `MODEL_NAMES +
get_user_models()`) e levanta `ValueError` **antes** de carregar o Cellpose e antes da checagem de GPU. O import é direto: se a
API mudar de novo, o erro aparece em vez de sumir. Testes: `tests/test_cellpose_step.py` (5, com um `cellpose` falso). Efeito nos
resultados: nenhum (`"cpsam"` passa).

### PD-30
**🟡 Parâmetros do Cellpose.** ✅ (conferido em `cellpose/models.py` v4.1.1)
- `diam_mean=30` no construtor é **ignorado** no Cellpose ≥ 4.0.1. O aviso aparece na saída do notebook de pré-processamento.
- `diameter=30` no `eval` faz a imagem ser reescalada por `30/diameter = 1`, ou seja, **não reescala**. O núcleo mediano
  medido tem ~24 px (40×) e ~13 px nas três imagens em 20× (PD-24). Passar o diâmetro real (fator 1,25, ou ~2,3 nas imagens
  em 20×) pode melhorar a segmentação do Cellpose, que é a entrada da MarkerUNet e a linha de base (PD-01).
- `flow_threshold=0.2` (padrão 0,4) e `min_size=4` (padrão 15) diferem do padrão sem motivo registrado. ❓
**Ação:** testar o Dice do Cellpose com `diameter` ≈ 24 (e por imagem, conforme `MicronsPerPixel`) e com os valores padrão,
antes de regerar `MoNuSegPreprocessed/`.
**Decisão (2026-09-27):** o autor topa testar o diâmetro. Os valores 0,2 e 4 vieram de testes dele ("melhoraram a segmentação
inicial"), mas os números não foram guardados. Plano do teste, no Colab e medido **nas 30 imagens de treino** (não no teste, por PD-02):
1. atual: `diameter=30`, `flow_threshold=0.2`, `min_size=4`;
2. padrão da biblioteca: `diameter=None`, `0.4`, `15`;
3. `diameter=24` com `0.2`/`4`;
4. diâmetro por imagem: 24 nas imagens em 40× e ~13 nas três em 20× (`TCGA-HE-7128/7129/7130`).

Registrar o Dice, o IoU e a razão de massa **por imagem**. Só regerar `MoNuSegPreprocessed/` com a melhor configuração.

### PD-31
**⚪ Código morto.** ✅ Não é usado por nada em `src/`, `tests/` nem nos notebooks:
- `ModelPipeline` ([model_pipeline.py](../../src/pipeline/model_pipeline.py)), resto da versão de maio;
- `OutputWriter.save_all`, `save_segmentation`, `save_markers`, `save_overlay`, `save_rgba` e `_colorize`. O `save_rgba`
  ainda usa um caminho sem subpasta e grava float em PNG;
- `to_uint8_rgb` ([image_utils.py:4](../../src/utils/image_utils.py#L4));
- `RMSEAccuracy` (exportada em `losses/__init__.py`);
- as chaves `flows` e `styles` gravadas pelo `CellposeStep`. `styles` é só zeros no Cellpose 4.
**Decisão (2026-09-27): remover.** Se alguma função for necessária para as figuras do TCC, será reimplementada depois
(ela continua no histórico do git).
**Parcial (2026-10-01, C7 parte do Cellpose):** o `CellposeStep` não grava mais `flows` nem `styles`. Do `flows`, só o mapa de
probabilidade continua, numa chave própria (`cellpose_prob`, PD-34). O resto da PD-31 (`ModelPipeline`, métodos do
`OutputWriter`, `to_uint8_rgb`, `RMSEAccuracy`) continua para a C7.

### PD-32
**⚪ Três classes de pipeline idênticas.** ✅
`PreprocessingPipeline`, `TrainingPipeline` e `ModelPipeline` têm o mesmo laço (percorrer Steps, logar, repassar exceção). A
separação entre pré-processamento e treino é **só de nome**: as regras 10, 13 e 14 não são verificadas pelo código.
**Opções:** manter (a intenção fica explícita e o custo é baixo); ou uma classe base comum. Como mínimo, remover o `ModelPipeline` (PD-31).

### PD-33
**🟡 A chave `segmentation` tem dois significados.** ✅
Na fase 1, é a máscara de instâncias do Cellpose (`CellposeStep`). Na fase 2, é a segmentação final do ScribblePrompt
([frozen_segmentation_step.py:105](../../src/pipeline/steps/inference/frozen_segmentation_step.py#L105)), que o `Trainer`
lê como `prediction_key="segmentation"`. Hoje não há perda de dados (a fase 2 não carrega a máscara do Cellpose), mas o nome
confunde e basta alguém carregar o `segmentation` do disco num experimento para ele ser sobrescrito sem aviso.
**Correção sugerida:** renomear a saída final para `final_segmentation`, o que exige mudar o `prediction_key` do `Trainer`,
os testes e os notebooks.
**Decisão (2026-09-27): renomear para `final_segmentation`.** Pontos a mudar juntos: `FrozenSegmentationStep.forward`, o
padrão de `Trainer.prediction_key`, as docstrings de `LossComposer`/`LossTerm`/`TrainingPipeline`, os testes que leem
`data["segmentation"]` depois do `TrainingPipeline` e as células dos notebooks que usam essa chave.

### PD-34
**⚪ Ideia: usar a probabilidade contínua do Cellpose.** ❓
O Cellpose calcula por pixel a probabilidade de ser célula (`flows[2]`), mas o `RGBAStep` passa à MarkerUNet só `segmentation > 0`
(binário). Um 4º canal contínuo carrega a incerteza do Cellpose, que é justamente onde a MarkerUNet poderia corrigir. A mesma
ideia vale para usar os fluxos (5 canais). Exige mudar o `CellposeStep`, o `RGBAStep` e regerar os dados.
**Decisão do autor (2026-10-01):** aplicar a **variante (a)** (probabilidade no lugar da máscara, 4 canais), **parametrizável**,
com `float16` em disco. As variantes (b) (máscara + probabilidade, 5 canais) e (c) (+ vetores de fluxo, 6 canais) ficam como ideia.
**Implementada como opção (2026-10-01)** ✅, base em [09 §14](09-correcoes-pontuais.md):
- `CellposeStep` grava `cellpose_prob` = sigmoide do `flows[2]` em `float16`. O `flows[2]` é um logit: conferido no Cellpose
  4.1.1, que treina esse canal com `BCEWithLogitsLoss` (`train.py`, L47/L51) e corta em `cellprob > 0` (`dynamics.py`, L647).
  No 2D, o mapa volta ao tamanho original mesmo com reescala (`_run_net`, `resample=True`); o step confere o formato mesmo assim.
- `RGBAStep(alpha="mask" | "prob")`, com `"mask"` como padrão: **nenhum resultado muda** até alguém escolher `"prob"`.
- `SaveResultsStep` salva `cellpose_prob` por padrão (~2 MB por imagem, ~102 MB para as 51). O notebook de pré-processamento
  ganhou `RGBA_ALPHA` e grava `rgba_alpha` no `meta.json`.
**Falta:** regerar os dados (etapa 5) e a **ablação** `"mask"` × `"prob"` na fase 2. A escolha do alpha na hora de carregar os
dados, sem refazer o `rgba`, fica para o módulo da PD-19. ❓ Ainda não foi medido quantos pixels têm probabilidade alta fora de
qualquer máscara do Cellpose (a máscara também depende dos fluxos, do `flow_threshold` e do `min_size`).

### PD-35
**⚪ Acoplamento implícito a atributos internos.** ✅
Sem imports, mas por nome de atributo:
- o `GradNormCallback` procura `step.model` e `step.final_network`
  ([grad_norm_callback.py:154-158](../../src/training/callbacks/grad_norm_callback.py#L154-L158));
- o `MarkerStep` usa `self.model.model`, a `smp.Unet` dentro da `MarkerUNet`
  ([marker_step.py:71](../../src/pipeline/steps/inference/marker_step.py#L71),
  [:204](../../src/pipeline/steps/inference/marker_step.py#L204)).

Se alguém renomear esses atributos, o callback passa a medir zero e o Step quebra, sem erro de import.
**Opção:** o `MarkerStep` chamar `self.model(x)` (o `forward` da `MarkerUNet`) e os Steps exporem `trainable_parameters()`.

### PD-36
**🟡 A entrada da MarkerUNet não tem a normalização do ImageNet.** ✅
O encoder ResNet34 vem pré-treinado no ImageNet com entradas normalizadas por canal (média ≈ 0,45, desvio ≈ 0,22). O `rgba`
chega em [0,1] sem normalização, e o `smp.encoders.get_preprocessing_fn` não é usado. As BatchNorms do encoder compensam em
parte durante o *fine-tuning*. Detalhe: com `in_channels=4`, o smp 0.5.0 inicializa o canal alpha com **cópia dos pesos de R**
× 3/4 ([03-modelos.md §2.2](03-modelos.md#22-como-o-imagenet-vira-4-canais-)).
**Ação:** ablação com normalização (RGB com média/desvio do ImageNet; alpha em [0,1] ou centrado em 0).

### PD-37
**🟡 As reduções de resolução não usam antialias.** ✅ (medido)
O `scribbleprompt.rescale_inputs` faz `F.interpolate(mode='bilinear')` sem antialias; o `MarkerStep` faz o mesmo para 256².
Numa redução bilinear sem antialias, cada pixel de saída lê só 2×2 pixels de entrada: **1000→128 lê 6,6% dos pixels** da
imagem e do marcador; 1000→256 lê 26%. O teto de Dice em 128² cai de 0,921 (com antialias) para 0,875.
**Correção:** reduzir para 128² no wrapper, **antes** de chamar `rescale_inputs` (que então não reescala), com
`F.interpolate(..., mode='bilinear', antialias=True)` ou `mode='area'`. Fazer o mesmo no `MarkerStep`.

**✅ Encerrada (2026-09-27): a correção proposta piora os resultados.** No oráculo (treino, 128²):

| Variante | GT, negativo atual | miolo, só positivo | centros, só positivo |
|---|---|---|---|
| atual (sem antialias) | **0,873** | **0,668** | **0,342** |
| antialias na imagem e nos scribbles | 0,718 | 0,417 | 0,536 ⚠️ |
| antialias só na imagem | 0,876 | 0,637 | 0,326 |
| antialias + scribbles binarizados de novo | 0,872 | 0,493 | 0,024 |

⚠️ O 0,536 vem de segmentação em excesso: massa 2,9× a do GT, precisão 0,38.

**Por quê:** com antialias, os scribbles deixam de ser 0/1 e viram um "borrão" de valores intermediários espalhado pela
imagem, fora da distribuição de treino do ScribblePrompt. Com o só positivo, a rede chega a marcar a imagem inteira (massa
4,7×). Binarizar de novo faz os marcadores pequenos sumirem. Na imagem, o antialias não muda quase nada. O teto calculado
antes (0,921) valia para uma **máscara** reduzida, não para um **prompt**. A redução atual, mesmo "errada", preserva os
valores 0/1 que a rede espera. **Não mudar.** O ganho real de resolução vem de subir a entrada para 256² (PD-05).

### PD-38
**🟡 24,4 M parâmetros treinados com 30 imagens, sem congelar nada.** ✅
A `MarkerUNet` inteira (encoder ImageNet + decoder aleatório; 24.439.505 parâmetros) é otimizada com lr 1e-4 sobre 8
batches de 4 imagens. O exp. 5 (500 épocas) mostra *overfitting*: o termo de soft Dice no treino termina em 0,149 (soft Dice ≈ 0,85), enquanto o
Dice binário na validação fica em 0,598.
**Opções:** congelar o encoder nas primeiras épocas, ou usar lr menor para ele; encoder mais leve (ResNet18, como na prova de
conceito da tese); mais amostras via recortes (PD-05); aumentação por época (PD-08).

**Decisão (2026-09-27): fazer a ablação da PD-36 e da PD-38 juntas.** Elas se afetam: um encoder congelado depende mais da
normalização certa. Plano, com a configuração do exp. 4 como base e mudando **um fator por vez** (mesma seed, mesmas épocas):

| Variante | Normalização ImageNet | Encoder |
|---|---|---|
| A (base, = exp. 4) | não | treinável, lr 1e-4 |
| B | **sim** | treinável, lr 1e-4 |
| C | sim | **congelado** o treino todo |
| D | sim | congelado nas primeiras N épocas, depois treinável |
| E | sim | treinável com **lr menor** (ex.: 1e-5; decoder 1e-4) |

Medir o Dice e o IoU de validação, além do *gap* treino × validação (overfitting), sempre ao lado do Cellpose (PD-01).
De preferência com a validação cruzada da PD-02, para não escolher a variante pelo conjunto de teste.
**O que muda no código:** normalizar o RGB no `MarkerStep` (ou na `MarkerUNet`) com média/desvio do ImageNet, deixando o
canal alpha em [0,1]; uma opção de congelar o encoder (`model.encoder.requires_grad_(False)`); e *param groups* no otimizador
para lr diferente por parte da rede.

### PD-39
**⚪ Ideia adiada: ScribblePrompt-SAM como rede final.** ✅ (conferido em `scribbleprompt/models/sam.py`, commit `182c449`)
- Vantagem: imagem em 1024² e máscara em 256², contra 128² da UNet.
- Obstáculo: `predict` converte os scribbles em **coordenadas de cliques** (`scribbles_to_clicks`), o que **não é
  diferenciável** em relação ao marcador. Além disso, `predict` é `@torch.no_grad()`.
- Caminho possível: entregar o marcador pelo `mask_input` (256², contínuo e diferenciável). É fora da distribuição de treino,
  porque esse canal recebe a predição anterior. O encoder ViT-b é pesado, mas a imagem não muda, então os *embeddings* podem
  ser calculados uma vez por imagem.

**Decisão (2026-09-27):** deixado de lado por enquanto. Reavaliar depois do oráculo (PD-03).

### PD-40
**🔴 Com marcadores do tipo da tese, o ScribblePrompt não alcança o Cellpose, nem com o marcador ideal.** ✅ (oráculo, 2026-09-27)
> Explicação em linguagem simples: [03-modelos.md §8.1](03-modelos.md). Abaixo, os números completos e os caminhos possíveis.

Dice no **treino**, sem MarkerUNet, com marcadores derivados do GT. Linha de base: **Cellpose sozinho = 0,802**.

| Marcador | Área / GT | 128² atual | 128² só pos. | 256² atual | 256² só pos. | 512² atual | 512² só pos. |
|---|---|---|---|---|---|---|---|
| GT inteiro | 100% | 0,873 | 0,796 | **0,931** | **0,851** | 0,899 | 0,690 |
| miolo (metade interna de cada componente) | 33% | 0,452 | 0,668 | 0,532 | 0,731 | 0,507 | **0,784** |
| centros (disco de 4 px por componente) | 9% | 0,088 | 0,342 | 0,150 | 0,482 | 0,157 | 0,573 |
| máscara do Cellpose | 79% | 0,762 | 0,752 | 0,799 | 0,796 | 0,792 | 0,749 |

"Atual" = negativo atual (complemento denso); "só pos." = só scribbles positivos. No **teste**, o padrão é o mesmo
(Cellpose 0,810). Com só positivo, em 128² / 256² / 512²: miolo 0,703 / 0,755 / **0,805**; centros 0,442 / 0,587 / 0,611;
GT inteiro 0,790 / 0,837 / 0,718. Com negativo atual, o GT inteiro dá 0,872 / 0,933 / 0,935. No teste em 512², o miolo fica a
0,005 do Cellpose. Os valores foram conferidos numa execução completa do notebook (a mesma versão que está no repositório).

**Efeito da resolução:** subir de 128² para 256² melhora tudo. Em 512², o comportamento muda de lado:
- marcadores pequenos crescem mais (miolo 0,784, perto do Cellpose; centros 0,573);
- marcadores grandes passam a vazar: o GT inteiro com só positivo vai a 0,690, com massa 1,95× a do GT.

Há um compromisso: quanto maior a resolução, mais a rede expande a partir do prompt.

**Leituras:**
1. **Só o GT inteiro passa do Cellpose.** Marcadores pequenos e internos, que são o tipo que a tese defende e que o L_size
   empurra, ficam abaixo da linha de base mesmo sendo perfeitos. Com o miolo, o melhor caso é 0,731.
2. Então, para o pipeline superar o Cellpose, a MarkerUNet precisaria produzir **praticamente a segmentação completa** como
   marcador. Nesse caso ela já é a rede de segmentação, e o ScribblePrompt só acrescenta perda de resolução. A ideia central
   da tese (um marcador pequeno que a camada de segmentação **expande** até o objeto) não se realiza com o ScribblePrompt neste
   dado. No FMBS, quem faz essa expansão é a hierarquia.
3. Entregar a máscara do Cellpose ao ScribblePrompt **piora** em 128² (0,762) e empata em 256² (0,799). Uma MarkerUNet que
   "só copie" a entrada já parte abaixo ou no mesmo nível da linha de base.
4. O exp. 4 (Dice 0,648 no teste, com negativo atual em 128²) fica **entre** dois oráculos nas mesmas condições: acima do
   miolo (0,483) e abaixo do GT inteiro (0,872). Os marcadores aprendidos já são maiores que o miolo, o que é coerente com a
   leitura 2: para subir o Dice, a rede é empurrada a marcar o núcleo inteiro.

**Ressalvas:** os oráculos são binários e derivados de componentes do GT, e núcleos colados viram um componente só (PD-07).
Houve uma única interação (sem `mask_input`), e uma MarkerUNet treinada pode achar padrões difusos melhores do que esses
oráculos. O melhor oráculo "de marcador" (miolo, só positivo, 512²: 0,784) fica **perto** do Cellpose. Com ajustes (tamanho do
miolo, interação iterativa), talvez passe, mas a margem é pequena.

**Decisão (2026-09-27):** rodar primeiro os experimentos planejados ([06-experimentos.md §5](06-experimentos.md)); a escolha
do caminho é do autor. O oráculo entra como **resultado do trabalho**, a ser publicado (PD-47).

**Caminhos possíveis:**
- (a) Aceitar marcadores grandes: tirar ou reduzir o L_size, subir o ScribblePrompt para 256² e usar só positivos. O teto sobe
  para 0,85–0,93, mas o "marcador" deixa de ser marcador no sentido da tese.
- (b) Interação iterativa: usar a saída como `mask_input` por 2 a 3 passos, que é como o ScribblePrompt foi treinado.
- (c) Outra camada de segmentação que **expanda** marcadores pequenos: o próprio FMBS, um watershed diferenciável ou uma
  camada baseada no Cellpose.
- (d) Reformular o objetivo: usar a MarkerUNet para **corrigir** a máscara do Cellpose (onde adicionar ou remover), com o
  ScribblePrompt como refinador.

### PD-41
**🟠 O `BorderTerm` pune marcadores sobre núcleos reais.** ✅ (medido)
[border_loss.py](../../src/losses/border_loss.py) pune qualquer marcador nos `border_size=50` px de cada borda. Em 1000², essa
faixa é **19% da imagem** e contém **16,4% (treino) e 17,3% (teste) dos pixels de núcleo do GT** (mínimo 12%, máximo 21% por
imagem). O MoNuSeg é um recorte de lâmina, com núcleos até a borda, e o termo empurra contra o Dice em ~1/6 dos núcleos. Foi
usado nos exp. 3 e 6 com peso 0,5. ❓ A origem provável é o viés de imagens naturais, com o objeto centralizado (tese: marcador
de fundo nos cantos).
**Opções:** remover o termo; reduzir a faixa para poucos pixels (o artefato de borda de uma convolução costuma ter a largura do
campo receptivo das primeiras camadas); ou puni-lo só **fora** do GT, o que exige passar o GT ao termo.
**Decisão (2026-09-27):** o orientador aplicou o termo e o autor seguiu; o motivo não é conhecido. Ele fica **fora** do
experimento-base (PD-44) e só volta se mostrar ganho numa ablação.

### PD-42
**⚪ Docstring do `LossComposer` desatualizada.** ✅
[loss_composer.py:22-25](../../src/losses/loss_composer.py#L22-L25) diz que `ctx["prediction"]` é "o alvo dos termos …
(Dice/RMSE/Size)". O `SizeTerm` lê `ctx["markers"]` ([terms.py:28-29](../../src/losses/terms.py#L28-L29)). A docstring de
`ObjectSizeLoss` ("penaliza o desvio de tamanho") também está errada (PD-11). Corrigir as duas juntas.

### PD-43
**🟡 Perdas instáveis com GT vazio (risco latente).** ✅
- `DistanceMapLoss` divide por `Σg` ([distance_map_loss.py:49-50](../../src/losses/distance_map_loss.py#L49-L50)).
- `TotalVariationLoss` divide por `√Σg` ([total_variation_loss.py:38](../../src/losses/total_variation_loss.py#L38)).

Com um batch sem nenhum pixel de núcleo, as duas dão `inf`/`NaN`. O `ObjectSizeLoss` trata o caso, mas devolve a massa **sem
normalizar**, milhões de vezes maior que o valor normal. Com as imagens inteiras do MoNuSeg isso não acontece, porque toda imagem
tem núcleos e a soma é sobre o batch. Passa a acontecer se o projeto usar **recortes** (ideia da PD-05).
**Correção:** `normalization.clamp_min(1.0)` (ou ε) nos três, e decidir o que um recorte sem núcleos deve custar.

### PD-44
**🟠 Plano do experimento-base "só Dice + TV".** Decidido pelo autor em 2026-09-27 ([04-losses.md §4 e §6](04-losses.md)).
**Objetivo:** medir o **teto prático** do pipeline sem as regularizações que puxam contra o Dice (Size, DMap, Border; §4 do
tema 4). Depois, reintroduzi-las uma a uma para ver quanto cada uma custa.

| Item | Valor no experimento-base | Origem da decisão |
|---|---|---|
| Loss de segmentação | **Dice ou RMSE, não os dois** (sugestão: começar pelo Dice, que é o da tese) | pergunta 2 do tema 4 |
| Regularização | só **TV** (peso pequeno, ex.: 0,001) | pergunta 4 do tema 4 |
| Size, DMap, Border | **fora** | PD-40, PD-41, PD-06 |
| Scribbles | **só positivos** (canal negativo zerado) | PD-04 |
| Entrada do ScribblePrompt | **256²** | PD-05 |
| Resto | igual ao exp. 4 (lr, épocas, batch, seed) | comparabilidade |
| Linha de base | Cellpose sozinho, no mesmo conjunto | PD-01 |
| Validação | de preferência k-fold nas 30 de treino | PD-02 |

**Sequência sugerida depois:**
1. base;
2. base + DMap **com o mapa corrigido** (PD-06);
3. base + Size;
4. base + ablação de normalização/encoder (PD-38).

**O que muda no código:** o modo "só positivo" no `ScribblePromptingNetwork` (novo `scribble_mode`, ou montar `[m, 0]`) e um
parâmetro de `input_size`. Nada nas losses.

### PD-45
**⚪ As médias por época são por batch, não por imagem.** ✅
[training_loop.py:139-160](../../src/training/training_loop.py#L139-L160) soma a loss de cada batch e divide pelo número de
batches. Com 30 imagens e batch 4 (7 × 4 + 2), o último batch, de 2 imagens, pesa como os de 4; na validação, 14 = 4 + 4 + 4 + 2.
O efeito é pequeno, mas se soma às normalizações por batch das losses (PD-12): a mesma imagem contribui diferente conforme o
batch em que cai.
**Correção:** ponderar pela quantidade de imagens do batch, ou usar `drop_last` no treino.

### PD-46
**🟡 Não há seleção do melhor modelo nem early stopping.** ✅
O `TrainingLoop` não guarda o melhor estado, e não existe `EarlyStoppingCallback` nem `CheckpointCallback` (o antigo
`ARCHITECTURE.md` citava os dois). O modelo avaliado no fim é o da **última época**. Os notebooks imprimem "melhor val_loss:
X (época N)", mas esse **não** é o modelo cujas métricas são mostradas. O exp. 5 (500 épocas) mostra que treinar demais faz
*overfitting* (PD-38).
**Decisão (2026-09-27): sim para os dois.** Guardar o checkpoint de melhor Dice na validação do *fold* e usar *early stopping*
por ela. Entra no módulo da PD-19.
**Proposta:** um callback que guarde o estado de **melhor Dice na validação do *fold*** (nunca no teste, PD-02) e, opcionalmente,
pare depois de N épocas sem melhora. Como o treino custa ~1,5 s por época, o custo é irrelevante.

### PD-47
**🟡 Resultados precisam ser salvos e reproduzíveis para a publicação.** Decisão do autor em 2026-09-27: o trabalho deve ser
publicado, e o oráculo entra como resultado ([06-experimentos.md §6](06-experimentos.md)).
**Situação hoje:**
- o notebook do oráculo **só imprime** as tabelas, sem salvar CSV nem figuras;
- os números atuais vieram de uma execução em CPU, fora do Colab;
- os experimentos de treino não salvam configuração, commit nem checkpoint (PD-10).

**O que falta para os resultados valerem como material de artigo:**
1. **Oráculo:** rodar no Colab e salvar a tabela por imagem (`df`) em CSV, junto com o commit (`git rev-parse HEAD`), o hash do
   checkpoint (PD-15) e as versões de `torch`/`scribbleprompt`. Guardar também a figura de exemplo.
2. **Próximos experimentos:** para cada run, salvar a configuração completa (JSON), o commit, o checkpoint do melhor modelo e as
   métricas **por imagem e por *fold***. Isso entra no módulo da PD-19.
3. **Linha de base:** relatar o Cellpose, com os parâmetros usados, em todas as tabelas (PD-01).
4. **Estatística:** com 14 imagens de teste, relatar média **e dispersão** (desvio ou intervalo de confiança por *bootstrap*
   sobre as imagens), e, na comparação com o Cellpose, um teste pareado por imagem (ex.: Wilcoxon). ❓ Confirmar com o orientador
   o padrão esperado pelo veículo de publicação.
5. **Figuras candidatas já produzidas:** `docs/estudo/figuras/dmap_normalizacao.png` (mapa de distância atual × por núcleo).

### PD-48
**🟡 Os testes não verificam a parte científica e não rodam automaticamente.** ✅ (executados em 2026-09-27)
Resultado: **74 testes, 70 passam, 3 falham (PD-11), 1 pulado**, em ~3 s em CPU. A cobertura de linhas é de 75%, mas só dos
arquivos importados.
- **Nunca importados (0%):** `monuseg_dataset.py` (XML → máscara), `cellpose_step.py`, `marker_unet.py`,
  `preprocessing_pipeline.py`, `model_pipeline.py`.
- **Só com dublês:** MarkerStep, FrozenSegmentationStep, ScribblePrompt (pacote falso) e Trainer.
- **Sem conferência de valor:** Dice, RMSE, DMap, TV, Border e Size. As linhas são executadas, mas o resultado não é comparado.
- **Sem teste nenhum:** o código dos notebooks (carregamento, aumentação, métricas), que é o que gera os números do trabalho.
- **Infraestrutura:** `pytest` fora do `requirements.txt`, sem CI e sem ambiente local. Por isso o PD-11 ficou um mês sem ser notado.

Detalhes, testes que precisam mudar com as correções decididas e lista de testes a escrever em [07-testes.md](07-testes.md).

### PD-49
**🟠 No modo só positivo, qualquer fundo no canal positivo faz o ScribblePrompt segmentar a imagem inteira.** ✅ (medido em 2026-09-28)
No oráculo, o positivo era o marcador binário **cru** (0 fora dele). Com o *sharpening* atual (`sigmoid(10·(s − 0,5))`), o fundo
vale 0,0067 em todo pixel, e o Dice cai para **0,387 com qualquer marcador**: GT inteiro, miolo ou centros (30 imagens de treino,
256²; massa 4,7× a do GT). Esse valor é o Dice de marcar a imagem toda. Com um fundo constante `c`, até 1e-4 nada muda, 1e-3 já faz
vazar (massa 1,3–2,5×) e ≥ 3e-3 inunda. No modo atual isso não aparece, porque o negativo denso (0,993) compensa.
**Por que importa:** a PD-44 (experimento-base só positivo) usa a saída sigmoid da MarkerUNet, que nunca é exatamente 0. Sem
tratar isso, o experimento-base tende a inundar.
**Opções** (detalhe e tabela em [09-correcoes-pontuais.md §9](09-correcoes-pontuais.md)): (a) limiar suave
`relu(s − τ)/(1 − τ)`; (b) binário com *straight-through*; (c) cru, com risco de inundar. Recomendação: (a).

### PD-50
**⚪ O código cita a história do projeto em vez de só descrever o que faz.** ✅ (contado em 2026-10-01)
**Regra do autor (2026-10-01):** quem clonar o repositório precisa saber **como o código funciona**, não que um *fallback* existiu
ou foi removido numa data. Comentários, docstrings, testes e textos de notebook descrevem o comportamento e o porquê do desenho;
a história (pendências, datas, "antes era…", itens da investigação) fica nos `docs/estudo/` e no git.
**Situação:**
- Já seguem a regra: `monuseg_dataset.py`, `cellpose_step.py`, os testes deles, o `requirements.txt` e o notebook de
  pré-processamento (os trechos alterados em 2026-10-01).
- **Faltam 37 menções em `src/` e `tests/`:** `test_training_integration.py` (14), `trainer.py` (5),
  `scribble_prompting_network.py` (4), `marker_step.py` (3), e 2 ou menos em `test_scribble_prompting_network.py`,
  `grad_norm_callback.py`, `save_results_step.py`, `terms.py`, `loss_composer.py` e `test_object_size_loss.py`. São do tipo
  "investigação, P5", "C2", "C3/C4", "regra arquitetural 23".
- **Notebooks:** 43 a 49 menções em cada `experiment_1..6` (vão ser podados depois das correções, PD-10), 14 no oráculo e 7 no
  restante do notebook de pré-processamento (por exemplo, "(P7)").
**Proposta:** uma tarefa própria para limpar `src/` e `tests/` (sem mudar comportamento), reescrevendo cada menção como explicação
do comportamento quando ela carrega informação útil. Os notebooks de experimento ficam para a poda da PD-10. ⚠️ O `CLAUDE.md` e o
02 §4.2/§7 pediam para manter válidas as referências "investigação, P3" e "regra 23" do código; com esta regra, elas deixam de ser
mantidas e passam a ser removidas.
