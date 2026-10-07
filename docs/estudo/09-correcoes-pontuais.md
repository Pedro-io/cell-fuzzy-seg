# 09 — Correções pontuais: base e registro

> **Para que serve:** antes de mexer no código, este documento reúne, para cada correção, **o problema**, **a base** (código,
> biblioteca, tese ou medição) e **a mudança proposta**, com o código atual e o proposto lado a lado. Depois da sua revisão,
> cada correção aprovada é aplicada num commit próprio, e o §11 registra o commit e o resultado dos testes.
> A pendência correspondente em [08-pendencias.md](08-pendencias.md) vira ✅ com data e commit.
>
> Legenda: ✅ verificado no código ou medido · 📜 histórico/commits · 📖 tese ou artigo · ❓ inferência a confirmar.
>
> **Estado de partida (2026-09-28):** commit `b865652`. Testes executados em CPU, no venv do scratchpad (torch 2.14 CPU, smp 0.5.0,
> scribbleprompt `182c449`, pytest 9.1): **70 passam, 3 falham (PD-11), 1 pulado**. ✅

---

## 1. Escopo

**Critério de "pontual":**
1. a mudança fica em poucas linhas e num lugar só;
2. não exige GPU nem Colab, nem regerar `MoNuSegPreprocessed/`;
3. não muda nenhum número já registrado nos documentos, ou só muda o que é intencional;
4. a correção já foi decidida, ou não há alternativa real.

| # | Pendência | Correção | Muda algum resultado? | Testes | Situação da decisão |
|---|---|---|---|---|---|
| C1 | PD-11, PD-42 | testes e docstrings do Size e do `LossComposer` seguem o L_size da tese | não (o cálculo não muda) | reescreve os 4 de `test_object_size_loss.py` | fórmula já decidida |
| C2 | PD-14 | registrar a norma do gradiente **antes** do clip | não (só o diagnóstico) | +1 | proposta |
| C3 | PD-15 | conferir o SHA-256 do checkpoint do ScribblePrompt antes de carregar | não (o hash confere) | +3 | proposta; hash registrado |
| C4 | PD-29 | checagem do nome do modelo do Cellpose que funciona no Cellpose 4 | não | +3 (Cellpose falso) | proposta |
| C5 | PD-43 | denominadores com `clamp_min(1.0)` em Size, DMap e TV | não (ΣGT ≥ 103.258 por imagem) | +3 | proposta |
| C6 | PD-16 | *fallbacks* silenciosos: `MarkerStep` e notebook de pré-processamento | não (nenhum run usou) | muda 1, +1 | proposta |
| C7 | PD-17, PD-20, PD-31 | remover código morto e arquivos gerados que estão no git | não | nenhum teste usa esse código | remoção já decidida |
| C8 | PD-44 (código), PD-49 (nova) | modo "só positivo" e `input_size` configurável no ScribblePrompt | não (os padrões continuam os mesmos) | +4 | ⚠️ depende da sua decisão sobre a PD-49 |

**Fora desta rodada, e por quê:**

| Pendência | Motivo |
|---|---|
| PD-33 (renomear `segmentation`) | Mexe no `Trainer` e quebra a reexecução dos notebooks 1 a 6, que a PD-10 manda não corrigir agora. Sugestão: fazer junto com o módulo da PD-19, quando o notebook novo nascer. |
| PD-45 (médias por época), PD-18 (`datasets.yml`) | Vão para o módulo da PD-19: é lá que as métricas por imagem e a configuração real vão existir. |
| PD-06, PD-25, PD-30 | Mudam o GT, o Dmap ou a máscara do Cellpose, o que exige regerar os dados e muda todos os números. Cada um é um passo do plano, não uma correção pontual. |
| PD-12 (normalização por batch) | Muda o significado das losses e ainda não tem decisão. |
| PD-27 (licença do código) | A escolha da licença é sua. |
| PD-32, PD-35 (arquitetura) | Sem decisão; não afetam resultados. |

---

## 2. C1 — Testes e docstrings do Size (PD-11) e do `LossComposer` (PD-42)

### Problema ✅
- [test_object_size_loss.py](../../tests/test_object_size_loss.py) testa o Size **simétrico** `peso·|ratio − 1|`. Três dos quatro
  testes falham (executado hoje). O `test_gradient_pushes_toward_gt_size` espera gradiente negativo e recebe **+0,0625** em
  cada pixel, que é exatamente `peso/ΣGT = 1/16`.
- [object_size_loss.py:8](../../src/losses/object_size_loss.py#L8) diz "penaliza o desvio de tamanho", mas o código calcula
  `peso·Σmarcador/ΣGT`, que não mede desvio nenhum.
- [loss_composer.py:22-25](../../src/losses/loss_composer.py#L22-L25) diz que `ctx["prediction"]` é o alvo de
  "Dice/RMSE/Size". O `SizeTerm` lê `ctx["markers"]` ([terms.py:28-29](../../src/losses/terms.py#L28-L29)).

### Base
- 📖 O L_size da tese (eq. 5.5) é `Σμ/ΣGT`, sem módulo. O mínimo é o marcador vazio, e quem impede o colapso é o L_seg. A ideia é
  que o marcador seja **o menor possível** que ainda produza a segmentação certa.
- 📜 A versão simétrica (P10) foi revertida de propósito em `4c4db98` (30/08). Decisão registrada no `CLAUDE.md`: não voltar para `|ratio − 1|`.
- ✅ Portanto o **código está certo**, e os testes e as docstrings estão errados. O gradiente do L_size em relação a cada pixel
  do marcador é a constante `peso/ΣGT > 0`: o termo sempre empurra o marcador para baixo, com a mesma força em todo pixel.

### Mudança proposta
Testes novos, na mesma ordem dos atuais:

| Atual (fórmula simétrica) | Proposto (L_size da tese) |
|---|---|
| `test_loss_is_zero_when_sizes_match`: pred = GT → 0 | `test_loss_is_mass_ratio`: pred 0,5 em 4×4, GT 1 → `0,1·8/16 = 0,05` |
| `test_loss_penalizes_overprediction_symmetrically`: 1,5× e 0,5× dão o mesmo | `test_loss_is_linear_not_symmetric`: 1,5× → 0,15 e 0,5× → 0,05 |
| `test_gradient_pushes_toward_gt_size`: gradiente < 0 | `test_gradient_is_constant_and_positive`: gradiente = `peso/ΣGT` em todo pixel |
| `test_empty_gt_penalizes_predicted_mass` | igual (continua valendo, inclusive depois da C5) |
| — | `test_normalizes_by_whole_batch`: 2 imagens com GT de tamanhos diferentes → `Σm_total/ΣGT_total`. Documenta o comportamento da PD-12; se ela mudar, este teste muda junto. |

Docstrings:

```python
# atual (object_size_loss.py:8)
"""Penaliza o desvio de tamanho entre a máscara prevista e o ground truth."""

# proposto
"""Regularização de tamanho do marcador (L_size da tese, eq. 5.5): ``peso·Σmarcador/ΣGT``.

Não é simétrica: o mínimo é o marcador vazio, e quem impede o colapso é o termo de
segmentação. A soma é sobre o batch inteiro, não por imagem (PD-12).
"""
```

No `LossComposer`, a lista passa a dizer: `prediction` → Dice e RMSE; `markers` → Size, DMap, TV, Border e NotTooThin.

**Efeito nos resultados:** nenhum. **Documentos a atualizar:** 07-testes (§1, §3, §6), 04-losses (§2.2, §3.3 e §5), `CLAUDE.md` (contagem dos testes).

---

## 3. C2 — Norma do gradiente antes do clip (PD-14)

### Problema ✅
Em [trainer.py:178-188](../../src/training/trainer.py#L178-L188) a ordem é `backward` → `clip_grad_norm_` → `step` →
callbacks. O `GradNormCallback` ([grad_norm_callback.py:117-126](../../src/training/callbacks/grad_norm_callback.py#L117-L126))
lê `p.grad` depois que o clip já reescalou os gradientes. Com `grad_clip=1.0`, sempre que a norma passa de 1 o callback
registra **1,0**, e não dá para saber qual era a norma de verdade.

### Base
- ✅ `torch.nn.utils.clip_grad_norm_` **devolve a norma total calculada antes do clip** (docstring no torch 2.14: "Total norm
  of the parameter gradients (viewed as a single vector)"). Medido: um gradiente de norma 20 com `max_norm=1` → a função
  devolve **20,0**, e a norma depois fica em 1,0. O valor certo já existe, sem custo extra.
- 📜 Por que importa: o callback nasceu para o diagnóstico E1 (gradiente morto, P1), e o clip é a correção P5. Para saber se o
  clip atua **sempre** (o que muda o passo efetivo do otimizador) ou só de vez em quando, é preciso a norma antes do clip.

### Mudança proposta

```python
# atual (trainer.py:179-181)                      # proposto
if self.grad_clip is not None:                    self.last_grad_norm = None
    params = [...]                                if self.grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(                   params = [...]
        params, self.grad_clip)                       self.last_grad_norm = float(
                                                          torch.nn.utils.clip_grad_norm_(params, self.grad_clip))
```

No `GradNormCallback`, uma chave nova `history["total_pre_clip"]`: é o `trainer.last_grad_norm` quando há clip e o próprio
`total` quando não há (sem clip, os dois são iguais). O log passa a mostrar os dois valores.

**Por que não renomear `total`:** os notebooks 1 a 6 plotam `grad_norm_cb.history["total"]`. Mantendo a chave e documentando
que ela é "depois do clip", eles continuam rodando.

**Efeito nos resultados:** nenhum. O valor devolvido só é lido, e o `.item()` por passo já é feito hoje pelo callback.
**Teste novo:** com `grad_clip` minúsculo (1e-6), `total_pre_clip` > `total` ≈ 1e-6.
**Documentos:** 05-treinamento §2.3, 00-visão geral §7 (fato "o `GradNormCallback` mede depois do clipping").

---

## 4. C3 — Conferir o SHA-256 do checkpoint do ScribblePrompt (PD-15)

### Problema ✅
- [download_checkpoint](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L359-L386) baixa o `.pt`
  de um link do Dropbox e não confere nada. Se o arquivo já existe, ele o devolve sem conferir.
- O arquivo é desserializado no **construtor**: `ScribblePromptUNet(...)` → `build_model` → `torch.load(f, map_location=...)`
  (`scribbleprompt/models/unet.py:40-41`, commit `182c449`), **sem** `weights_only`.

### Base
- **Segurança.** `torch.load` usa pickle. ✅ No torch ≥ 2.6, o padrão virou `weights_only=True` (mensagem em
  `torch/serialization.py`), que recusa objetos arbitrários. Mas o [requirements.txt](../../requirements.txt) aceita
  `torch>=2.1.0`, e entre 2.1 e 2.5 o padrão é `False`: um arquivo trocado na origem **executa código** ao ser carregado. O
  HTTPS protege o caminho, mas não uma troca do arquivo no Dropbox.
- **Reprodutibilidade (PD-47).** O `weights_only` não detecta um arquivo **válido, mas diferente**: outros pesos carregariam
  sem erro e mudariam os resultados. O hash cobre os dois riscos.
- ✅ **Hash de referência:** `43f57ee8…acbe` (PD-15, download de 2026-09-27). Conferido de novo hoje no arquivo que o oráculo
  usou: igual. ❓ Falta conferir contra a saída do Colab (o notebook do oráculo imprime o hash).

### Mudança proposta
A conferência tem que acontecer **antes do `torch.load`**, ou seja, no construtor, e não só no download:

```python
# proposto (scribble_prompting_network.py)
CHECKPOINT_SHA256: Dict[str, str] = {
    "v1": "43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe",
}

@classmethod
def verify_checkpoint(cls, path: str, version: str = "v1") -> None:
    """Levanta RuntimeError se o SHA-256 do arquivo não for o esperado."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):   # lê em blocos de 1 MB
            digest.update(chunk)
    if digest.hexdigest() != cls.CHECKPOINT_SHA256[version]:
        raise RuntimeError(f"Checkpoint do ScribblePrompt com hash inesperado ({digest.hexdigest()}). ...")
```

| Onde | Atual | Proposto |
|---|---|---|
| `__init__` | instancia `ScribblePromptUNet` direto | se o arquivo existe, `verify_checkpoint` antes; se não existe, o erro atual ("baixe com `download_checkpoint`") continua |
| `download_checkpoint`, arquivo já existe | devolve o caminho | confere; se não bater, levanta erro **sem apagar** (o arquivo é do usuário) |
| `download_checkpoint`, depois de baixar | devolve o caminho | confere; se não bater, **apaga** o arquivo e levanta erro |

**Opcional (C3b):** subir o mínimo para `torch>=2.6` no `requirements.txt`, como segunda camada. O Colab já usa uma versão mais
nova, e nenhuma dependência fixada exige menos (❓ conferir o `cellpose==4.1.1` ao aplicar).

**Efeito nos resultados:** nenhum, porque o hash confere. **Testes novos** (com o pacote falso atual): arquivo com hash errado →
`RuntimeError` **antes** de instanciar a rede; arquivo certo (hash esperado trocado via `monkeypatch`) → passa; download de
um arquivo existente com hash errado → erro.
**Documentos:** 03-modelos §3.3, `CLAUDE.md` (fato "Segurança: … sem checar hash").

---

## 5. C4 — Checagem do nome do modelo do Cellpose (PD-29)

### Problema ✅
[cellpose_step.py:77-80](../../src/pipeline/steps/preprocessing/cellpose_step.py#L77-L80) importa `MODEL_LIST`, que não existe
no Cellpose 4.1.1. O `except` devolve em silêncio, e o aviso nunca aparece.

### Base ✅ (código do `cellpose==4.1.1`, `cellpose/models.py`, baixado do PyPI hoje)
- `MODEL_NAMES = ["cpsam"]` (L27), e `get_user_models()` (L57-65) lê os modelos que o usuário registrou em `gui_models.txt`.
- A regra do próprio Cellpose (L130-140): o nome é aceito se for **um arquivo que existe** ou se estiver em
  `MODEL_NAMES + get_user_models()`. Senão, ele usa o `cpsam` e loga um aviso.
- **O aviso da biblioteca engana:** a variável já foi trocada pelo caminho do modelo padrão antes do log. A mensagem sai como
  `pretrained model ~/.cellpose/models/cpsam not found, using default model`, com o caminho do padrão, e não com o nome pedido.
- Hoje o risco prático é pequeno, porque o único modelo embutido é o `cpsam`. Ele aparece quando alguém passa o **caminho** de
  um modelo próprio com erro de digitação: o Cellpose troca em silêncio pelo `cpsam`, e a linha de base (PD-01) muda sem aviso.

### Mudança proposta

```python
# atual                                              # proposto
try:                                                 from cellpose.models import MODEL_NAMES, get_user_models
    from cellpose.models import MODEL_LIST
except (ImportError, AttributeError):                if os.path.exists(pretrained_model):
    return                                               return
if pretrained_model not in MODEL_LIST:               if pretrained_model not in MODEL_NAMES + get_user_models():
    logger.warning(...)                                  raise ValueError(
                                                             f"Modelo do Cellpose desconhecido: {pretrained_model!r}. "
                                                             f"Disponíveis: {MODEL_NAMES + get_user_models()}.")
```

**Decisão do autor (2026-10-01): erro.** ✅ Aplicado (§11).

**Erro ou aviso?** Recomendo **erro**, pela mesma lógica da PD-16: um modelo trocado muda a entrada da MarkerUNet e a linha de
base, e um aviso no meio do log do Colab passa despercebido. Se preferir o aviso, basta trocar o `raise` por `logger.warning`.
O import passa a ser direto: se a API do Cellpose mudar de novo, o erro aparece em vez de sumir.

**Fica de fora:** o `diam_mean` passado ao construtor (ignorado no Cellpose 4) é assunto da PD-30.
**Efeito nos resultados:** nenhum (`"cpsam"` passa). **Testes novos:** hoje o `cellpose_step.py` não é importável sem o
Cellpose. Com um módulo `cellpose` falso em `sys.modules` (o mesmo truque dos testes do ScribblePrompt), dá para testar só a
checagem: `"cpsam"` passa; um arquivo existente passa; `"cpsam_v2"` levanta erro.

---

## 6. C5 — Denominadores protegidos contra GT vazio (PD-43)

### Problema ✅
`Σg` no denominador: [distance_map_loss.py:49-50](../../src/losses/distance_map_loss.py#L49-L50) e
[total_variation_loss.py:38](../../src/losses/total_variation_loss.py#L38) (`√Σg`). Com um batch sem nenhum pixel de núcleo,
dão `inf`/`NaN`. O Size tem um `if` próprio ([object_size_loss.py:32-35](../../src/losses/object_size_loss.py#L32-L35)).

### Base
- ✅ **Medido hoje:** a menor quantidade de pixels de núcleo numa imagem é **119.199** no treino e **103.258** no teste. Como a
  soma é sobre o batch, o denominador nunca fica abaixo de 103.258. Para `x ≥ 1`, `max(x, 1) = x`: **nenhum valor e nenhum
  gradiente dos experimentos muda.**
- **Por que 1 e não ε:** o GT é binário, então `Σg` é um inteiro e todo batch com algum núcleo já tem `Σg ≥ 1`. Com 1, o batch
  vazio custa a soma **sem normalizar**, que é o que o `if` do Size já faz hoje (`peso·Σm`). Com ε = 1e-6, esse custo seria
  multiplicado por um milhão.
- **O que continua em aberto:** quanto um recorte sem núcleos **deve** custar (ele deveria empurrar o marcador a zero com força?)
  só importa quando houver recortes (PD-05). A PD-43 fica como "resolvida quanto ao `NaN`; decisão de custo adiada para os recortes".

### Mudança proposta

```python
# DistanceMapLoss                                   # proposto
normalization = y_true.sum()                        normalization = y_true.sum().clamp_min(1.0)

# TotalVariationLoss
normalization = torch.sqrt(y_true.sum())            normalization = torch.sqrt(y_true.sum().clamp_min(1.0))

# ObjectSizeLoss
gt_sum = y_true.sum()                               ratio = y_pred.sum() / y_true.sum().clamp_min(1.0)
if gt_sum == 0:                                     return self.weight * ratio
    return self.weight * y_pred.sum().abs()
ratio = y_pred.sum() / gt_sum
return self.weight * ratio
```

O `.abs()` do Size some sem mudar nada, porque os marcadores saem de uma sigmoid e já são ≥ 0.
**Testes novos:** GT vazio → valor finito nas três losses; GT não vazio → o mesmo valor calculado à mão pela fórmula da tese.

---

## 7. C6 — *Fallbacks* silenciosos (PD-16)

### Problema ✅
- [marker_step.py:113-124](../../src/pipeline/steps/inference/marker_step.py#L113-L124): sem modelo, o marcador vira
  `segmentation > 0` (a máscara do Cellpose), só com um *warning*.
- [preprocessamento_monuseg_persistido.ipynb](../../notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb),
  célula 9: se o `CellposeStep()` falha, o notebook usa o **ground truth** como `segmentation`. O canal alpha passa a conter a resposta.

### Base
- ✅ Quem usa o fallback do `MarkerStep`: só `test_marker_step.py` e o `pipeline_test.py` (que sai na C7). Nenhum notebook cria
  `MarkerStep(model=None)`.
- ✅ Por que é perigoso: o erro de configuração não aparece como erro, e sim como **um número plausível**. Pelo oráculo, a máscara
  do Cellpose entregue ao ScribblePrompt dá Dice ~0,76 em 128² (PD-40), que pode ser confundido com um resultado. No
  pré-processamento é pior: com o GT no canal alpha, a MarkerUNet aprende a copiar a resposta.
- **Por que o notebook de pré-processamento entra, apesar da PD-10:** a PD-10 trata dos notebooks de experimento antigos. Este
  notebook vai ser **reexecutado em breve** para a PD-30 (diâmetro do Cellpose), e uma falha do Cellpose no Colab vazaria o GT
  para os dados novos sem ninguém perceber.

### Mudança proposta

| Onde | Atual | Proposto |
|---|---|---|
| `MarkerStep.__init__` | `model=None` aceito | `model=None` só com `allow_segmentation_fallback=True`; senão, `ValueError` já na construção |
| `MarkerStep.forward` | *warning* e segue | igual quando a flag está ligada (uso explícito, por exemplo em teste) |
| notebook, célula 9 | `try/except` → GT como `segmentation` | sem `try/except`: se o Cellpose não carrega, a célula falha. ✅ aplicado em 2026-10-01 (`9395df2`) |

**Efeito nos resultados:** nenhum. Os runs registrados usaram o Cellpose (`meta.json`: `segmentation_source: cellpose`).
**Testes:** o `test_forward_without_model_falls_back_to_segmentation` passa a ligar a flag; um novo verifica o `ValueError` sem ela.
O `DummyFinalNetwork` dos notebooks de experimento fica como está (PD-10).

---

## 8. C7 — Código morto e arquivos gerados no git (PD-17, PD-20, PD-31)

### Base ✅ (busca em `src/`, `tests/`, `notebooks/`, `configs/` e na raiz)

| Item | Onde | Quem usa |
|---|---|---|
| `pipeline_test.py` (PD-17) | raiz | ninguém; importa `src.pipeline.steps.cellpose_step`, `marker_step` e `segmentation_step`, que não existem mais. Os notebooks só o citam na saída de um `ls`. |
| `ModelPipeline` | [model_pipeline.py](../../src/pipeline/model_pipeline.py) | só o `pipeline_test.py`; citado numa docstring de [monuseg_preprocessed_dataset.py:85](../../src/data/load/monuseg_preprocessed_dataset.py#L85) |
| `OutputWriter.save_all`, `save_segmentation`, `save_markers`, `save_overlay`, `save_rgba`, `_colorize` (+ `seg_dir`, `marker_dir`, `overlay_dir`) | [output_writer.py:59-205](../../src/io/output_writer.py#L59-L205) | ninguém. O `SaveResultsStep` e o notebook usam só o `save_preprocessed`. |
| `to_uint8_rgb` | [image_utils.py:4-26](../../src/utils/image_utils.py#L4-L26) | ninguém |
| `RMSEAccuracy` | [rmse_accuracy.py](../../src/losses/rmse_accuracy.py), exportada em [losses/\_\_init\_\_.py:9](../../src/losses/__init__.py#L9) | ninguém |
| chaves `flows` e `styles` | [cellpose_step.py:120-122](../../src/pipeline/steps/preprocessing/cellpose_step.py#L120-L122) | o `SaveResultsStep` não as grava (as pastas persistidas são `distance_map`, `ground_truth`, `image`, `rgba` e `segmentation`). O notebook-tutorial `preprocessamento_monuseg.ipynb` lê `flows` com `if 'flows' in ...`, então não quebra. |
| 3 `.pyc` e 5 arquivos de `src/cell_fuzzy_seg.egg-info/` (PD-20) | git | gerados pelo Python e pelo `pip install -e .`; o `.gitignore` já tem `__pycache__/`, mas eles entraram antes |

**A decisão de remover** foi sua (02-pipeline, pergunta 3): "podemos marcar como remoção e depois, se precisar, implementamos".
Tudo continua recuperável com `git show b865652:<caminho>`.

**Um cuidado (PD-34):** a ideia da PD-34 usa `flows[2]`, a probabilidade contínua do Cellpose. Remover a chave `flows` não
atrapalha essa ideia. Se ela for adiante, o certo é o `CellposeStep` gravar só esse mapa, numa chave própria (por exemplo,
`cellpose_prob`), e não a lista inteira de fluxos, que ninguém usa.

### Mudança proposta
- apagar `pipeline_test.py`, `model_pipeline.py`, `rmse_accuracy.py` e `to_uint8_rgb`; enxugar o `OutputWriter` para o
  `save_preprocessed` (e tirar o `import cv2` e os tipos que ficarem sem uso);
- tirar `flows` e `styles` do `CellposeStep` (e da docstring);
- `git rm -r --cached` nos `.pyc` e no `egg-info`, e acrescentar `*.egg-info/` e `*.pyc` ao `.gitignore`.

**Efeito nos resultados:** nenhum. **Documentos:** 00-visão geral (§5, tabela de arquivos), 02-pipeline (§5.2, §5.5 e §6),
04-losses (§2.2, `RMSEAccuracy`) e `CLAUDE.md` (fatos sobre `pipeline_test.py`, `.pyc` e código morto).

---

## 9. C8 — Modo "só positivo" e resolução do ScribblePrompt (código da PD-44)

> ⚠️ Esta seção tem um achado novo que muda o desenho do experimento-base. Registrado como **PD-49** em 08-pendencias.

### Problema ✅
O experimento-base (PD-44) precisa de duas coisas que o wrapper não oferece:
1. **só scribbles positivos.** Os modos atuais são `sharpened`, com o negativo denso, e `dense_soft`
   ([scribble_prompting_network.py:345-357](../../src/models/networks/final_segmentation/scribble_prompting_network.py#L345-L357));
2. **entrada em 256².** O `input_size` vem fixo do pacote (128²). O oráculo trocava `net.input_size` à mão.

### Base
- 📜 Os números vêm do oráculo (PD-04, PD-05, PD-40): só positivo e 256² melhoram todos os marcadores.
- ✅ A U-Net do ScribblePrompt tem **4 max-poolings** (`scribbleprompt/models/network.py`, um por item de `features`), então a
  altura e a largura precisam ser **divisíveis por 16**.
- ✅ **Achado novo (medido hoje, 30 imagens de treino, 256²):** no oráculo, o canal positivo era o marcador **cru** (0 fora dele).
  A forma natural de implementar o modo seria reaproveitar o *sharpening*, `pos = sigmoid(10·(s − 0,5))`. Mas isso põe
  **0,0067 em todo pixel** fora do marcador, e o resultado desaba:

  | Marcador | positivo cru (= oráculo) | positivo com *sharpening* |
  |---|---|---|
  | GT inteiro | 0,851 | 0,387 |
  | miolo | 0,731 | 0,387 |
  | centros | 0,482 | 0,386 |

  O 0,387 é praticamente o Dice de marcar a **imagem inteira** como núcleo. Com o primeiro plano ocupando ~24,6% dos pixels,
  isso dá `2·0,246/1,246 ≈ 0,39`. No modo atual, isso não acontece porque o negativo denso (0,993) compensa.
- **Consequência para o experimento-base:** a MarkerUNet termina numa sigmoid, que **nunca dá exatamente 0**. Sem o negativo, o
  fundo baixo do marcador aprendido pode fazer o ScribblePrompt marcar tudo.
- ✅ **Tolerância medida (mesmas 30 imagens, 256², negativo zero; positivo = marcador binário + `c` constante fora dele):**

  | `c` fora do marcador | GT inteiro: Dice (massa/GT) | miolo | centros |
  |---|---|---|---|
  | 0 (= oráculo) | 0,851 (1,34) | 0,731 (0,61) | 0,482 (0,35) |
  | 0,0001 | 0,847 (1,36) | 0,734 (0,61) | 0,483 (0,36) |
  | 0,001 | 0,592 (2,47) | 0,665 (1,28) | 0,569 (1,31) |
  | 0,003 | 0,405 (4,48) | 0,399 (4,36) | 0,401 (4,39) |
  | 0,0067 (*sharpening* T=10) | 0,387 (4,72) | 0,387 (4,72) | 0,386 (4,73) |
  | 0,01 e 0,03 | 0,386 (4,73) | 0,386 (4,73) | 0,386 (4,73) |

  Até 1e-4, nada muda. Com 1e-3 a saída já cresce (massa 1,3 a 2,5× o GT), e a partir de 3e-3 a rede marca a imagem inteira. Nos
  centros, 1e-3 até **melhora** (0,569), porque o vazamento faz marcadores pequenos crescerem. Isso é sensibilidade, não um recurso
  confiável. Conclusão: no modo só positivo, o fundo precisa ser **< 1e-4, na prática zero**. Scripts:
  `modo_positivo.py` e `fundo_denso.py`, no scratchpad da sessão de 2026-09-28 (❓ ainda não versionados; ver PD-47).

### Mudança proposta (depende da sua decisão sobre a PD-49)
- parâmetro `input_size: Optional[Tuple[int, int]] = None` no construtor. `None` mantém os 128² do pacote; outros valores têm
  de ser divisíveis por 16 (senão, `ValueError`);
- `scribble_mode="positive"`: canal negativo zerado e canal positivo **exatamente 0 abaixo de um limiar**. Opções para a PD-49:

| Opção | Canal positivo | Vantagem | Custo |
|---|---|---|---|
| (a) limiar suave | `relu(s − τ)/(1 − τ)` | zero exato abaixo de τ, diferenciável acima; com marcador binário, é igual ao oráculo | pixels abaixo de τ não recebem gradiente direto |
| (b) binário com *straight-through* | `(s > 0,5)` na ida, identidade na volta | igual ao oráculo (0/1) | gradiente enviesado; o marcador deixa de ser difuso |
| (c) cru | `s` | o mais simples | risco de inundar, conforme a tolerância medida |

**Recomendação:** (a), com τ configurável e o padrão escolhido pela tabela da PD-49. Os padrões atuais (`sharpened`, 128²) não
mudam, então os notebooks antigos continuam iguais.
**Testes novos:** `input_size` não divisível por 16 → erro; `input_size=(256, 256)` chega à UNet em 256²; no modo `positive`,
o negativo é zero e o positivo é zero abaixo de τ; o gradiente chega aos scribbles acima de τ.

---

## 10. Ordem de aplicação (revista em 2026-09-29: seguir o fluxo dos dados)

**Regra:** corrigir na ordem em que os dados passam pelo pipeline. As etapas 1, 2 e 4 mudam os `.npy`. Corrigidas em sequência,
os dados são regerados **uma vez só**, na etapa 5; corrigidas fora de ordem, a regeração se repete. O GT vem antes do diâmetro do
Cellpose porque o Dice do Cellpose, que escolhe o diâmetro, é medido contra o GT. Isso muda a ordem de 27/09 (infraestrutura →
diâmetro → Dmap), que punha o diâmetro antes.

| Etapa | Onde | Pendências | Muda os `.npy`? | GPU? | Situação |
|---|---|---|---|---|---|
| 0 | Dados brutos | PD-23, PD-26, (PD-24) | sim | não | ✅ 2026-09-29: novo download, 37 + 14, reorganizado |
| 1 | `MonusegDataset` (XML → máscara) | PD-25, PD-07 | sim | não | ✅ 2026-09-29: E1-a, E1-b e E1-c aplicadas (§13.5) |
| 2 | `CellposeStep` | PD-29 (C4), PD-30, PD-31 (C7, `flows`/`styles`), PD-16 (C6, notebook) | sim (PD-30) | sim | ✅ 2026-10-06: C4, C6 (notebook), C7 (`flows`/`styles`) + PD-34 e PD-30 |
| 3 | `RGBAStep` | PD-34 (ideia) | — | — | — |
| 4 | `DistanceMapStep` | PD-06 (depende da instância da etapa 1) | sim | não | ✅ 2026-10-06 (`308ba92`) |
| 5 | `SaveResultsStep` | regerar tudo **uma vez**, com commit e parâmetros no `meta.json` (PD-47) | — | sim | ✅ 2026-10-07 (`0cc2d7f`, gerado no Colab sobre `308ba92`) |
| 6 | carregamento no treino | PD-19, PD-08, PD-02, PD-46 | não | não | — |
| 7 | `MarkerStep` | PD-16 (C6), PD-36/PD-38 | não | — | — |
| 8 | ScribblePrompt | PD-15 (C3), PD-44/PD-49 (C8), PD-04, PD-05 | não | — | — |
| 9 | Losses | PD-11/PD-42 (C1), PD-43 (C5), PD-12 | não | — | — |
| 10 | `Trainer` / `TrainingLoop` | PD-14 (C2), PD-33, PD-45 | não | — | — |

As correções C1–C8 deste documento entram na etapa correspondente. **Uma tarefa por vez:** base → ok do autor → commit
próprio → 08 e documentos atualizados. Em cada commit: código, testes, a pendência no 08 (✅ com data e commit) e os documentos da
seção. No fim de cada etapa, rodar a suíte inteira e registrar o resultado no §11.

---

## 11. Registro de execução

| Correção | Commit | Testes depois | Observações |
|---|---|---|---|
| Etapa 0 (PD-23, PD-26) | `f263379` | nenhum código mudou | 37 + 14 pares achados pelo `MonusegDataset` real |
| Etapa 1 (PD-25, PD-07) | `3dc9d75` | 81: 77 ok, 3 falhas conhecidas (PD-11), 1 pulado | GT novo idêntico à medição da base em 37/37 e 14/14 |
| C1 | — | — | — |
| C2 | — | — | — |
| C3 | — | — | — |
| C4 (PD-29) | `f4fcf3a` | 86: 82 ok, 3 falhas conhecidas (PD-11), 1 pulado | `ValueError` antes de carregar o modelo e antes da checagem de GPU |
| C5 | — | — | — |
| C6, parte do notebook (PD-16) | `9395df2` (validado pelo autor) | 82 ok, 3 falhas conhecidas (PD-11), 1 pulado (o notebook não tem testes; sintaxe da célula conferida) | célula 9 sem fallback; markdown das células 0 e 16 descreve o comportamento; não executado (precisa de GPU). No mesmo pacote: menções a pendências tiradas do código desta sessão (PD-50) |
| C6, parte do `MarkerStep` | — | — | fica para a etapa 7 |
| PD-30, teste no Colab | `2922d37` | — (notebook + resultados) | limiares padrão +0,031 de Dice no treino; diâmetro ≈ 0 |
| PD-30, parâmetros no `CellposeStep` | `fdd08c0` | 95: 91 ok, 3 falhas conhecidas (PD-11), 1 pulado | `diam_mean=None`, `flow_threshold=0.4`, `min_size=15` |
| Etapa 4: Dmap por núcleo (PD-06) | `308ba92` | 100: 96 ok, 3 falhas conhecidas (PD-11), 1 pulado | idêntico à medição da base em 37/37 e 14/14 |
| Etapa 5: regeração dos `.npy` | `0cc2d7f` (autor, no Colab) | conferência dos dados: nenhuma falha (§17) | 37 + 14; Dice do Cellpose 0,845 (treino) / 0,837 (teste) |
| C7, parte do Cellpose (PD-31) + PD-34 (a) | `64a624f` (validado pelo autor) | 94: 90 ok, 3 falhas conhecidas (PD-11), 1 pulado | `flows`/`styles` fora; `cellpose_prob` e `RGBAStep(alpha=...)`; notebook não executado (precisa de GPU) |
| C7, resto (PD-17, PD-20, PD-31) | — | — | — |
| C8 | — | — | — |

---

## 12. Dúvidas para você validar

1. **Escopo:** as oito correções (C1 a C8) estão boas? Quer tirar ou incluir alguma (por exemplo, a PD-33)?
2. **C3:** subir o mínimo para `torch>=2.6` (C3b)? E você consegue conferir o hash no Colab (a saída do oráculo imprime)?
3. **C4:** nome de modelo desconhecido deve dar **erro** (recomendado) ou só aviso?
4. **C6:** tirar o `try/except` do notebook de pré-processamento, para que uma falha do Cellpose pare a execução?
5. **C8 / PD-49:** qual forma do canal positivo: (a) limiar suave (recomendado), (b) *straight-through* ou (c) cru? E qual τ,
   depois de ver a tabela de tolerância?
6. **C7:** confirma apagar `flows`/`styles` do `CellposeStep`, sabendo que a PD-34 usaria só `flows[2]`, numa chave própria?

---

## 13. Etapa 1 — XML → máscara: base (medida em 2026-09-29, 37 + 14 imagens)

> Só leitura: nada no repositório mudou. Os scripts (`etapa1_xml.py`, `etapa1_raster.py` e `etapa1_inst.py`) estão no
> scratchpad da sessão. ❓ Versioná-los faz parte da PD-47.

### 13.0 Em linguagem simples (pergunta do autor: "isso nos prejudicou?")

O XML guarda o **contorno** de cada núcleo. O `cv2.fillPoly` pinta o interior **e todo pixel que o contorno encosta**, como um
contorno traçado com caneta grossa. Cada núcleo do GT ganha uma borda de ~meio pixel: ~10% a mais de área na média, ~13% no núcleo
típico e mais de 26% nos 5% mais afetados (os pequenos, em que a borda pesa mais).

| Onde | Efeito | Tamanho |
|---|---|---|
| Avaliação | todos os métodos foram medidos contra o mesmo GT "gordo"; a comparação entre experimentos é justa, mas o Dice absoluto sai um pouco subestimado para quem segmenta justo (o Cellpose) | Cellpose no teste 0,810 → ~0,822; o exp. 4 não foi medido ❓ |
| Treino | a rede aprendeu a mirar em núcleos um pouco maiores | pequeno |
| Pipeline 0,648 × Cellpose 0,810 (PD-01) | **não é explicado por isso**: ~0,01 de Dice contra uma distância de 0,16 | — |

Os 5 polígonos degenerados são desprezíveis (5 em 24 mil). A fusão de núcleos (PD-07) não muda o Dice por pixel, só o que é
calculado por núcleo (o Dmap, PD-06). **Por que corrigir mesmo assim:** o GT precisa corresponder às anotações no artigo; o Dmap
por núcleo depende do contorno certo; e os dados já vão ser regerados na etapa 5, então corrigir agora evita uma segunda regeração.

### 13.1 Como o código rasteriza hoje ✅
[monuseg_dataset.py:60-72](../../src/data/load/monuseg_dataset.py#L60-L72) trunca cada vértice com `int()` e pinta cada
polígono com `cv2.fillPoly(mask, [pts], 1)`, tudo com valor 1.

### 13.2 O que foi medido ✅

**(a) As coordenadas.** 84,3% dos vértices do treino são fracionários, mas só em 30 das 37 imagens: as 7 novas têm vértices
inteiros. No teste, só 11,2% são fracionários. O campo `Area` de cada região no XML é exatamente a área do polígono (fórmula do
laço, razão 1,0000). Isso dá uma referência de área.

**(b) A rasterização atual "engorda" o GT.** O `fillPoly` pinta todo pixel que a borda toca. O GT tem **~10% mais pixels** que a
área dos polígonos (1,096 no treino e 1,099 no teste), o que equivale a meia camada de pixels a mais em volta de cada núcleo.

**(c) Qual rasterização bate com a imagem?** Duas referências: a área dos polígonos e a máscara do Cellpose, que é independente do
GT (vem só da imagem). "Centro" pinta o pixel cujo centro cai dentro do polígono.

| Rasterização | pixels / área dos polígonos (treino · teste) | Dice Cellpose × GT (treino 30 · teste 14) | massa Cellpose / GT (teste) |
|---|---|---|---|
| **atual**: `fillPoly` com `int()` | 1,096 · 1,099 | 0,8016 · 0,8103 | 0,897 |
| `fillPoly` com `round()` | 1,096 · 1,099 | 0,7991 · 0,8097 | 0,897 |
| centro do pixel em `(c, r)` | 1,024 · 1,083 | 0,8062 · 0,8102 | 0,911 |
| **centro do pixel em `(c + 0,5, r + 0,5)`** | **0,995 · 1,011** | **0,8099 · 0,8218** | **0,977** |

Leitura: a última linha é a única que acerta a área nos dois conjuntos, e é também a que melhor alinha com o Cellpose. Isso indica
que o XML usa a convenção de **canto do pixel**: `X = 0` é a borda esquerda do pixel 0, e o centro dele é 0,5. ❓ É uma inferência
a partir dos dados; o formato do Aperio ImageScope não foi conferido em documentação.

**(d) A PD-25 muda de figura.** Com a convenção de canto, `int()` = `floor` = o índice do pixel que contém o ponto, que é o certo.
Trocar por `round()` **piora** o alinhamento no treino (0,7991 contra 0,8016). O problema real não é a truncagem, e sim o
`fillPoly`, que inclui a borda. A diferença entre `int()` e `round()` no treino (1,3% dos pixels) é dos vértices fracionários; no
teste, quase não há diferença (0,1%).

**(e) Polígonos degenerados.** São 5, todos no treino, com **2 vértices e `Area = 0`** (por exemplo, `TCGA-50-5931` região 80:
dois pontos a 0,33 px um do outro). São cliques soltos na anotação, não núcleos. Hoje o `fillPoly` pinta 1–2 px para cada um.
Com o centro do pixel, mais 2 regiões pequenas ficam com 0 px.

**(f) Fusão e sobreposição (PD-07, agora com 37 imagens).** O GT binário tem 18.160 componentes para 24.140 regiões no treino
(**−24,8%**) e 6.086 para 6.697 no teste (−9,1%). Com o centro do pixel, a sobreposição entre polígonos é 1,00% do primeiro plano
no treino e 0,25% no teste. Numa máscara por instância, a regra "o último desenhado vence" apaga 2 núcleos inteiros no treino;
a regra "o menor vence" não apaga nenhum.

### 13.3 Proposta (depende da sua decisão)

| # | Mudança | Base |
|---|---|---|
| E1-a | rasterizar pelo **centro do pixel em (c + 0,5, r + 0,5)** (`skimage.draw.polygon` sobre `X − 0,5`, `Y − 0,5`), no lugar de `int()` + `fillPoly` | (b), (c), (d) |
| E1-b | **descartar** regiões com menos de 3 vértices ou com área 0, registrando no log quantas | (e) |
| E1-c | gerar também a **máscara por instância** (`int32`, um rótulo por região do XML; na sobreposição, **o menor vence**); o GT binário passa a ser `instâncias > 0` | (f); pré-requisito do Dmap por núcleo (PD-06) |

**O que muda nos números:** o GT de treino e de teste muda, então **todas** as linhas de base precisam ser refeitas com o GT novo
(etapa 5). Por exemplo, o Dice do Cellpose no teste passaria de 0,810 para ~0,822 (medido acima) só por causa do GT. Os números
antigos continuam válidos como histórico, desde que se diga com qual GT foram calculados.
**Dependências:** o `scikit-image` não está no `requirements.txt`. A alternativa sem nova dependência é
`matplotlib.path.Path.contains_points` (❓ o matplotlib também não está fixado) ou implementar o teste de ponto no polígono em numpy.
**Testes a escrever junto:** um quadrado com vértices em canto de pixel dá exatamente a área esperada; um polígono com 2
vértices é descartado; dois polígonos sobrepostos → o menor mantém os seus pixels; o número de rótulos = número de regiões válidas.

### 13.4 Dúvidas para você validar

1. **E1-a:** trocar o `fillPoly` pela rasterização pelo centro do pixel? (Recomendado. Muda o GT em ~10% da borda de cada núcleo.)
2. **E1-b:** descartar os 5 polígonos de 2 vértices?
3. **E1-c:** regra de sobreposição da máscara por instância: "o menor vence" (recomendado) ou outra?
4. **Dependência:** aceitar `scikit-image` fixado no `requirements.txt`, ou prefere a implementação em numpy (sem dependência nova)?

### 13.5 Decisões e execução (2026-09-29)

| # | Pergunta | Resposta do autor | Onde ficou |
|---|---|---|---|
| 1 | Rasterizar pelo centro do pixel? | sim (recomendação) | `xml_to_instance_mask`; PD-25 ✅ |
| 2 | Descartar os 5 polígonos de 2 vértices? | sim | idem, com log por arquivo |
| 3 | Máscara por instância, "o menor vence"? | sim | chave `ground_truth_instances`; PD-07 parcial |
| 4 | `scikit-image` ou numpy? | `scikit-image` | `requirements.txt`: `scikit-image==0.24.0` |

**O que mudou no código** ✅:
- [monuseg_dataset.py](../../src/data/load/monuseg_dataset.py): funções `read_xml_regions`, `polygon_area` e
  `xml_to_instance_mask`; o `__getitem__` devolve `ground_truth` (binário, `uint8`) **e** `ground_truth_instances` (`int32`) quando a
  anotação é XML; o `_xml_to_mask` passa a ser a versão binária da função nova. O `import cv2` saiu (não era usado em mais nada).
- [tests/test_monuseg_dataset.py](../../tests/test_monuseg_dataset.py): 7 testes (§13.3).

**Conferência nos dados reais:** o binário novo é idêntico, pixel a pixel, à rasterização "centro − 0,5" da medição em 37/37
(treino) e 14/14 (teste) imagens. Área / área anotada: 0,995 e 1,011. Instâncias: 24.133 de 24.135 regiões válidas no treino (2
ocupam 0 px) e 6.697 de 6.697 no teste. Dice do Cellpose contra o GT novo: 0,8099 (treino, 30 imagens) e 0,8218 (teste).
**Ainda não muda nenhum resultado:** os notebooks leem o `MoNuSegPreprocessed/`, que só é regerado na etapa 5.

---

## 14. Etapa 2 — `flows`/`styles` fora e probabilidade do Cellpose como opção (C7 parte do Cellpose + PD-34)

### 14.1 Base ✅
- **`flows` e `styles`** não são lidos por nenhum código nem salvos pelo `SaveResultsStep`. O único leitor é o notebook-tutorial
  `preprocessamento_monuseg.ipynb`, que faz `if 'flows' in ...` e não quebra. No Cellpose 4, `styles` é só zeros.
- **O que o `eval` devolve** (`cellpose/models.py` v4.1.1, L342): `masks, [fluxo_em_cores, dP, cellprob], styles`.
- **O `cellprob` (`flows[2]`) é um logit,** não uma probabilidade: o Cellpose o treina com `BCEWithLogitsLoss` (`train.py`, L47 e
  L51), e os pixels de célula são `cellprob > cellprob_threshold`, com padrão 0 (`dynamics.py`, L647), que equivale a 0,5 depois da
  sigmoide. A sigmoide põe o mapa na mesma escala da máscara binária ([0, 1]) e cabe em `float16`.
- **Formato:** no 2D, com `resample=True` (padrão), o mapa volta ao tamanho original mesmo com reescala (`_run_net`,
  `resize_image(yf, shape[1], shape[2])`), o que importa para a PD-30.
- **Disco:** 1000×1000 em `float16` = 2 MB por imagem, ~102 MB para as 51 imagens (contra ~204 MB em `float32`).

### 14.2 Decisões do autor (2026-10-01)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | Qual variante da PD-34? | (a): probabilidade no lugar da máscara, 4 canais | `RGBAStep(alpha="prob")` |
| 2 | Parametrizável? | sim | `alpha="mask"` (padrão) ou `"prob"`; `RGBA_ALPHA` no notebook |
| 3 | `float16` ou `float32`? | `float16` | `CellposeStep` |
| 4 | Junto com a C7 (`flows`/`styles`)? | sim, um pacote só | este §14 |

### 14.3 O que mudou ✅ (`64a624f`)
- `cellpose_step.py`: o `forward` grava `segmentation` e `cellpose_prob` (`expit(flows[2])` em `float16`) e confere o formato;
  não grava mais `flows` nem `styles`.
- `rgba_step.py`: parâmetro `alpha` (`"mask"` | `"prob"`, validado no construtor); confere o formato da origem e, com `"prob"`,
  a faixa [0, 1].
- `save_results_step.py`: `cellpose_prob` entra nas chaves padrão.
- Notebook de pré-processamento: `RGBA_ALPHA = "mask"`, `RGBAStep(alpha=RGBA_ALPHA)`, `meta.json` com `rgba_alpha` e as chaves
  vindas de `SaveResultsStep.DEFAULT_KEYS`; a célula 13 lê `cellpose_prob`; markdown das células 0 e 16 atualizado.
- Testes: +2 no `test_cellpose_step.py` (chaves do `forward`, sigmoide, formato), +5 no `test_rgba_step.py` (padrão, `"prob"`,
  valor inválido, chave ausente, faixa e formato), +1 no `test_save_results_step.py` (`float16` preservado); o teste das chaves
  padrão foi atualizado.

**Efeito nos resultados:** nenhum enquanto `alpha="mask"`. Os `.npy` só mudam na etapa 5.

---

## 15. Etapa 2 — teste dos parâmetros do Cellpose (PD-30): base e notebook

### 15.1 Base ✅
- **O que o `diameter` faz** (`cellpose/models.py` v4.1.1, L271-273): `image_scaling = 30 / diameter` se `diameter > 0`; com
  `None`, o fator é 1. Ou seja, `diameter=None` e `diameter=30` dão a mesma reescala (nenhuma). A configuração "padrão da
  biblioteca" do plano só muda `flow_threshold` (0,4) e `min_size` (15).
- **Diâmetro real com o GT novo** (37 imagens, máscara por instância da etapa 1; diâmetro equivalente `2·√(área/π)`, mediana por
  imagem e depois mediana entre imagens):

  | Grupo | Imagens | Diâmetro mediano | Faixa entre imagens |
  |---|---|---|---|
  | 40× (`MicronsPerPixel` < 0,4) | 30 | **22,3 px** | 16,9–32,5 |
  | 20× (`TCGA-HE-7128/7129/7130`) | 3 | **12,4 px** | 10,9–12,6 |
  | sem `MicronsPerPixel` | 4 | 23,4 px | 18,4–30,3 |

  Os 24 e 13 px do plano antigo vinham do GT antigo (~10% maior). As 4 imagens sem escala têm núcleos do tamanho das 40×.
- **Canais:** as 37 imagens de treino são RGB (3 canais).

### 15.2 Configurações do teste

| Config. | `diameter` | `flow_threshold` | `min_size` |
|---|---|---|---|
| `atual` | 30 | 0,2 | 4 |
| `padrao` | `None` (= fator 1) | 0,4 | 15 |
| `d22` | 22 | 0,2 | 4 |
| `por_imagem` | 12 se `MicronsPerPixel ≥ 0,4`, senão 22 | 0,2 | 4 |

### 15.3 O notebook
[notebooks/exploration/cellpose_parametros.ipynb](../../notebooks/exploration/cellpose_parametros.ipynb):
- **Setup:** clona o branch `homolog` ou, se o clone já existir no Colab, faz `fetch` + `pull --ff-only`; instala o
  `requirements.txt`; exige GPU; registra o commit e as versões.
- **Execução:** um único `CellposeStep` (o modelo carrega uma vez), com `diam_mean`, `flow_threshold` e `min_size` trocados por
  configuração; só as 37 de treino.
- **Saída:** tabela média ± desvio, Dice por grupo de escala (20× × 40×), comparação pareada com `atual` (diferença média,
  nº de imagens que melhoram, Wilcoxon) e os arquivos `docs/estudo/resultados/cellpose_parametros_treino.csv` +
  `..._meta.json`, baixados automaticamente no Colab.
- **Ensaio a seco** (local, sem GPU, com um Cellpose de mentira): o fluxo inteiro roda nas 37 imagens, e os diâmetros chegam
  certos (12 nas três em 20× e 22 nas outras 34).

**Para rodar:** o notebook precisa estar no GitHub (o Colab clona o `homolog`), então ele só roda depois do commit + push.

### 15.4 Resultado (Colab, 2026-10-02) ✅
Rodado pelo autor: commit `64a624f` (sem mudanças locais em `src/`), Tesla T4, torch 2.11.0+cu128, numpy 2.0.2, cellpose 4.1.1.
Arquivos: [cellpose_parametros_treino.csv](resultados/cellpose_parametros_treino.csv) e
[..._meta.json](resultados/cellpose_parametros_treino_meta.json). 37 imagens de treino, GT da etapa 1. Média ± desvio padrão:

| Config. | Dice | IoU | Precisão | Revocação | Massa / GT | Instâncias / núcleos | s por imagem |
|---|---|---|---|---|---|---|---|
| `atual` (30; 0,2; 4) | 0,814 ± 0,038 | 0,688 | 0,879 | 0,761 | 0,867 | 0,80 | 10,4 |
| **`padrao`** (`None`; 0,4; 15) | **0,845 ± 0,030** | **0,733** | 0,867 | **0,826** | 0,955 | 0,92 | 10,7 |
| `d22` (22; 0,2; 4) | 0,815 ± 0,039 | 0,690 | 0,872 | 0,768 | 0,883 | 0,80 | 20,6 |
| `por_imagem` (22/12; 0,2; 4) | 0,816 ± 0,038 | 0,691 | 0,871 | 0,771 | 0,887 | 0,81 | 23,7 |

**Comparação pareada com `atual` (Dice, 37 imagens):**

| Config. | Δ médio | Δ mediano | Melhora / piora | Wilcoxon p |
|---|---|---|---|---|
| `padrao` | **+0,0310** | +0,0279 | **37 / 0** | **1,5e-11** |
| `d22` | +0,0011 | −0,0006 | 16 / 21 | 0,59 |
| `por_imagem` | +0,0021 | +0,0012 | 19 / 18 | 0,15 |

**Por escala (Dice):** em 40× (34 imagens) `atual` 0,816, `padrao` 0,846, `por_imagem` 0,818; em 20× (3 imagens) `atual` 0,790,
`padrao` 0,836, `por_imagem` 0,798. Nas três em 20×, ampliar pelo diâmetro de 12 px ajuda um pouco (+0,008 a +0,014 em relação
ao `atual`); usar 22 nelas piora um pouco.

**Leitura:**
1. **Os limiares são o fator que importa.** `flow_threshold=0,2` e `min_size=4` são mais rígidos que o padrão e fazem o Cellpose
   descartar máscaras boas: ele acha 80% dos núcleos anotados, contra 92% com os padrões. A revocação sobe 6,5 pontos e a precisão
   cai só 1,2. O ganho é consistente: melhora as 37 imagens, a menor melhora é +0,006.
2. **O diâmetro quase não importa** para o `cpsam` neste conjunto, exceto, um pouco, nas três imagens em 20×. Custa ~2× o tempo.
3. ❓ **Não testado:** qual dos dois limiares causou o ganho, se afrouxar mais ajuda (a revocação ainda está abaixo da precisão)
   e a combinação limiares padrão + diâmetro por imagem.

### 15.5 Decisões (autor, 2026-10-06)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | Adotar `flow_threshold=0,4` e `min_size=15`? | sim | padrões do `CellposeStep` (§15.6) |
| 2 | Diâmetro: sem reescala ou por imagem? | sem reescala | `diameter=None` |
| 3 | Segunda rodada antes de fixar? | não; seguir | o que não foi testado fica registrado na PD-30 |

#### Perguntas originais (histórico)
1. Adotar `flow_threshold=0,4` e `min_size=15` no `CellposeStep`? Os dados apoiam com folga.
2. Diâmetro: deixar sem reescala (`None`/30) ou usar o diâmetro por imagem, que ajuda só as três imagens em 20× e dobra o tempo?
3. Rodar uma segunda rodada curta (≈ 5 configurações, ~40 min numa T4) para separar o efeito de cada limiar e testar afrouxar
   mais, antes de fixar? Ou fixar os padrões já?

### 15.6 Aplicação (2026-10-06, `fdd08c0`, validada pelo autor)
- `cellpose_step.py`: padrões `diam_mean=None`, `flow_threshold=0.4`, `min_size=15`; o `diam_mean` deixou de ser passado ao
  construtor do `CellposeModel`, que o ignora no Cellpose 4 (vai só para o `eval`, como `diameter`). Docstrings explicam o que
  cada parâmetro faz e por que o padrão foi escolhido.
- `test_cellpose_step.py`: +1 teste (os padrões chegam ao `eval`). Suíte: **95 testes, 91 ok, 3 falhas conhecidas (PD-11), 1 pulado**.
- `cellpose_parametros.ipynb`: a descrição da config. `atual` deixou de dizer "os valores do `CellposeStep` hoje".
- **Efeito:** nenhum número muda até a regeração dos dados (etapa 5); ali, a linha de base do Cellpose passa a ser a do `padrao`.

---

## 16. Etapa 4 — mapa de distância por núcleo (PD-06): base

### 16.1 Como é hoje ✅
[compute_distance_map](../../src/pipeline/steps/preprocessing/distance_map_step.py#L23-L43): `1 − EDT/máx` sobre o GT binário,
com o **máximo da imagem inteira**. Só o maior núcleo de cada imagem tem centro 0; os pequenos têm o centro "caro". Na tese
(eq. 5.7), a normalização é por objeto (explicação em linguagem simples no [04 §7](04-losses.md)).

### 16.2 Medido com o GT novo (2026-10-06) ✅
Script `etapa4_dmap.py` (scratchpad da sessão). Para cada núcleo (instância da etapa 1), o valor mínimo do mapa dentro dele, que é
o valor no seu centro:

| Mapa | Centro do núcleo, mediana (treino · teste) | Núcleos com centro > 0,5 | Pixels de núcleo < 0,5 |
|---|---|---|---|
| atual (máximo da imagem) | 0,473 · 0,506 | **43,5% · 52,0%** | 6,9% · 6,7% |
| por componente conexa do GT binário | 0,000 · 0,000 | 0,4% · 0,1% | 32,2% · 33,3% |
| **por núcleo** (`ground_truth_instances`, recorte com 1 px de margem) | **0,000 · 0,000** | **0,0% · 0,0%** | **33,3% · 33,6%** |
| por núcleo, recorte justo (esboço do 04 §7.6) | 0,000 · 0,000 | 0,0% · 0,0% | 36,1% · 35,9% ⚠️ |

- **O recorte justo erra:** a EDT só "vê" fundo dentro do recorte, então os pixels na borda do recorte ficam com distância maior
  que a real, e o mapa fica barato demais. Com 1 px de margem (limitada à imagem), o mapa por núcleo é **idêntico, pixel a pixel**,
  ao calculado com a EDT da imagem inteira para cada núcleo (conferido em 3 imagens: diferença máxima 0).
- **Custo:** 0,03 s por imagem.
- **Núcleos vizinhos:** com a máscara por instância, a fronteira entre dois núcleos que se tocam conta como borda dos dois, o que
  o mapa por componente não faz (ele os trata como um núcleo só).
- **Borda da imagem:** como no mapa atual, a borda da imagem **não** conta como fundo. Num núcleo cortado pela borda, o ponto
  mais barato fica junto da borda, onde o centro verdadeiro provavelmente estaria.

### 16.3 Proposta de código (aguarda decisão)
- `distance_map_step.py`: função nova `compute_instance_distance_map(labels)` (EDT por rótulo, recorte com 1 px de margem) e
  parâmetro `normalization` no `DistanceMapStep`: `"instance"` (padrão; lê `ground_truth_instances`) ou `"image"` (o mapa atual,
  lê `ground_truth`). Sem *fallback*: com `"instance"` e sem a chave de instâncias, `KeyError`.
- Notebook de pré-processamento: o `resize_sample` e o `run_preprocess` passam a levar `ground_truth_instances` adiante (hoje o
  descartam, PD-07); a redução, se houver, é por vizinho mais próximo. O `meta.json` registra a normalização usada.
- Testes: dois núcleos de raios 20 e 6 têm os **dois** centros em 0; núcleos que se tocam; recorte com margem igual à EDT na imagem
  inteira; modo `"image"` igual ao atual; chave ausente e opção inválida dão erro. Os testes atuais do `DistanceMapStep` passam a
  usar `normalization="image"`, que é o que eles testam.
- **Fica para depois:** recalibrar o peso do DMap (o mapa novo tem ~5× mais pixels baratos), na fase 2.

### 16.4 Decisões para o autor
1. Parametrizar (`"instance"` como padrão e `"image"` como opção), como foi feito com o alpha?
2. Salvar `ground_truth_instances` em disco na regeração? Recomendação: **não**. Ela sai dos XMLs em CPU (~0,5 s por imagem) sempre
   que for preciso, e salvar custaria ~102 MB a mais no git (em `uint16`).
3. Borda da imagem: manter como hoje (a borda não conta como fundo)?

### 16.5 Decisões (autor, 2026-10-06)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | Parametrizar (`"instance"` / `"image"`)? | não: o mapa pela imagem inteira **não é útil** e é **removido** | só `compute_instance_distance_map` |
| 2 | Salvar `ground_truth_instances`? | não | sai dos XMLs quando for preciso |
| 3 | Borda da imagem conta como fundo? | não (como hoje) | docstring e teste |

**Por que remover o modo "imagem"** (pergunta do autor): não é uma alternativa científica (a tese define o mapa por objeto, e o
da imagem inteira foi um erro de implementação); não serve para reproduzir os exp. 1–6, que também usaram o GT antigo, 30 imagens
e outros limiares do Cellpose (o código antigo continua no git); e nada em `src/` ou nos testes o usava.

### 16.6 Aplicação (2026-10-06, `308ba92`, validada pelo autor)
- `distance_map_step.py`: `compute_distance_map` (binário, máximo da imagem) deu lugar a `compute_instance_distance_map(labels)`
  (EDT por rótulo num recorte com 1 px de margem; valida que a entrada é 2D de inteiros). O `DistanceMapStep` lê
  `instances_key="ground_truth_instances"` e levanta `KeyError` sem ela.
- `test_distance_map_step.py` reescrito: 11 testes (faixa e tipo; centro 0 / fundo 1 / borda 0,5; núcleos de raios 20 e 6 com os
  dois centros em 0; núcleos que se tocam; recorte com margem igual à EDT na imagem inteira; borda da imagem; entrada não inteira;
  chaves do step; falta das instâncias; chaves personalizadas). Suíte: **100 testes, 96 ok, 3 falhas conhecidas (PD-11), 1 pulado**.
- Notebook de pré-processamento: `resize_sample` e `run_preprocess` levam `ground_truth_instances` (redução por vizinho mais
  próximo, se houver); `meta.json` ganha `distance_map`; markdown das células 0, 8 e 16 atualizado (sai o "(P7)", PD-50).
- **Conferência:** a função nova é idêntica ao mapa da medição da base em 37/37 e 14/14 imagens. Ensaio a seco do notebook (Cellpose
  de mentira, 2 + 1 imagens, gravando no scratchpad): a máscara chega ao step, os 480 núcleos da primeira imagem têm centro 0, e
  a máscara por instância não é salva.
- ⚠️ Os notebooks antigos `experiment_1/2` e `test_e2e_pipeline` chamam `DistanceMapStep()` sobre o GT binário e deixariam de
  rodar com o código atual; ficam para a poda da PD-10.

---

## 17. Etapa 5 — dados regerados (2026-10-07)

### 17.1 Procedência ✅
- Gerado pelo autor no Colab e enviado como `0cc2d7f` ("chore: rodando pre-processamento"), cujo pai é `308ba92`: o código usado é
  o da etapa 4. O commit mexe só em `data_source/MoNuSegPreprocessed/` (263 arquivos).
- `meta.json`: 37 + 14, `rgba_alpha: "mask"`, `distance_map` por núcleo, chaves com `cellpose_prob`. ⚠️ Ele ainda não registra o
  commit, as versões nem os parâmetros do Cellpose (item 2 da revisão do notebook). Os parâmetros ficam provados pela conferência
  abaixo: o Dice por imagem no treino é **idêntico** (diferença máx. 4e-14) ao da config. `padrao` da PD-30, e difere em até 0,08 do
  `atual`.
- Tamanho: `MoNuSegPreprocessed/` passou de 1,1 GB para 1,4 GB (PD-21).

### 17.2 Conferência (script `verifica_regeracao.py`, scratchpad) ✅ — nenhuma falha
Para cada uma das 51 imagens:
- `image` idêntica ao `.tif`;
- `ground_truth` idêntico ao GT da etapa 1 (`xml_to_instance_mask > 0`);
- `distance_map` idêntico ao mapa por núcleo, e o centro de todo núcleo vale 0;
- `rgba`: RGB = imagem/255 e alpha = máscara do Cellpose;
- `cellpose_prob` em [0, 1].

Tipos: `image` uint8 (1000,1000,3); `segmentation` uint16; `cellpose_prob` float16; `rgba` float32 (1000,1000,4); `ground_truth`
uint8; `distance_map` float32.

### 17.3 Números novos ✅

| | Treino (37) | Teste (14) |
|---|---|---|
| Dice do Cellpose × GT (média ± desvio por imagem) | **0,845 ± 0,030** | **0,837 ± 0,034** |
| Instâncias do Cellpose / núcleos anotados | 0,93 | 1,09 |
| Pixels das máscaras com `cellpose_prob ≥ 0,5` | 100% | 100% |
| Pixels com `cellpose_prob > 0,5` fora de máscaras | 1,45% (máx. 3,07%) | 1,40% (máx. 2,93%) |

- **Linha de base nova (PD-01):** 0,837 no teste, contra 0,810 com o GT e os limiares antigos. É a régua para os próximos
  experimentos. Os números dos exp. 1–6 são do GT antigo.
- No teste, o Cellpose acha **mais** instâncias que núcleos anotados (1,09). ❓ Pode ser que ele divida núcleos ou ache núcleos não
  anotados; não foi investigado.

### 17.4 Correções do notebook de pré-processamento (2026-10-07, `f25eaf3`, validadas pelo autor)
Da revisão feita antes da regeração. **Nenhuma muda os dados** gerados com o mesmo código; elas tornam a próxima regeração segura
e documentada.
1. **Setup:** clona o `homolog` ou, se o clone já existir no Colab, faz `fetch` + `pull --ff-only`, e imprime o commit (antes, um
   clone antigo rodava código velho sem aviso, e a busca da raiz do repositório olhava acima de `/content`).
2. **Procedência no `meta.json`:** commit, mudanças locais em `src/`, versões (python, torch, cellpose, numpy), GPU e os
   parâmetros do Cellpose (modelo, `diameter`, `flow_threshold`, `min_size`, `cellprob_threshold`) (PD-47).
3. **Limpeza:** apaga `train/` e `test/` antes de gravar, para não misturar arquivos de runs diferentes.
4. **Textos:** 37 imagens de treino; tipos certos na árvore de pastas (`segmentation` uint16, `ground_truth` uint8).
5. **Sem referências à história** (PD-50): saem "(C6)", "(P7)", `experiment_3` e "regra 10".
6. **Checagens automáticas** (célula nova): formatos e tipos, `cellpose_prob` em [0, 1], alpha coerente com `RGBA_ALPHA` e centro
   de todo núcleo em 0 no mapa de distância, para todas as imagens.
7. **Metadata do Colab:** abre com GPU (T4).
8. **Sem redimensionamento:** o `resize_sample` (que não fazia nada, porque o MoNuSeg já é 1000×1000) virou `check_sample`, que
   levanta `ValueError` se alguma imagem ou máscara não tiver 1000×1000. O `meta.json` mantém `work_size: 1000`, lido pelos
   notebooks de experimento.

As saídas guardadas (do run de 16/08) foram **limpas**: não correspondem mais ao código nem aos dados, regerados em 07/10.
**Ensaio a seco** (Cellpose de mentira, 2 + 1 imagens, gravando no scratchpad e conferindo antes que o redirecionamento foi
aplicado): fluxo completo, pasta antiga apagada, `meta.json` com procedência, checagens aprovadas; os dados do repositório ficaram
intactos (hash de todos os arquivos igual antes e depois).

---

## 18. Fase 2 — módulo de dados, métricas e k-fold (PD-19, PD-08, PD-02, PD-46): base e desenho

### 18.1 O que existe hoje ✅ (lido no `experiment_6.ipynb`, igual nos exp. 3–6, e em `src/`)
- **`load_preprocessed(split)`** (célula 8): lê `image`, `rgba`, `ground_truth` e `distance_map` de cada `.npy`. Não lê
  `segmentation` nem `cellpose_prob`, então a linha de base do Cellpose não é calculada no notebook.
- **`build_batch` / `make_batches`** (célula 12): rot90 (k ∈ 0..3) + flip horizontal + flip vertical, sorteados com `np.random` e
  aplicados juntos aos quatro arrays; converte para tensores `(B,C,H,W)`. Os batches são montados **uma vez**, antes do treino,
  sem shuffle: todas as épocas veem as mesmas aumentações na mesma ordem (PD-08).
- **`compute_binary_metrics` / `evaluate_val_quality`** (célula 22): Dice, IoU, precisão, revocação e massa por imagem, limiar 0,5,
  mais a fração de marcador. Roda sob `no_grad`, mas **não põe a rede em `eval()`**: depende de a última chamada ter sido um
  `eval_step` do `TrainingLoop`. Funciona por acaso.
- **`TrainingLoop`**: aceita qualquer iterável de batches e itera de novo a cada época; por isso um `DataLoader` com shuffle já dá
  batches novos por época. **Não tem como parar antes** (early stopping) nem guarda o melhor modelo (PD-46).
- **Callbacks:** `on_validation_step_end(trainer, data, loss, loss_log)` recebe o `data` de saída do pipeline, com `segmentation` e
  `ground_truth`. Dá para calcular o Dice de validação por imagem **sem um forward a mais**.
- **`MonusegPreprocessedDataset`**: roda o pipeline de pré-processamento na hora, sobre o dataset bruto; não lê os `.npy`. Só é
  usado pelo próprio teste e pelo notebook-tutorial `preprocessamento_monuseg.ipynb`.
- **Custos:** em memória, ~24 MB por imagem (`image` + `rgba` + `ground_truth` + `distance_map`), ~1,2 GB para as 51 imagens; o
  Colab tem ~12 GB. Uma época leva ~1,5 s (30 imagens, T4).

### 18.2 Desenho proposto
Cinco peças em `src/`, cada uma com testes, implementadas e validadas **uma por vez**:

| # | Peça | Onde | O que faz |
|---|---|---|---|
| 1 | Métricas | `src/evaluation/metrics.py` | `binary_metrics(pred, gt, limiar)` por imagem (Dice, IoU, precisão, revocação, massa); `evaluate(...)` põe as redes em `eval()` + `no_grad` e devolve uma linha por imagem, **com a linha de base do Cellpose** (`segmentation > 0`) ao lado |
| 2 | Dataset + aumentação + collate | `src/data/load/preprocessed_dataset.py` | `PreprocessedDataset(split, ids=None, augment=False, alpha="mask")`: lê os `.npy` uma vez para a memória; com `augment`, sorteia rot90 + flips **a cada acesso** (portanto a cada época), aplicados juntos a todas as chaves espaciais; `alpha="prob"` troca o 4º canal do `rgba` pela `cellpose_prob` na hora de carregar (PD-34). `collate_samples` empilha o batch no formato de hoje. Usado com `DataLoader(shuffle=True)` e semente fixa |
| 3 | Callbacks de validação | `src/training/callbacks/` | Dice de validação por imagem a cada época (a partir do `data` do callback); guarda o estado da MarkerUNet na época de **melhor Dice**; *early stopping* com paciência. Exige uma mudança pequena no `TrainingLoop`: parar quando um callback pedir |
| 4 | K-fold | `src/training/kfold.py` | divide as 37 imagens de treino em *folds* (por imagem, semente fixa); para cada *fold*, treina do zero e guarda as métricas por imagem da validação e a melhor época; depois, treino final com as 37 e **uma** avaliação no teste |
| 5 | Registro dos resultados | junto do k-fold | por run: configuração (JSON), commit, CSV por imagem e por *fold*, CSV do teste e resumo (média ± desvio, comparação pareada com o Cellpose) em `docs/estudo/resultados/` (PD-47) |

Depois das cinco peças, um notebook-modelo de experimento usa o módulo (em vez do código copiado), começando pelo
experimento-base (PD-44).

### 18.3 Decisões para o autor
1. **Número de *folds*:** 5 (7 ou 8 imagens de validação por *fold*)?
2. **Estratificação:** distribuir as três imagens em 20× por *folds* diferentes, para nenhum *fold* concentrar a escala diferente?
   ❓ Estratificar por órgão exigiria a lista de órgãos por imagem, que não está nos dados do repositório.
3. **Seleção e parada:** melhor Dice médio de validação do *fold* (já decidido, PD-46), com paciência de ~20 épocas e máximo de
   200?
4. **Modelo final:** treinar com as 37 pelo número de épocas da mediana das melhores épocas dos *folds* e avaliar no teste uma
   vez? (A alternativa é a média dos 5 modelos dos *folds*.)
5. **Checkpoints:** a MarkerUNet tem ~98 MB por checkpoint. Guardar fora do git (no Colab ou no Drive) e registrar só o caminho e
   o hash?
6. **`MonusegPreprocessedDataset`:** remover, junto com o código morto da C7, já que ninguém mais vai precisar dele?
7. **Ordem:** métricas → dataset → callbacks → k-fold → registro → notebook-modelo?

### 18.4 Decisões (autor, 2026-10-07)

| # | Pergunta | Resposta | Onde fica |
|---|---|---|---|
| 1 | Número de *folds* | 5, **parametrizável** por experimento | `k` no k-fold |
| 2 | Estratificação | não, por enquanto (não se sabe com certeza o órgão de cada imagem) | divisão aleatória por imagem, semente fixa |
| 3 | Seleção e parada | melhor Dice de validação, paciência ~20, máximo 200 (parametrizáveis) | callbacks |
| 4 | Modelo final | treino com as 37 e uma avaliação no teste. **Dúvida do autor sobre o número de épocas** (rodar mais ou menos conforme a queda da loss) | ver nota abaixo |
| 5 | Checkpoints | ainda não são salvos. Quando forem, ficam **fora do git, numa pasta na raiz do usuário** (ex.: `~/cell-fuzzy-models/`) | registro: caminho + hash |
| 6 | `MonusegPreprocessedDataset` | **remover todo código não usado conforme for encontrado** | sai junto com a peça 2 |
| 7 | Ordem | métricas → dataset → callbacks → k-fold → registro → notebook-modelo | — |

**Nota sobre o item 4:** parar o treino final pela queda da loss **de treino** não é um bom critério, porque ela continua caindo
com o *overfitting* (exp. 5: soft Dice ≈ 0,85 no treino contra 0,60 de Dice na validação; PD-38). O número de épocas do treino
final vem dos próprios *folds*: em cada um, o *early stopping* acha a época de melhor Dice de validação, e o treino final usa a
mediana delas. Com isso o treino final dura mais ou menos conforme o experimento, medido na validação e não na loss de treino.
O máximo, a paciência e a regra (mediana) ficam parametrizáveis.

### 18.5 Peça 1 — métricas (2026-10-07, `ba0b85c`, validada pelo autor)
- `src/evaluation/metrics.py`:
  - `binary_metrics(prediction, ground_truth, threshold=0.5)`: Dice, IoU, precisão, revocação e massa por imagem; aceita NumPy ou
    tensor; o GT conta como primeiro plano o que for diferente de zero (vale para rótulos de instância). Casos vazios com valor
    definido (Dice/IoU = 1 com os dois vazios; precisão = 1 sem nada previsto; revocação = 1 com GT vazio), em vez do `eps` dos
    notebooks, que dava Dice 0 com os dois vazios.
  - `evaluate(pipeline, batches, ...)`: põe os steps em modo de avaliação e roda sem gradiente; trabalha sobre uma **cópia** do batch
    (os steps escrevem no dicionário, e a rede final sobrescreve `segmentation`, PD-33); devolve uma linha por imagem, com as
    métricas da predição, a **linha de base do Cellpose** (`cellpose_*`, lida da chave `cellpose_segmentation` do batch, que a peça 2
    vai fornecer) e a fração de marcador.
- `tests/test_metrics.py`: 10 testes (caso calculado à mão, limiar inclusivo, rótulos de instância, 3 casos vazios, tensor = NumPy,
  formatos diferentes, `evaluate` com modo eval, sem gradiente, batch intacto e linha de base, e sem linha de base). Suíte: **110
  testes, 106 ok, 3 falhas conhecidas (PD-11), 1 pulado**.
- **Conferência:** nas 14 imagens de teste, a função nova difere da `compute_binary_metrics` dos notebooks em no máximo 5e-8 (o `eps`)
  e reproduz o Dice do Cellpose (0,8370).

### 18.6 Peça 2 — dataset, aumentação e collate (2026-10-07, `76b71a6`, validada pelo autor)
- `src/data/load/preprocessed_dataset.py`:
  - `load_preprocessed(split, root, ids=None, alpha="mask")`: lê os `.npy` do split uma vez e devolve `id -> amostra`. A
    segmentação do Cellpose entra como **`cellpose_segmentation`** (a rede final grava a saída em `segmentation`, PD-33), e é dela
    que o `evaluate` tira a linha de base. `alpha="prob"` troca o 4º canal do `rgba` pela `cellpose_prob` na carga (PD-34).
  - `PreprocessedDataset(samples, ids=None, augment=False, seed=None)`: um subconjunto de ids sobre as amostras já carregadas (os
    *folds* compartilham os arrays, sem cópia); com `augment`, sorteia rot90 (k ∈ 0..3) + flip horizontal + flip vertical **a cada
    acesso**, aplicados juntos a todas as chaves, com gerador próprio e semente; os arrays carregados nunca são alterados. Devolve
    tensores `(C, H, W)` no formato dos notebooks (`image` float em [0, 255]).
  - `collate_samples` (batch com `id` em lista) e `make_loader(dataset, batch_size, shuffle, seed)` (`num_workers=0`, ordem com
    semente).
- **Removido** (código não usado): `MonusegPreprocessedDataset` e `tests/test_monuseg_preprocessed_dataset.py`; as citações nas
  docstrings do `PreprocessingPipeline`; a seção 7 do tutorial `preprocessamento_monuseg.ipynb` (3 células) e a linha dele no resumo.
- `tests/test_preprocessed_dataset.py`: 10 testes (carga e renome, subconjunto, `alpha="prob"`, erros, tensores sem aumentação,
  ids ausentes, **mesma transformação em todas as chaves**, sorteio diferente a cada acesso sem alterar os originais, mesma
  semente = mesma sequência, collate e loader com ordem reproduzível). Suíte: **119 testes, 115 ok, 3 falhas conhecidas (PD-11),
  1 pulado**.
- **Dados reais:** carga de 37 + 14 imagens em 1,0 s, 1,27 GB em memória; uma época do loader (37 imagens, com aumentação) em 1,7 s
  em CPU; alpha = máscara do Cellpose e RGB = imagem/255 em todos os batches aumentados; com `alpha="prob"`, o 4º canal fica em [0, 1].

### 18.7 Peça 3 — callbacks de validação e parada antecipada (2026-10-07, `955af95`, validada pelo autor)
- `src/training/callbacks/best_model_callback.py` — `BestModelCallback(model, patience=None, min_delta=0.0, threshold=0.5)`:
  - em `on_validation_step_end`, calcula o Dice binário (`binary_metrics`) de cada imagem a partir do `data` que o `Trainer` já
    produz (sem forward extra);
  - em `on_epoch_end` (só nas épocas com validação), tira a média por imagem, registra em `history` e, se superar o melhor em mais
    de `min_delta`, guarda uma cópia em CPU do `state_dict` do modelo (`best_epoch`, `best_dice`, `best_state`);
  - com `patience`, depois de `patience` validações seguidas sem melhora, marca `trainer.stop_training = True`;
  - `restore_best()` carrega o melhor estado; levanta erro se não houve validação.
  - O modelo é passado explicitamente ao callback (não é procurado por nome de atributo, como faz o `GradNormCallback`, PD-35).
- `Trainer`: atributo `stop_training = False`. `TrainingLoop`: zera o sinal no início, para ao fim da época em que ele for marcado
  e devolve `history["epochs_run"]`. Um `Trainer` sem o atributo (como os de mentira dos testes antigos) nunca para.
- `tests/test_best_model_callback.py`: 7 testes (melhor época e restauração dos pesos; parada após a paciência; `min_delta`
  comparado ao melhor, não à época anterior; épocas sem validação não contam; erros; o sinal é zerado no início; com o `Trainer`
  real). Suíte: **126 testes, 122 ok, 3 falhas conhecidas (PD-11), 1 pulado**.
- Nada é gravado em disco: o melhor estado fica em memória (~98 MB em CPU para a MarkerUNet).

### 18.8 Peça 4 — k-fold e treino final (2026-10-07; aguardando validação, sem commit)
- `src/training/kfold.py`:
  - `TrainConfig`: `k=5`, `batch_size=4`, `max_epochs=200`, `patience=20`, `min_delta=0`, `lr=1e-4`, `grad_clip=1.0`,
    `threshold=0.5`, `seed=42`, `device`, `log_every` — tudo parametrizável por experimento (decisões do §18.4).
  - `make_folds(ids, k, seed)`: *folds* disjuntos por imagem, tamanhos o mais iguais possível (37 em 5 → 8, 8, 7, 7, 7), aleatório
    com semente, sem estratificação.
  - `run_kfold(factory, samples, ids, config)`: para cada *fold*, chama `factory()` para montar um experimento **novo** (modelo,
    pipeline e compositor de perdas), treina com aumentação por época e shuffle, `BestModelCallback` com paciência, restaura a
    melhor época e avalia a validação com `evaluate` (uma linha por imagem, com `fold` e a linha de base do Cellpose). As sementes
    variam por *fold* (`seed + fold`).
  - `final_epochs(folds)`: mediana das melhores épocas, arredondada.
  - `run_final(factory, train, test, config, num_epochs)`: treina com todas as imagens de treino por `num_epochs`, sem validação,
    e avalia o teste **uma vez**.
- **O *cosine schedule* cobre sempre `max_epochs`.** No treino final, ele é planejado para `max_epochs` e o treino só para em
  `num_epochs`. Assim a taxa de aprendizado em cada época é a mesma que os *folds* viram até a melhor época; com `T_max = num_epochs`,
  ela cairia a zero mais cedo e o treino final não reproduziria o dos *folds*.
- O otimizador é Adam sobre todos os parâmetros do modelo. ❓ A ablação de congelar o encoder ou usar *param groups* (PD-38) vai
  exigir que a `factory` (ou a configuração) escolha o otimizador; fica para quando a ablação for feita.
- `tests/test_kfold.py`: 6 testes (divisão disjunta, completa, equilibrada e reproduzível; erros de argumento; um modelo novo por
  *fold* e **nenhuma imagem de validação no treino**; reproduzibilidade das métricas e da curva de perda; mediana; treino final sem
  validação e avaliação do teste). Suíte: **132 testes, 128 ok, 3 falhas conhecidas (PD-11), 1 pulado**.
- **Ensaio com as peças reais** (CPU, script `ensaio_kfold_real.py` no scratchpad): `MarkerUNet` + `ScribblePromptingNetwork`
  reais sobre os `.npy` regerados, recorte de 6 imagens de treino e 2 de teste, `k=2`, 1 época, Dice + TV. O fluxo inteiro roda
  (2 *folds* + treino final + teste) em 26 s. O Dice de validação que o `BestModelCallback` registra é **igual** à média das linhas
  do `evaluate` em cada *fold* (0,352 e 0,411), ou seja, as duas medições batem. Os valores em si (0,15 a 0,53 contra 0,80 a 0,88 do
  Cellpose) não significam nada: 1 época com 3 imagens.

