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
| notebook, célula 9 | `try/except` → GT como `segmentation` | sem `try/except`: se o Cellpose não carrega, a célula falha |

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
| 1 | `MonusegDataset` (XML → máscara) | PD-25, PD-07 | sim | não | **próxima** |
| 2 | `CellposeStep` | PD-29 (C4), PD-30, PD-31 (C7, `flows`/`styles`), PD-16 (C6, notebook) | sim (PD-30) | sim | — |
| 3 | `RGBAStep` | PD-34 (ideia) | — | — | — |
| 4 | `DistanceMapStep` | PD-06 (depende da instância da etapa 1) | sim | não | — |
| 5 | `SaveResultsStep` | regerar tudo **uma vez**, com commit e parâmetros no `meta.json` (PD-47) | — | sim | — |
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
| Etapa 0 (PD-23, PD-26) | ver `git log -- data_source/` (2026-09-29) | nenhum código mudou | 37 + 14 pares achados pelo `MonusegDataset` real |
| C1 | — | — | — |
| C2 | — | — | — |
| C3 | — | — | — |
| C4 | — | — | — |
| C5 | — | — | — |
| C6 | — | — | — |
| C7 | — | — | — |
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
