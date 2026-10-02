# 07 — Testes

> Legenda: ✅ verificado (os testes foram **executados** nesta revisão) · 📜 histórico · ❓ a confirmar.
> Problemas ficam em [08-pendencias.md](08-pendencias.md) e são citados pelo ID.
>
> Execução: 2026-09-27, commit `f9b1e9b`, em CPU, num ambiente temporário fora do repositório (torch 2.14 CPU, smp 0.5.0,
> scribbleprompt `182c449`, pytest 9.1). Nada foi instalado no projeto.

---

## 1. Resumo

> **Atualização (2026-09-29, etapa 1):** entrou `test_monuseg_dataset.py` (7 testes da conversão XML → máscara). A suíte passou a
> ter **81 testes em 13 arquivos: 77 passam, as mesmas 3 falhas (PD-11) e 1 pulado**, em ~4 s (torch 2.14 CPU, scikit-image
> 0.24.0). O resto deste documento descreve o estado de 2026-09-27.
>
> **Atualização (2026-10-01, C4):** entrou `test_cellpose_step.py` (5 testes da checagem do nome do modelo, com um `cellpose` falso).
> Agora são **86 testes em 14 arquivos: 82 passam, 3 falhas (PD-11) e 1 pulado**.
>
> **Atualização (2026-10-01, C7 + PD-34):** +2 em `test_cellpose_step.py` (chaves do `forward`), +5 em `test_rgba_step.py`
> (`alpha`), +1 em `test_save_results_step.py` (`float16`). Total: **94 testes: 90 passam, 3 falhas (PD-11) e 1 pulado**.

- Existem **74 testes em 12 arquivos** ([tests/](../../tests/)), escritos entre julho e agosto junto com as correções da
  investigação do exp. 1. ✅
- **Resultado da execução: 70 passam, 3 falham e 1 é pulado, em ~3 segundos.** ✅
  - As **3 falhas** são todas de `test_object_size_loss.py`: os testes esperam o Size simétrico, que foi revertido de propósito em
    30/08 (PD-11).
  - O teste **pulado** verifica o erro quando o pacote `scribbleprompt` não está instalado, e ele estava instalado.
- **O que os testes cobrem:** a "tubulação". Chaves do dicionário, formatos, faixas de valores, o gradiente chegando à
  MarkerUNet, a rede final congelada, a ordem dos passos no `Trainer` e no `TrainingLoop`. Fazem isso com **redes de brinquedo**
  (dummies, fakes), não com as redes reais.
- **O que não cobrem**, e é o que importa para o TCC:
  - as redes reais (a MarkerUNet do smp e o ScribblePrompt de verdade);
  - a conversão XML → máscara;
  - o Cellpose;
  - a matemática das losses;
  - o código dos notebooks (carregamento, aumentação, métricas).

  A frase "não temos praticamente nada implementado" é justa **para a parte científica**: os testes garantem que as peças se
  encaixam, não que os resultados estão certos (PD-48).
- Os testes **não rodam automaticamente**: o `pytest` não está no `requirements.txt`, não há CI, e o ambiente local não tem as
  dependências (PD-48).

---

## 2. Como rodar

No Colab, depois das células de setup de qualquer notebook:

```bash
pip install pytest
python -m pytest tests -q
```

Localmente, é preciso um ambiente com as dependências do `requirements.txt`. Os testes **não usam GPU nem Cellpose**, então CPU
basta:

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt pytest            # cellpose incluso, embora nenhum teste o use
python -m pytest tests -q
```

❓ Esta receita completa não foi executada exatamente assim. Nesta revisão foi usado só o necessário: torch/torchvision CPU,
`segmentation-models-pytorch==0.5.0`, `scribbleprompt` (git, `--no-deps`), `segment-anything==1.0`, `numpy`, `scipy`,
`opencv-python-headless`, `loguru`, `pyyaml` e `pytest`, com `PYTHONPATH=.` na raiz do repositório.

---

## 3. O que cada arquivo testa ✅

| Arquivo | Nº | O que verifica | Com o quê | Situação |
|---|---|---|---|---|
| `test_distance_map_step.py` | 6 | faixa [0,1], inversão (centro 0, fundo 1), chaves | máscaras sintéticas 5×5 | ✅ passa |
| `test_rgba_step.py` | 1 | RGBA float32 em [0,1] | array sintético | ✅ passa |
| `test_save_results_step.py` | 6 | grava `.npy`, preserva float32, chaves padrão e personalizadas, erros | `tmp_path` | ✅ passa |
| `test_monuseg_preprocessed_dataset.py` | 1 | roda o pipeline e preserva os campos | dataset e pipeline dummy | ✅ passa |
| `test_marker_step.py` | 8 | modo diferenciável (tensor, gradiente), inferência (binário), fallback, layouts | `DummyMarkerNet` | ✅ passa |
| `test_frozen_segmentation_step.py` | 7 | chave `segmentation`, erros, classe abstrata | rede final dummy | ✅ passa |
| `test_scribble_prompting_network.py` | 14 | sigmoid na saída, entrada NumPy, pesos congelados, gradiente até os scribbles, `predict` sem grad, resize, modos `sharpened`/`dense_soft` | **pacote falso** (`FakeScribblePromptUNet`, via `monkeypatch`) | ✅ 13 passam, 1 pulado |
| `test_monuseg_dataset.py` (2026-09-29) | 7 | rasterização pelo centro do pixel (área exata), descarte de regiões degeneradas, "o menor vence" na sobreposição, rótulos consecutivos, leitura de vários `<Annotation>`, área ≈ anotada num XML real | XMLs sintéticos em `tmp_path` + 1 XML real do teste (pulado se faltar) | ✅ passa |
| `test_object_size_loss.py` | 4 | Size **simétrico** `|ratio−1|` | tensores sintéticos | ❌ **3 falham** (PD-11) |
| `test_training_pipeline.py` | 4 | ordem dos Steps, erros, alias `__call__` | Steps que só adicionam uma chave | ✅ passa |
| `test_trainer.py` | 8 | o passo otimiza, o eval não otimiza, gradiente, callbacks, scheduler, chaves | rede e composer dummy | ✅ passa |
| `test_training_loop.py` | 10 | épocas e batches, histórico, `validate_every`, callbacks, validações | `FakeTrainer` | ✅ passa |
| `test_training_integration.py` | 5 | modos train/eval, gradiente através da rede congelada, DMap sobre os marcadores, laço completo com as correções | MarkerNet de 2 convs + rede final dummy, com **as losses e o Trainer reais** | ✅ passa |

📜 A maioria dos testes nasceu junto com as correções de agosto: os docstrings citam P1, P3, P5, C2, C3/C4 e E1. Eles protegem essas
correções.

---

## 4. Cobertura ✅ (medida com `pytest-cov`)

| Grupo | Arquivos | Cobertura de linhas |
|---|---|---|
| Treino | `trainer.py` 94%, `training_loop.py` 100%, `grad_norm_callback.py` 93% | alta |
| Steps de inferência | `marker_step.py` 86%, `frozen_segmentation_step.py` 91% | alta |
| Persistência / pré-processamento | `save_results_step.py` 100%, `distance_map_step.py` 100%, `rgba_step.py` 64% | alta a média |
| Losses | Dice, RMSE, DMap, Size 100% (só por execução); `terms.py` 81%; TV 42%; Border 29%; NotTooThin 23% | média; ver a ressalva abaixo |
| Rede final | `scribble_prompting_network.py` 66% (**com pacote falso**) | média |
| Dados | `base_dataset.py` 43%; `monuseg_preprocessed_dataset.py` 79% | baixa a média |
| I/O e utilitários | `output_writer.py` 37%; `image_utils.py` 47% | baixa |
| **Nunca importados pelos testes (0%)** | ~~`monuseg_dataset.py`~~ (coberto desde 2026-09-29; cobertura não remedida), **`cellpose_step.py`**, **`marker_unet.py`** (+ config), `preprocessing_pipeline.py`, `model_pipeline.py` | **nula** |
| **Total** (só dos arquivos importados) | | 75% |

**Ressalvas:**
- **Linha coberta não é comportamento verificado.** A `DistanceMapLoss` tem 100% de cobertura porque o teste de integração a
  **executa**, mas nenhum teste confere o valor que ela devolve. O mesmo vale para Dice e RMSE.
- O total de 75% **não conta** os arquivos que os testes nunca importam; eles ficam de fora da conta.

---

## 5. O que não está testado (e importa)

| Lacuna | Por que importa | Ligação |
|---|---|---|
| ~~**Conversão XML → máscara**~~ (`MonusegDataset`) | ✅ coberta desde 2026-09-29 (`test_monuseg_dataset.py`) | PD-07, PD-25 |
| **Redes reais** (MarkerUNet do smp; ScribblePrompt real) | os testes usam fakes. A entrada de 5 canais, os 128², a falta de antialias e o canal negativo denso só foram vistos no oráculo | PD-37, PD-04 |
| **Valores das losses** contra as fórmulas da tese | nenhuma conferência numérica de Dice, DMap, TV, Border e Size | PD-06, PD-12 |
| **Mapa de distância com vários núcleos de tamanhos diferentes** | o teste atual usa **um** objeto 3×3. Ele passa tanto com a normalização por imagem quanto por núcleo, então **não detecta** o PD-06 | PD-06 |
| **Código dos notebooks** (`load_preprocessed`, `build_batch`, aumentação, `compute_binary_metrics`) | é o código que gera os números do TCC, copiado em cada notebook e sem nenhum teste | PD-19, PD-47 |
| **`CellposeStep`** | exige GPU; o bug do `MODEL_LIST` (PD-29) passou sem ser notado. Desde 2026-10-01 a checagem do nome tem teste (`test_cellpose_step.py`); o `forward` continua sem teste | PD-29, PD-30 |
| **Ponta a ponta com dados reais** | nenhum teste roda uma imagem do MoNuSeg pelo pipeline | — |

---

## 6. Testes que vão precisar mudar com as correções já decididas

| Teste | Por quê | Pendência |
|---|---|---|
| `test_object_size_loss.py` (3 testes) | já falha: espera a fórmula simétrica, mas a fórmula certa é a da tese | PD-11 |
| `test_distance_map_step.py::test_compute_distance_map_inverts_normalized_distance` | continua passando depois da correção (um objeto só), mas precisa de um caso com **dois núcleos de tamanhos diferentes**, que é o que o PD-06 corrige | PD-06 |
| `test_marker_step.py::test_forward_without_model_falls_back_to_segmentation` | **garante** o fallback silencioso que a PD-16 quer tornar explícito | PD-16 |
| `test_frozen_segmentation_step.py::test_frozen_segmentation_step_adds_segmentation_key` e o `Trainer` (`prediction_key`) | a saída final vai virar `final_segmentation` | PD-33 |
| `test_scribble_prompting_network.py` (modos) | falta o modo "só positivo" e o `input_size` configurável | PD-04, PD-05, PD-44 |

---

## 7. Recomendação: testes para escrever junto com o trabalho planejado

Na ordem do plano ([06-experimentos.md §5](06-experimentos.md)), cada correção já pode nascer com o seu teste:

1. **Métricas** (Dice, IoU, precisão, recall e massa binários): comparar com casos calculados à mão. É o que sustenta as tabelas
   da publicação (PD-47).
2. **Módulo de dados/k-fold** (PD-19):
   - os *folds* têm ids **disjuntos** e cobrem as 30 imagens;
   - a aumentação aplica **a mesma** transformação a imagem, rgba, GT e Dmap;
   - com a mesma seed, os sorteios se repetem.
3. **Mapa de distância por núcleo** (PD-06): com dois núcleos de raios 20 e 6, o centro dos **dois** vale 0.
4. **XML → máscara:** o número de instâncias bate com o de regiões do XML, e a máscara binária bate com o `ground_truth` persistido.
5. **Losses:** valor de cada uma num caso pequeno calculado à mão, seguindo as eqs. 5.5–5.8 da tese.
6. **Teste lento, opcional, com o ScribblePrompt real** (marcado como `@pytest.mark.slow`): carrega o checkpoint e confere o
   formato da saída e que o oráculo "GT inteiro" dá Dice alto numa imagem. Pode reaproveitar o notebook do oráculo.
7. **Infraestrutura:** `pytest` num `requirements-dev.txt` e, como o repositório é público, um GitHub Actions em CPU. Os testes
   rodam em segundos.

---

## 8. Dúvidas para você validar

1. **Escrever os testes junto com as correções** (itens 1 a 5 do §7), em vez de numa etapa separada?
2. **CI:** quer um GitHub Actions rodando os testes a cada push ou PR? Custa pouco e evita, por exemplo, o PD-11 passar
   despercebido por um mês.
3. **Teste com o ScribblePrompt real (item 6):** vale ter, sabendo que baixa o checkpoint e leva alguns segundos em CPU?
