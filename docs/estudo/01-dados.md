# 01 — Dados

> Legenda: ✅ verificado no código ou medido nos arquivos · 📜 histórico/commits · 📖 fonte externa · ❓ a confirmar.
> Problemas encontrados ficam em [08-pendencias.md](08-pendencias.md) e são citados aqui pelo ID (`PD-nn`).
>
> Última verificação: 2026-09-27 (commit `f9b1e9b`). Os números da §4 foram medidos nesta data com um script à parte
> sobre os arquivos de `data_source/`; nada no repositório foi alterado para isso.

---

## 1. Resumo

- O TCC usa o **MoNuSeg**: 30 imagens de treino e 14 de teste, todas RGB 1000×1000, de lâminas H&E do TCGA, com os
  contornos de cada núcleo anotados em XML. ✅📖
- O código transforma os polígonos do XML numa **máscara binária** (núcleo = 1, fundo = 0) e perde a separação entre núcleos
  que se tocam. ✅ (PD-07)
- O pré-processamento (Cellpose, RGBA, mapa de distância) roda **uma vez** e é salvo em
  `data_source/MoNuSegPreprocessed/` como `.npy`. Os experimentos **leem esses arquivos com código próprio de cada notebook**,
  não pelas classes de dataset do `src/`. ✅ (PD-19)

---

## 2. O dataset MoNuSeg 📖

**MoNuSeg** (*Multi-Organ Nucleus Segmentation*) foi o desafio de segmentação de núcleos do MICCAI 2018. Segundo a página
oficial do desafio ([monuseg.grand-challenge.org/Data](https://monuseg.grand-challenge.org/Data/)):

- as imagens vêm do arquivo **TCGA** (*The Cancer Genome Atlas*), de pacientes com tumores de órgãos diferentes;
- coloração **H&E** (hematoxilina e eosina: a hematoxilina deixa os núcleos roxo-escuros), ampliação **40×**;
- **30 imagens de treino** (~22.000 contornos de núcleo) e **14 de teste** (~7.000);
- anotações em **XML** (o desafio fornecia código MATLAB para converter em máscaras binárias ou por instância);
- licença **CC BY-NC-SA 4.0** (atribuição, uso não comercial, mesma licença). Ver PD-27.

Citações exigidas pelo desafio:
- N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane e A. Sethi, "A Dataset and a Technique for Generalized Nuclear
  Segmentation for Computational Pathology", *IEEE Transactions on Medical Imaging*, 36(7):1550–1560, 2017. O artigo
  fala em "mais de 21.000" núcleos anotados.
- Kumar et al., "A Multi-organ Nucleus Segmentation Challenge", *IEEE Transactions on Medical Imaging* (a página indica
  "in press"; ❓ conferir volume e ano antes de citar).

Na literatura, o treino é descrito como 7 órgãos: mama, fígado, rim, próstata, bexiga, cólon e estômago. ❓ Isso vem de
um resumo de busca, não da página oficial; conferir no artigo antes de usar no texto.

**Por que o MoNuSeg é difícil para este pipeline** ✅ (medido, §4):
- centenas de objetos pequenos por imagem (mediana de ~400 núcleos, com um máximo de 1.863);
- núcleos colados uns nos outros;
- variação de escala entre imagens (PD-24).

---

## 3. O que há no disco ✅

```
data_source/                                   (1,6 GB, versionado no git — PD-21)
├── MoNuSegTrainingData/                       454 MB
│   ├── Tissue_Images/*.tif        30   RGB uint8 1000×1000 (TIFF com compressão LZW)
│   ├── Annotations/*.xml          30   polígonos dos núcleos  ← usado pelo código
│   ├── Binary_masks/*.png         30   máscara binária (download original)   ← NÃO usado (PD-26)
│   └── Binary_masks_instance/*.npy 30  máscara por instância, int64 (idem)    ← NÃO usado (PD-26)
├── MoNuSegTestData/                           58 MB
│   ├── Tissue_Images/*.tif        14
│   └── Annotations/*.xml          14   (não há máscaras prontas para o teste)
└── MoNuSegPreprocessed/                       1,1 GB (gerado pelo pré-processamento)
    ├── meta.json                  {"work_size": 1000, "segmentation_source": "cellpose", splits 30/14, created_at 2026-08-16}
    └── {train,test}/<chave>/<id>.npy
```

O `<id>` é o nome do arquivo sem extensão (ex.: `TCGA-18-5592-01Z-00-DX1`), o mesmo nos `.tif`, nos `.xml` e nos `.npy`.

### Formato do XML ✅

O formato parece ser o do software de anotação da Aperio (ImageScope) ❓. Na hierarquia
`Annotations > Annotation > Regions > Region > Vertices > Vertex(X, Y)`, cada `Region` é o contorno de **um** núcleo,
com coordenadas fracionárias em pixels. O cabeçalho traz `MicronsPerPixel`, a escala física, que varia entre as imagens (PD-24).
Cada `Region` também traz `Area` e `Length` calculados pelo software; o código não usa esses campos.

### Arrays persistidos ✅

| Chave | Forma | Tipo | Conteúdo |
|---|---|---|---|
| `image` | 1000×1000×3 | uint8 | cópia exata do `.tif` (verificado pixel a pixel) |
| `ground_truth` | 1000×1000 | uint8 | 0/1, idêntico à rasterização do XML feita pelo código |
| `segmentation` | 1000×1000 | uint16 | instâncias do Cellpose (0 = fundo, 1..K = núcleos) |
| `rgba` | 1000×1000×4 | float32 | RGB em [0,1] + alpha = `segmentation > 0` (verificado) |
| `distance_map` | 1000×1000 | float32 | `1 − EDT/max(EDT)` sobre o GT, em [0,1] (PD-06) |

---

## 4. Números medidos ✅

> ⚠️ **Desatualizado para o treino (2026-09-29).** O download novo do MoNuSeg tem **37** imagens de treino (24.140 núcleos), e
> não 30. As 30 abaixo continuam idênticas, e a coluna do teste continua valendo. A tabela do treino será refeita com as 37 na
> etapa 1 do plano (XML → máscara). Ver PD-23.

| | Treino (30) | Teste (14) |
|---|---|---|
| Núcleos anotados (regiões nos XMLs) | **16.966** | **6.697** |
| Núcleos por imagem (mín / mediana / máx) | 294 / 403 / 1.863 | 249 / 466 / 818 |
| Componentes conexos no GT binário | 12.648 (**−25,5%**) | 6.086 (−9,1%) |
| Pixels cobertos por mais de um polígono | 128.007 | 11.236 |
| Fração de pixels de núcleo (média; mín–máx) | 24,6% (11,9–42,7%) | 21,7% (10,3–30,4%) |
| Área do núcleo, mediana das medianas por imagem | 438 px (≈ 24 px de diâmetro) | 440 px |
| Área, 5% menores / 5% maiores (mediana por imagem) | 175 px (≈ 15 px) / 1.062 px (≈ 37 px) | 163 px / 794 px |
| Instâncias encontradas pelo Cellpose | 13.289 (78% das anotadas) | 6.374 (95%) |
| Fração de pixels marcada pelo Cellpose | 19,3% | 18,9% |

"Diâmetro" é o diâmetro do círculo de mesma área. Leituras:

- **A fusão é bem maior no treino** (−25,5%) que no teste (−9,1%). Boa parte vem das imagens `TCGA-HE-7128/7129/7130`
  (20×, núcleos pequenos e densos) e de `TCGA-KB-A93J` e `TCGA-RD-A8N9` (> 1.100 núcleos cada). Treino e teste têm perfis
  diferentes, e isso afeta a generalização. ❓
- **O Cellpose encontra menos núcleos no treino** (78%) do que no teste (95%), coerente com as imagens densas do treino.
- **O primeiro plano é ~1/4 da imagem**: o desbalanceamento é moderado (PD-13).
- **Em que escala o modelo vê um núcleo:** a MarkerUNet trabalha em 256² (fator 3,9), e o núcleo mediano fica com ~6 px; o
  ScribblePrompt trabalha em 128² (fator 7,8), e ele fica com **~3 px** (PD-05).
- **A contagem de núcleos não bate com a oficial** (16.966 contra ~22.000 no treino). Ver PD-23 antes de citar.
  **Causa (2026-09-29):** faltavam 7 das 37 imagens de treino. Com elas, o treino tem 24.140 núcleos.

---

## 5. Do XML à máscara ✅

> **Atualizado em 2026-09-29 (etapa 1, PD-25/PD-07).** A rasterização mudou. A versão abaixo, com `int()` + `cv2.fillPoly`,
> fica como histórico: é a que gerou o `MoNuSegPreprocessed/` atual e todos os números dos exp. 1 a 6 e do oráculo.

**Versão atual** — [xml_to_instance_mask](../../src/data/load/monuseg_dataset.py) (base medida em
[09-correcoes-pontuais.md §13](09-correcoes-pontuais.md)):

```python
for vertices in read_xml_regions(xml_path):               # todos os núcleos, de todos os <Annotation>
    if len(vertices) < 3 or area == 0: descarta            # 5 cliques soltos no treino
    pixels = skimage.draw.polygon(y - 0.5, x - 0.5)        # pixel entra se o CENTRO está dentro (convenção de canto)
# pinta do maior para o menor → na sobreposição, o menor vence; rótulos 1..N na ordem do XML
ground_truth_instances = rótulos (int32);  ground_truth = rótulos > 0 (uint8 0/1)
```

A área rasterizada fica a 0,995 (treino) e 1,011 (teste) da área anotada, contra ~1,10 antes. Das 24.135 regiões válidas do
treino, 24.133 ocupam pelo menos um pixel. No teste, as 6.697 ocupam.

**Versão antiga (até `9a59c29`)** — `git show 9a59c29:src/data/load/monuseg_dataset.py`, linhas 60-72:

```python
mask = np.zeros(shape, dtype=np.uint8)
for region in root.iter("Region"):                         # todos os núcleos, de todos os <Annotation>
    points = [[int(x), int(y)] for cada Vertex]            # int(): na convenção de canto, é o certo (PD-25)
    cv2.fillPoly(mask, [points], 1)                        # pinta a borda também (~+10% de área) e com 1 → BINÁRIA
```

Consequências (da versão antiga; a 1 e a 2 continuam valendo para o `ground_truth` binário, e a versão por instância as resolve):

1. **Núcleos que se tocam ou se sobrepõem viram um único objeto**, porque todos são pintados com o mesmo valor. Para Dice/IoU
   por pixel (a métrica escolhida pelo coordenador) isso não muda nada. Muda para o **mapa de distância** (PD-06) e para
   qualquer métrica ou loss por objeto. (PD-07)
2. **Sobreposições desaparecem:** um pixel coberto por dois polígonos vale 1 do mesmo jeito.
3. O arquivo `TCGA-HE-7129` tem dois blocos `<Annotation>`; o segundo está vazio, então não gera duplicatas.

**Alternativa que já está no disco:** `Binary_masks_instance/` tem, para as 30 imagens de treino, uma máscara por instância
cujo maior rótulo é igual ao número de regiões do XML. Poderia servir para calcular o mapa de distância por núcleo. Mas
falta a versão do teste e não se sabe como foi gerada (PD-26). Gerar a versão por instância a partir do XML seria simples:
pintar cada polígono com o próprio índice em vez de 1.

---

## 6. As classes de dataset ✅

### 6.1 `BaseDataset`: [base_dataset.py](../../src/data/load/base_dataset.py)

Classe abstrata (herda de `torch.utils.data.Dataset`). No construtor, ela:
1. lê [configs/datasets.yml](../../configs/datasets.yml) (caminho fixo em [base_dataset.py:28](../../src/data/load/base_dataset.py#L28));
2. escolhe `dataset_name` (ex.: `monuseg`) e `config_key` (ex.: `monuseg_training`);
3. exige `root_dir`, `image_dir`, `mask_dir`, `image_extension` e `mask_extension`
   ([base_dataset.py:95-105](../../src/data/load/base_dataset.py#L95-L105));
4. guarda `loader_config` e `preprocessing` do YAML, que **nenhum código de treino usa** (PD-18).

**Por quê** 📜: permitir trocar de base de dados só pelo YAML, mantendo a mesma interface (ver o docstring da classe; o antigo
`ARCHITECTURE.md`, hoje substituído pelo [02](02-pipeline-e-arquitetura.md), exigia que todo dataset herdasse desta classe). Na prática só existe o MoNuSeg.

### 6.2 `MonusegDataset`: [monuseg_dataset.py](../../src/data/load/monuseg_dataset.py)

- `_get_file_pairs` ([L97-126](../../src/data/load/monuseg_dataset.py#L97-L126)) lista os `.tif` **em ordem alfabética**
  e exige um `.xml` com o mesmo nome.
- `_load_image` ([L52-58](../../src/data/load/monuseg_dataset.py#L52-L58)) usa `cellpose.io.imread`, o mesmo leitor do
  Cellpose, e devolve RGB uint8.
- `_load_mask` aceita `.xml` (rasteriza, §5), `.npy` ou imagem.
- `__getitem__` ([L31-50](../../src/data/load/monuseg_dataset.py#L31-L50)) devolve
  `{"id", "image", "ground_truth", "meta": {image_path, mask_path}}` e aplica `transform` no dicionário inteiro, se houver.
- Não redimensiona, não normaliza e não aumenta os dados.

### 6.3 `MonusegPreprocessedDataset`: [monuseg_preprocessed_dataset.py](../../src/data/load/monuseg_preprocessed_dataset.py)

Envolve um dataset bruto e **roda o pipeline de pré-processamento a cada `__getitem__`**
([L66-78](../../src/data/load/monuseg_preprocessed_dataset.py#L66-L78)), sobre uma cópia profunda da amostra. Portanto
**não lê os `.npy` do disco**. Só é usado no notebook-tutorial `preprocessamento_monuseg.ipynb`. (PD-19)

---

## 7. Como os dados chegam ao treino na prática ✅

### 7.1 Geração (uma vez): `notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb`

1. Carrega as 30 + 14 amostras com `MonusegDataset` (configs `monuseg_training` e `monuseg_test`).
2. `resize_sample` redimensiona para `WORK_SIZE = 1000`, o que **não muda nada**, porque as imagens já têm 1000×1000. O GT
   usaria `INTER_NEAREST` se precisasse.
3. Roda `CellposeStep → RGBAStep → DistanceMapStep → SaveResultsStep` e grava o `meta.json`.
4. Confere que o que foi salvo é idêntico ao que está em memória.

⚠️ Até 2026-10-01, se o Cellpose não carregasse, o notebook **usava o GT como segmentação**, o que seria vazamento total. Não
aconteceu no run atual (o log diz "Cellpose disponível" e o `meta.json` diz `cellpose`). Desde então, a execução para com erro
(PD-16, parte do notebook).

### 7.2 Leitura (em cada experimento): funções `load_preprocessed` e `build_batch` (no exp. 6, células 8 e 12)

Isso vale para os exp. 3 a 6. Os exp. 1 e 2 são anteriores à persistência: rodam o pré-processamento dentro do próprio
notebook, a partir do `MonusegDataset`, mas usam o mesmo `build_batch`.

- `load_preprocessed(split)` lê `image`, `rgba`, `ground_truth` e `distance_map` de cada `id`, em ordem alfabética. A chave
  `segmentation` do Cellpose **não é lida**: ela chega à rede só como o canal alpha de `rgba`.
- `build_batch` empilha 4 amostras em tensores `(B, C, H, W)`; a imagem fica em [0, 255] float e o resto em [0, 1].
- **Aumentação:** rotação de 90° × k (k sorteado de 0 a 3), flip horizontal (p = 0,5) e flip vertical (p = 0,5), aplicados
  juntos a imagem, rgba, GT e mapa de distância. Como são rotações de 90° e flips, **o mapa de distância continua válido**
  sem recálculo. ✅
- ⚠️ `make_batches(..., augment=True)` roda **uma vez**, antes do treino: a aumentação é sorteada uma vez só, e as épocas
  repetem os mesmos 8 batches na mesma ordem (PD-08).
- **Divisão:** treino = 30 oficiais; validação = **14 oficiais de teste** (PD-02).

---

## 8. Ligações com os próximos temas

| Assunto | Onde será detalhado |
|---|---|
| Cellpose, RGBA, mapa de distância e persistência como Steps | 02 — pipeline e arquitetura |
| Por que a MarkerUNet recebe RGBA em [0,1] sem normalização ImageNet | 03 — modelos |
| Mapa de distância dentro do DMapTerm; normalização por batch | 04 — losses (PD-06, PD-12) |
| Aumentação, batches, validação | 05 — treinamento (PD-08, PD-02) |

---

## 9. Decisões confirmadas (2026-09-27)

| # | Pergunta | Resposta | Onde ficou |
|---|---|---|---|
| 1 | Origem de `Binary_masks/` e `Binary_masks_instance/` | Download original | PD-26 (resta decidir se serão usadas) |
| 2 | Origem do download / contagem de núcleos | Provavelmente o site do desafio; baixar de novo da fonte oficial para conferir | PD-23 (ação do autor) |
| 3 | 1,6 GB de dados no git | Sem problemas até agora; fica como está | PD-21 (adiada) |
| 4 | `data_source/README.md` com atribuição e licença | Sim | ✅ [data_source/README.md](../../data_source/README.md) criado; PD-27 parcial (falta a licença do código) |
| 5 | Imagens em 20× | O autor não sabia | PD-24, agora com o impacto medido no Cellpose e as opções de tratamento |

### Perguntas originais (histórico)

1. **`Binary_masks/` e `Binary_masks_instance/`**: de onde vieram? Do download original, de algum código do orientador ou
   gerados por você? (PD-26) vieram do download original 
2. **Contagem de núcleos**: de onde você baixou o MoNuSeg (site do desafio, Kaggle, outro)? Isso ajuda a explicar os
   16.966 contra ~22.000 (PD-23). acredito que foi do site do desafio, mas pode colocar como pendencia realizar o download direto da fonte para eu verificar
3. **Dados no git**: os 1,6 GB estão versionados para o Colab clonar tudo de uma vez? Tudo bem trocar por Google Drive ou
   Git LFS? (PD-21) Não tive problemas com isso, então podemos deixar por enquanto 
4. **Licença**: posso propor um `data_source/README.md` com a atribuição e a licença CC BY-NC-SA 4.0 do MoNuSeg? (PD-27) sim
5. **Escala**: você sabia das três imagens em 20×? Quer que elas sejam tratadas à parte (reamostradas para 40×, ou com
   diâmetro próprio no Cellpose)? (PD-24) não sabia 
