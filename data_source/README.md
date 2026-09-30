# Dados

Esta pasta contém o dataset **MoNuSeg** (Multi-Organ Nucleus Segmentation, desafio do MICCAI 2018) e dados derivados dele.

## Origem e licença

- **Fonte:** [MoNuSeg Grand Challenge](https://monuseg.grand-challenge.org/Data/). As imagens vêm do arquivo TCGA
  (The Cancer Genome Atlas): lâminas H&E, ampliação 40× (3 imagens de treino estão em 20×; ver PD-24 em
  `docs/estudo/08-pendencias.md`). Baixado de novo em 2026-09-29 (PD-23). ❓ Falta registrar a URL exata do arquivo baixado.
- **Licença:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Os dados desta pasta, **incluindo os derivados**, são distribuídos sob a mesma licença e **não podem ser usados para fins
  comerciais**. A licença do código deste repositório é independente e não se aplica aos dados.
- **Citação exigida pelos autores:**

  > N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear
  > Segmentation for Computational Pathology," *IEEE Transactions on Medical Imaging*, vol. 36, no. 7, pp. 1550–1560, 2017.

  > N. Kumar et al., "A Multi-organ Nucleus Segmentation Challenge," *IEEE Transactions on Medical Imaging*.

## Conteúdo

| Pasta | Conteúdo | Origem |
|---|---|---|
| `MoNuSegTrainingData/Tissue_Images/` | 37 imagens `.tif` RGB 1000×1000 | download oficial ("MoNuSeg 2018 Training Data") |
| `MoNuSegTrainingData/Annotations/` | contornos dos núcleos em `.xml` (24.140 regiões) | download oficial |
| `MoNuSegTestData/Tissue_Images/`, `Annotations/` | 14 imagens de teste e seus contornos (6.697 regiões) | download oficial |
| `MoNuSegPreprocessed/` | **dados derivados**: segmentação do Cellpose, RGBA, ground truth rasterizado e mapa de distância, em `.npy` | gerado por `notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb` |

### Alterações em relação ao original

- Arquivos originais **sem alteração de conteúdo** (conferido por SHA-256). A organização das pastas foi ajustada: o zip de
  treino traz `MoNuSeg 2018 Training Data/{Annotations,Tissue Images}/`, e o de teste traz os arquivos soltos. Aqui os dois
  seguem `Tissue_Images/` + `Annotations/`, como espera o `configs/datasets.yml`. Os arquivos de metadados do macOS
  (`__MACOSX/`, `.DS_Store`) foram descartados.
- As pastas `Binary_masks/` e `Binary_masks_instance/`, que existiam aqui até 2026-09-28, **não fazem parte** do download
  oficial e foram removidas (PD-26). Continuam no histórico do git.
- `MoNuSegPreprocessed/` não faz parte do MoNuSeg: é gerado a partir dele. O ground truth foi rasterizado a partir dos XMLs
  como máscara binária. Os demais arquivos são saídas do pipeline de pré-processamento deste projeto. ⚠️ Hoje ele cobre só
  30 das 37 imagens de treino, e será regerado (PD-23). Detalhes em
  [`docs/estudo/01-dados.md`](../docs/estudo/01-dados.md).
