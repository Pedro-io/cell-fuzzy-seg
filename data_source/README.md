# Dados

Esta pasta contém o dataset **MoNuSeg** (Multi-Organ Nucleus Segmentation, desafio do MICCAI 2018) e dados derivados dele.

## Origem e licença

- **Fonte:** [MoNuSeg Grand Challenge](https://monuseg.grand-challenge.org/Data/). As imagens vêm do arquivo TCGA
  (The Cancer Genome Atlas): lâminas H&E, ampliação 40×.
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
| `MoNuSegTrainingData/Tissue_Images/` | 30 imagens `.tif` RGB 1000×1000 | download original |
| `MoNuSegTrainingData/Annotations/` | contornos dos núcleos em `.xml` | download original |
| `MoNuSegTrainingData/Binary_masks/`, `Binary_masks_instance/` | máscaras binária (`.png`) e por instância (`.npy`) | download original (segundo o autor) |
| `MoNuSegTestData/Tissue_Images/`, `Annotations/` | 14 imagens de teste e seus contornos | download original |
| `MoNuSegPreprocessed/` | **dados derivados**: segmentação do Cellpose, RGBA, ground truth rasterizado e mapa de distância, em `.npy` | gerado por `notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb` |

### Alterações em relação ao original

`MoNuSegPreprocessed/` não faz parte do MoNuSeg: é gerado a partir dele. O ground truth foi rasterizado a partir dos XMLs
como máscara binária. Os demais arquivos são saídas do pipeline de pré-processamento deste projeto. Detalhes em
[`docs/estudo/01-dados.md`](../docs/estudo/01-dados.md).
