# Dados

Esta pasta contém o dataset **MoNuSeg** (Multi-Organ Nucleus Segmentation, desafio do MICCAI 2018) e dados derivados dele.

## Origem e licença

- **Fonte:** [MoNuSeg Grand Challenge](https://monuseg.grand-challenge.org/Data/). As imagens vêm do arquivo TCGA
  (The Cancer Genome Atlas): lâminas H&E, ampliação 40× (3 imagens de treino estão em 20×; ver PD-24 em
  `docs/estudo/08-pendencias.md`). Baixado de novo em 2026-09-29 pela página oficial de dados do desafio,
  <https://monuseg.grand-challenge.org/Data/> (arquivos "MoNuSeg 2018 Training Data" e de teste; PD-23).
- **Licença:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Os dados desta pasta, **incluindo os derivados**, são distribuídos sob a mesma licença e **não podem ser usados para fins
  comerciais**. A licença do código deste repositório é independente e não se aplica aos dados.
- **Citação exigida pelos autores:**

  > N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear
  > Segmentation for Computational Pathology," *IEEE Transactions on Medical Imaging*, vol. 36, no. 7, pp. 1550–1560, 2017,
  > doi:10.1109/TMI.2017.2677499.

  > N. Kumar et al., "A Multi-organ Nucleus Segmentation Challenge," *IEEE Transactions on Medical Imaging*, vol. 39, no. 5,
  > pp. 1380–1391, 2020, doi:10.1109/TMI.2019.2947628.

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
- `MoNuSegPreprocessed/` não faz parte do MoNuSeg: é gerado a partir dele, para as 37 imagens de treino e as 14 de teste
  (versão de 2026-10-07, registrada em `MoNuSegPreprocessed/meta.json`). O ground truth é a rasterização dos XMLs: um pixel
  pertence ao núcleo se o seu centro cai dentro do contorno, e os contornos degenerados (menos de 3 vértices ou área zero) são
  descartados. Essa regra é escolha deste projeto, e não dos autores do MoNuSeg. Os demais arquivos são saídas do
  pré-processamento, entre elas a segmentação do Cellpose-SAM, cujo modelo, segundo os autores, foi treinado com dados
  CC BY-NC. Detalhes em [`docs/notas-do-projeto/03-dados.md`](../docs/notas-do-projeto/03-dados.md).
