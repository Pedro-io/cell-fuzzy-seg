# cell-fuzzy-seg

Geração automática de marcadores difusos para a segmentação de núcleos celulares em imagens histológicas (MoNuSeg). Uma rede
(MarkerUNet) propõe marcadores a partir da imagem e da segmentação do Cellpose-SAM, e uma rede de segmentação interativa
congelada (ScribblePrompt-UNet) transforma esses marcadores na segmentação final. O treino é de ponta a ponta, com o gradiente
atravessando a rede congelada. O trabalho segue a ideia de aprendizado de marcadores de Fonseca (2026), sem a camada de
segmentação hierárquica usada lá.

## Licença

- **Código:** licença MIT ([LICENSE](LICENSE)).
- **Dados** em [data_source/](data_source/): o MoNuSeg e tudo o que foi derivado dele (inclusive `MoNuSegPreprocessed/`) são
  distribuídos sob a **CC BY-NC-SA 4.0**, que proíbe o uso comercial. A licença MIT do código não se aplica a essa pasta. Origem,
  citações e alterações em [data_source/README.md](data_source/README.md).
- **Bibliotecas e modelos de terceiros** mantêm as licenças deles (tabela abaixo). Nenhum peso de modelo é distribuído neste
  repositório: os pesos do Cellpose-SAM, do ScribblePrompt e do encoder ImageNet são baixados na execução.

## Créditos e citações

Quem usar este trabalho deve citar também o que ele usa:

| Componente | Uso aqui | Licença | Citação |
|---|---|---|---|
| MoNuSeg | dados de treino e teste | CC BY-NC-SA 4.0 | Kumar et al. (2017); Kumar et al. (2020) |
| Cellpose 4.1.1, modelo `cpsam` (Cellpose-SAM) | segmentação inicial, congelada | código: BSD-3-Clause. Segundo os autores, o modelo foi treinado com dados CC BY-NC | Pachitariu, Rariden e Stringer (2025) |
| ScribblePrompt-UNet (commit `182c449`, checkpoint v1) | camada de segmentação, congelada | código: Apache-2.0 | Wong et al. (2024) |
| segmentation-models-pytorch 0.5.0 | U-Net da MarkerUNet | MIT | Iakubovskii (2019) |
| Encoder ResNet34 pré-treinado no ImageNet | encoder da MarkerUNet | conforme a biblioteca | He et al. (2016) |
| Tese de Fonseca (2026) | base teórica: marcadores difusos aprendidos de ponta a ponta e suas regularizações | | Fonseca (2026) |

Também são usados PyTorch e torchvision (licenças no estilo BSD), NumPy e SciPy (BSD-3-Clause), scikit-image (BSD-3-Clause),
OpenCV (Apache-2.0), segment-anything 1.0 (Apache-2.0, exigido pelo pacote do ScribblePrompt) e loguru (MIT). As versões estão
em [requirements.txt](requirements.txt).

### Referências

- Fonseca, G. B. (2026). *Hierarchy-Based Fuzzy Segmentation and Marker Learning*. Tese de doutorado, PUC Minas e Université
  Gustave Eiffel. <https://theses.hal.science/tel-05743320>
- He, K.; Zhang, X.; Ren, S.; Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. arXiv:1512.03385.
- Iakubovskii, P. (2019). *Segmentation Models Pytorch*. GitHub. <https://github.com/qubvel/segmentation_models.pytorch>
- Kumar, N.; Verma, R.; Sharma, S.; Bhargava, S.; Vahadane, A.; Sethi, A. (2017). A dataset and a technique for generalized
  nuclear segmentation for computational pathology. *IEEE Transactions on Medical Imaging*, 36(7), 1550-1560.
  doi:10.1109/TMI.2017.2677499
- Kumar, N. et al. (2020). A multi-organ nucleus segmentation challenge. *IEEE Transactions on Medical Imaging*, 39(5),
  1380-1391. doi:10.1109/TMI.2019.2947628
- Pachitariu, M.; Rariden, M.; Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation.
  *bioRxiv*. doi:10.1101/2025.04.28.651001
- Wong, H. E.; Rakic, M.; Guttag, J.; Dalca, A. V. (2024). ScribblePrompt: fast and flexible interactive segmentation for any
  biomedical image. *ECCV 2024*. arXiv:2312.07381
