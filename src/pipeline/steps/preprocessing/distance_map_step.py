"""Step de pré-processamento que calcula o mapa de distância por núcleo.

O mapa de distância é consumido durante o treinamento pela
:class:`~src.losses.distance_map_loss.DistanceMapLoss` (chave ``"distance_map"``
no dicionário de dados, lida pelo ``Trainer``). Ele é calculado uma única vez por
imagem no :class:`~src.pipeline.preprocessing_pipeline.PreprocessingPipeline` e
persistido como os demais resultados.
"""

from typing import Any, Dict

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects

from src.utils.logger import logger

from ..base_step import PipelineStep


def compute_instance_distance_map(labels: np.ndarray) -> np.ndarray:
    """Calcula o mapa de distância invertido e normalizado **por núcleo**.

    Para cada núcleo (rótulo > 0), aplica a transformada de distância euclidiana (EDT) sobre os
    pixels dele e calcula ``1 − EDT / máx(EDT)`` com o máximo **daquele núcleo**: o centro de todo
    núcleo vale ``0``, e a borda tende a ``1``. O fundo vale ``1``. É o mapa de distância da tese
    (normalização por objeto), consumido pela
    :class:`~src.losses.distance_map_loss.DistanceMapLoss`.

    Detalhes:

    - Núcleos que se tocam são tratados separadamente: a fronteira entre dois núcleos conta como
      borda dos dois.
    - A EDT de cada núcleo é calculada num recorte com 1 pixel de margem em volta dele, o que dá o
      mesmo resultado que calculá-la na imagem inteira (sem a margem, a borda do recorte não teria
      fundo do lado de fora e a distância sairia maior).
    - A borda da imagem não conta como fundo: num núcleo cortado pela borda, o ponto mais barato
      fica junto dela, onde o centro verdadeiro provavelmente está.

    Args:
        labels: Máscara por instância ``(H, W)`` de inteiros não negativos: ``0`` é fundo e cada
            núcleo tem um rótulo próprio. Uma máscara binária é tratada como um único núcleo.

    Returns:
        Array ``float32`` com formato ``(H, W)`` e valores em ``[0, 1]``.

    Raises:
        ValueError: Se ``labels`` não for uma matriz 2D de inteiros.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"Esperada máscara por instância 2D de inteiros, recebido {labels.dtype} {labels.shape}."
        )

    height, width = labels.shape
    dmap = np.ones((height, width), dtype=np.float32)
    for label, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        r0, r1 = max(slices[0].start - 1, 0), min(slices[0].stop + 1, height)
        c0, c1 = max(slices[1].start - 1, 0), min(slices[1].stop + 1, width)
        nucleus = labels[r0:r1, c0:c1] == label
        distance = distance_transform_edt(nucleus)
        peak = distance.max()
        if peak > 0:
            region = dmap[r0:r1, c0:c1]
            region[nucleus] = 1.0 - distance[nucleus] / peak
    return dmap


class DistanceMapStep(PipelineStep):
    """Passo do pipeline que adiciona o mapa de distância por núcleo ao dicionário de dados.

    Lê a máscara por instância em ``instances_key`` (por padrão ``ground_truth_instances``,
    produzida pelo ``MonusegDataset``) e grava o mapa em ``output_key`` (por padrão
    ``distance_map``). Ver :func:`compute_instance_distance_map`.

    Atributos:
        instances_key: Chave do dicionário com a máscara por instância.
        output_key: Chave onde o mapa de distância será adicionado.
        name: Identificador deste passo no pipeline.
    """

    def __init__(
        self,
        instances_key: str = "ground_truth_instances",
        output_key: str = "distance_map",
        name: str = "DistanceMapStep",
    ) -> None:
        """Inicializa DistanceMapStep.

        Args:
            instances_key: Chave do dicionário com a máscara por instância.
            output_key: Chave onde o mapa de distância será adicionado.
            name: Identificador deste passo no pipeline.
        """
        super().__init__(name=name)
        self.instances_key = instances_key
        self.output_key = output_key

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula o mapa de distância e o adiciona ao dicionário de dados.

        Args:
            data: Dicionário contendo:
                - ``instances_key``: máscara por instância ``(H, W)`` de inteiros.

        Returns:
            O mesmo dicionário ``data`` estendido com:
                - ``output_key``: array ``float32`` com formato ``(H, W)``
                  e valores em ``[0, 1]``.

        Raises:
            KeyError: Se ``instances_key`` estiver ausente em ``data``.
        """
        if self.instances_key not in data:
            raise KeyError(
                f"Missing '{self.instances_key}' in data. Ensure it is provided before {self.name}."
            )

        data[self.output_key] = compute_instance_distance_map(data[self.instances_key])

        logger.debug(
            f"[{self.name}] Computed distance map with shape {data[self.output_key].shape}"
        )

        return data
