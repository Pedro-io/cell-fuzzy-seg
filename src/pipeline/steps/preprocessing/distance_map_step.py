

"""Step de pré-processamento que calcula o mapa de distância a partir da máscara.

O mapa de distância é consumido durante o treinamento pela
:class:`~src.losses.distance_map_loss.DistanceMapLoss` (chave ``"distance_map"``
no dicionário de dados, lida pelo ``Trainer``). Antes, esse cálculo vivia dentro
do notebook de treino; agora ele é uma etapa do
:class:`~src.pipeline.preprocessing_pipeline.PreprocessingPipeline`, executada
uma única vez por imagem e persistida como os demais resultados.
"""

from typing import Any, Dict

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.utils.logger import logger

from ..base_step import PipelineStep


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """Calcula o mapa de distância invertido e normalizado a partir de uma máscara.

    Aplica a transformada de distância euclidiana sobre o primeiro plano da máscara
    (``mask > 0``), normaliza pelo valor máximo e inverte o resultado (``1.0 - dt``).
    O mapa resultante tem valores em ``[0, 1]``: próximo de ``0`` no interior das
    células (longe das bordas) e próximo de ``1`` nas fronteiras entre objetos e no
    fundo — exatamente o formato consumido pela
    :class:`~src.losses.distance_map_loss.DistanceMapLoss`.

    Args:
        mask: Máscara inteira com formato ``(H, W)``; valores diferentes de zero
            indicam o primeiro plano.

    Returns:
        Array ``float32`` com formato ``(H, W)`` e valores em ``[0, 1]``.
    """
    dt = distance_transform_edt((mask > 0).astype(np.uint8))
    if dt.max() > 0:
        dt = dt / dt.max()
    return (1.0 - dt).astype(np.float32)


class DistanceMapStep(PipelineStep):
    """Passo do pipeline que adiciona o mapa de distância ao dicionário de dados.

    Calcula o mapa de distância a partir da máscara presente em ``mask_key``
    (por padrão ``ground_truth``) e o armazena em ``output_key`` (por padrão
    ``distance_map``), mantendo a mesma semântica do cálculo que antes era feito
    no notebook de treino.

    Atributos:
        mask_key: Chave do dicionário com a máscara usada como entrada.
        output_key: Chave onde o mapa de distância será adicionado.
        name: Identificador deste passo no pipeline.
    """

    def __init__(
        self,
        mask_key: str = "ground_truth",
        output_key: str = "distance_map",
        name: str = "DistanceMapStep",
    ) -> None:
        """Inicializa DistanceMapStep.

        Args:
            mask_key: Chave do dicionário com a máscara de entrada.
            output_key: Chave onde o mapa de distância será adicionado.
            name: Identificador deste passo no pipeline.
        """
        super().__init__(name=name)
        self.mask_key = mask_key
        self.output_key = output_key

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula o mapa de distância e o adiciona ao dicionário de dados.

        Args:
            data: Dicionário contendo:
                - ``mask_key``: array NumPy com formato ``(H, W)``.

        Returns:
            O mesmo dicionário ``data`` estendido com:
                - ``output_key``: array ``float32`` com formato ``(H, W)``
                  e valores em ``[0, 1]``.

        Raises:
            KeyError: Se ``mask_key`` estiver ausente em ``data``.
        """
        if self.mask_key not in data:
            raise KeyError(
                f"Missing '{self.mask_key}' in data. Ensure it is provided before {self.name}."
            )

        mask = data[self.mask_key]
        data[self.output_key] = compute_distance_map(mask)

        logger.debug(
            f"[{self.name}] Computed distance map with shape {data[self.output_key].shape}"
        )

        return data
