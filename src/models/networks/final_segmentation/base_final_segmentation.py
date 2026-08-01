"""Interface comum para as redes finais de segmentação.

Este módulo implementa o **Strategy Pattern** para o ponto de variação explícito
do projeto: a rede final de segmentação (FMBS, Scribble Prompting ou outra).
O :class:`FrozenSegmentationStep` depende apenas desta interface, sem conhecer
qual implementação concreta está sendo utilizada.
"""

from abc import abstractmethod
from typing import Any, Dict

from ..base_networks import BaseNetwork


class BaseFinalSegmentation(BaseNetwork):
    """Contrato comum que qualquer rede final de segmentação deve implementar.

    Toda rede final de segmentação do projeto deve herdar desta classe,
    permitindo que o :class:`~src.pipeline.steps.inference.frozen_segmentation_step.FrozenSegmentationStep`
    a execute de forma intercambiável. Adicionar uma nova rede final exige
    apenas criar uma nova subclasse — nenhuma outra camada do sistema precisa
    ser alterada.

    A rede final recebe um dicionário com a imagem de entrada e os marcadores
    nebulosos (scribbles) produzidos pela MarkerNet, e produz a segmentação
    final usada no cálculo da loss.

    Exemplo::

        class MyNewFinalNetwork(BaseFinalSegmentation):
            def forward(self, data):
                image = data["image"]
                scribbles = data["scribbles"]
                ...
    """

    @abstractmethod
    def forward(self, data: Dict[str, Any]):
        """Executa o forward da rede final sobre um dicionário de entradas.

        Args:
            data: Dicionário contendo pelo menos as chaves:
                - ``"image"``: imagem de entrada.
                - ``"scribbles"``: marcadores nebulosos (scribbles) produzidos
                  pela MarkerNet.

        Returns:
            Segmentação final produzida pela rede (tensor com formato
            ``(N, C, H, W)``).
        """
        pass
