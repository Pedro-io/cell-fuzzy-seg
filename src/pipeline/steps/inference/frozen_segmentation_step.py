"""Step de inferência que executa a rede final de segmentação sobre os marcadores."""

from typing import Any, Dict, Optional

import torch

from src.models.networks.final_segmentation.base_final_segmentation import BaseFinalSegmentation
from src.utils.logger import logger

from ..base_step import PipelineStep


class FrozenSegmentationStep(PipelineStep):
    """Passo do pipeline que executa a rede final de segmentação.

    Encapsula, dentro do :class:`~src.pipeline.training_pipeline.TrainingPipeline`,
    a execução da rede final de segmentação (uma implementação de
    :class:`BaseFinalSegmentation`). Recebe a imagem e os marcadores produzidos
    pela MarkerNet (via dicionário de dados) e executa o forward da rede final,
    passando um dicionário ``{"image": ..., "scribbles": ...}`` — o contrato
    definido pela interface. Adiciona o resultado (``segmentation``) ao
    dicionário.

    **Não implementa** a arquitetura da rede final — apenas a recebe e a executa.
    **Não realiza** backward nem otimização.

    Diferente do :class:`MarkerStep` (orientado à inferência), este passo mantém
    o grafo computacional diferenciável intacto para permitir a backpropagation
    posterior pelo Trainer. Por isso, a rede final é colocada em modo
    ``train()``: os modos ``train()``/``eval()`` das redes são gerenciados por
    este pipeline de treinamento (e não pelo Trainer), à semelhança de como o
    :class:`MarkerStep` força ``eval()`` no contexto de inferência.

    Atributos:
        final_network: Rede final de segmentação (implementação de
            :class:`BaseFinalSegmentation`).
        device: Dispositivo PyTorch para inferência (cpu ou cuda).
        negative_key: Chave de ``data`` com um canal negativo fixo, repassado à
            rede final como ``"negative"`` (``None``: nenhum).
        name: Identificador deste passo no pipeline.
    """

    def __init__(
        self,
        final_network: Optional[BaseFinalSegmentation] = None,
        device: Optional[str] = None,
        negative_key: Optional[str] = None,
        name: str = "FrozenSegmentationStep",
    ):
        """Inicializa o FrozenSegmentationStep.

        Args:
            final_network: Rede final de segmentação. Se None, a execução
                levanta erro ao tentar gerar a segmentação.
            device: Dispositivo PyTorch. Se None, usa GPU se disponível.
            negative_key: Chave de ``data`` com um canal negativo fixo (por
                exemplo, ``"cellpose_negative"``). A rede final precisa aceitar a
                chave ``"negative"`` (como a ``ScribblePromptingNetwork``).
                ``None``: nenhum negativo é repassado.
            name: Identificador deste passo no pipeline.
        """
        super().__init__(name=name)
        self.final_network = final_network
        self.negative_key = negative_key

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.final_network is not None:
            self.final_network.to(self.device)
            self.final_network.train()

        logger.debug(f"[{self.name}] Initialized on {self.device}")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa a rede final de segmentação sobre imagem e marcadores.

        Args:
            data: Dicionário contendo:
                - ``"image"``: imagem de entrada (numpy ou tensor).
                - ``"markers"``: marcadores nebulosos gerados pela MarkerNet
                  (numpy ou tensor). São repassados à rede final como
                  ``"scribbles"``, o contrato da
                  :class:`BaseFinalSegmentation`.
                - ``negative_key`` (se configurada): canal negativo fixo,
                  repassado como ``"negative"``.

        Returns:
            O mesmo dicionário ``data``, agora estendido com:
                - ``"segmentation"``: tensor com formato ``(N, C, H, W)`` com a
                  segmentação final produzida pela rede.

        Raises:
            KeyError: Se ``"image"``, ``"markers"`` ou ``negative_key`` estiverem ausentes em ``data``.
            RuntimeError: Se ``final_network`` não foi fornecido ou a inferência falhar.
        """
        if "image" not in data:
            raise KeyError("Missing 'image' in data. Ensure the image is provided before FrozenSegmentationStep.")
        if "markers" not in data:
            raise KeyError("Missing 'markers' in data. Ensure MarkerStep runs before FrozenSegmentationStep.")
        if self.negative_key is not None and self.negative_key not in data:
            raise KeyError(f"Missing '{self.negative_key}' in data (negative_key of {self.name}).")

        if self.final_network is None:
            raise RuntimeError(
                f"[{self.name}] No final_network provided. Pass a BaseFinalSegmentation instance."
            )

        try:
            # Contrato da rede final: dict com image + scribbles (marcadores).
            inputs = {"image": data["image"], "scribbles": data["markers"]}
            if self.negative_key is not None:
                inputs["negative"] = data[self.negative_key]
            # Sem torch.no_grad(): o grafo computacional deve permanecer
            # diferenciável para permitir backpropagation durante o treinamento.
            segmentation = self.final_network(inputs)
            data["segmentation"] = segmentation
        except Exception as e:
            logger.error(f"[{self.name}] Inference failed: {e}")
            raise RuntimeError(f"Final segmentation inference failed: {e}") from e

        logger.debug(f"[{self.name}] Generated segmentation with shape {tuple(segmentation.shape)}")
        return data
