"""Fechamento morfológico em tons de cinza dos marcadores, antes da rede final.

O fechamento (dilatação seguida de erosão pelo mesmo elemento estruturante) tapa buracos e frestas
menores que o elemento estruturante sem diminuir o marcador. Sobre um marcador difuso (valores em
``[0, 1]``), a dilatação é o máximo na vizinhança e a erosão, o mínimo (morfologia em tons de cinza
com elemento estruturante plano; Haralick, Sternberg e Zhuang, IEEE TPAMI, 1987). Operadores planos
comutam com a limiarização (Maragos e Schafer, IEEE TASSP, 1987), então fechar o marcador difuso e
depois cortá-lo em τ dá o mesmo que cortar e depois fechar.

O máximo e o mínimo deixam o gradiente passar (para o pixel que vence em cada vizinhança), então o
passo pode ficar dentro do treino.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.utils.logger import logger

from ..base_step import PipelineStep


def disk_offsets(radius: int) -> List[Tuple[int, int]]:
    """Deslocamentos ``(dy, dx)`` do disco de raio ``radius`` px (``dy² + dx² <= radius²``)."""
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]


def _flat_filter(x: torch.Tensor, radius: int, dilate: bool) -> torch.Tensor:
    """Máximo (``dilate``) ou mínimo na vizinhança em disco; fora da imagem não conta."""
    pad_value = float("-inf") if dilate else float("inf")
    padded = F.pad(x, (radius, radius, radius, radius), mode="constant", value=pad_value)
    h, w = x.shape[-2:]
    combine = torch.maximum if dilate else torch.minimum
    out = None
    for dy, dx in disk_offsets(radius):
        view = padded[..., radius + dy:radius + dy + h, radius + dx:radius + dx + w]
        out = view if out is None else combine(out, view)
    return out


def grey_dilation(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilatação em tons de cinza por um disco de raio ``radius`` px."""
    return _flat_filter(x, radius, dilate=True)


def grey_erosion(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Erosão em tons de cinza por um disco de raio ``radius`` px."""
    return _flat_filter(x, radius, dilate=False)


def grey_closing(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Fechamento em tons de cinza por um disco de raio ``radius`` px.

    Args:
        x: Tensor ``(..., H, W)`` (por exemplo, marcadores ``(B, 1, H, W)``).
        radius: Raio do disco, em px. ``0``: devolve ``x`` sem mudança.

    Returns:
        Tensor do mesmo formato, ``>= x`` em todo pixel (o fechamento é extensivo). Pixels fora da
        imagem não entram na dilatação nem na erosão, então a borda não cria nem apaga marcador.

    Raises:
        ValueError: Se ``radius`` for negativo ou ``x`` tiver menos de 2 dimensões.
    """
    if radius < 0:
        raise ValueError(f"radius deve ser >= 0, recebido {radius}.")
    if x.ndim < 2:
        raise ValueError(f"x deve ter ao menos 2 dimensões (H, W), recebido {tuple(x.shape)}.")
    if radius == 0:
        return x
    return grey_erosion(grey_dilation(x, radius), radius)


class MarkerClosingStep(PipelineStep):
    """Passo do pipeline que fecha os marcadores antes da rede final.

    Lê ``input_key`` e grava o marcador fechado em ``output_key``, sem alterar a entrada: o marcador
    cru da MarkerUNet continua disponível, e a rede final passa a ler ``output_key``
    (``FrozenSegmentationStep(markers_key=...)``).

    Atributos:
        radius: Raio do disco do fechamento, em px da imagem.
        input_key: Chave dos marcadores de entrada.
        output_key: Chave dos marcadores fechados.
        name: Identificador deste passo no pipeline.
    """

    def __init__(
        self,
        radius: int,
        input_key: str = "markers",
        output_key: str = "markers_closed",
        name: str = "MarkerClosingStep",
    ):
        """Inicializa o MarkerClosingStep.

        Args:
            radius: Raio do disco do fechamento, em px (``0``: cópia sem mudança).
            input_key: Chave dos marcadores de entrada, um tensor ``(B, C, H, W)``.
            output_key: Chave em que o marcador fechado é gravado.
            name: Identificador deste passo no pipeline.

        Raises:
            ValueError: Se ``radius`` for negativo ou ``output_key`` for igual a ``input_key``.
        """
        super().__init__(name=name)
        if radius < 0:
            raise ValueError(f"radius deve ser >= 0, recebido {radius}.")
        if output_key == input_key:
            raise ValueError("output_key deve ser diferente de input_key (o passo só acrescenta chaves).")
        self.radius = radius
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Grava em ``output_key`` o fechamento de ``data[input_key]``.

        Raises:
            KeyError: Se ``input_key`` estiver ausente.
            TypeError: Se os marcadores não forem um tensor.
        """
        if self.input_key not in data:
            raise KeyError(f"Missing '{self.input_key}' in data. Ensure MarkerStep runs before {self.name}.")
        markers = data[self.input_key]
        if not isinstance(markers, torch.Tensor):
            raise TypeError(f"'{self.input_key}' deve ser um torch.Tensor, recebido {type(markers).__name__}.")
        data[self.output_key] = grey_closing(markers, self.radius)
        logger.debug(f"[{self.name}] Closed '{self.input_key}' with radius {self.radius}")
        return data
