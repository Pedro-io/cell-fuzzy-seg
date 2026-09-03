"""Classe base abstrata para termos de perda composáveis."""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class LossTerm(nn.Module, ABC):
    """Interface para um único termo de perda composável.

    Cada termo concreto envolve uma função de perda e expõe uma assinatura
    uniforme ``compute(ctx)``. O dicionário compartilhado de contexto carrega
    todos os tensores que um termo pode precisar; cada termo usa apenas o que
    necessita.

    Chaves esperadas no contexto:
        - ``"prediction"``: tensor da predição principal supervisionada
          ``(N, C, H, W)`` — por padrão a segmentação final produzida pela
          rede congelada (``Trainer.prediction_key``).
        - ``"markers"``: tensor dos marcadores produzidos pela MarkerNet
          ``(N, C, H, W)``. Quando a MarkerNet não está no pipeline, assume o
          valor de ``"prediction"``.
        - ``"distance_maps"``: tensor do mapa de distância ``(N, C, H, W)``.
        - ``"gt_masks"``: tensor da máscara de ground truth ``(N, C, H, W)``.

    As subclasses devem implementar :attr:`name` e :meth:`compute`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Chave única usada no log de perdas retornado por :class:`LossComposer`."""
        ...

    @abstractmethod
    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calcula a perda a partir do contexto compartilhado.

        Args:
            ctx: Contexto compartilhado com tensores. As chaves obrigatórias
                dependem do termo concreto — veja cada subclasse para detalhes.

        Returns:
            Tensor escalar da perda.
        """
        ...

    def forward(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.compute(ctx)
