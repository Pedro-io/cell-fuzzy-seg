"""Composição baseada em estratégia de instâncias de LossTerm."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .loss_term import LossTerm


class LossComposer(nn.Module):
    """Compõe uma lista arbitrária de instâncias de :class:`LossTerm` em uma única perda.

    Os termos são registrados como submódulos do PyTorch via ``nn.ModuleList``,
    então seus parâmetros e buffers são rastreados corretamente. O conjunto
    ativo é determinado inteiramente no momento da construção — basta trocar a
    lista para executar um experimento diferente sem alterar o código-fonte.

    Os nomes dos termos devem ser únicos dentro do compositor; nomes duplicados
    sobrescrevem a entrada anterior no log.

    Exemplo::

        composer = LossComposer([SizeTerm(0.1), TVTerm(0.05)])
        total, log = composer(markers, distance_maps, gt_masks)
        # log = {"size": tensor, "tv": tensor}
    """

    def __init__(self, terms: List[LossTerm]) -> None:
        """Inicializa LossComposer.

        Args:
            terms: Lista de instâncias de :class:`LossTerm` para compor.
        """
        super().__init__()
        self._terms = nn.ModuleList(terms)

    @property
    def terms(self) -> List[LossTerm]:
        return list(self._terms)

    def forward(
        self,
        markers: torch.Tensor,
        distance_maps: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calcula a perda total e um log por termo.

        Args:
            markers: Tensor de marcadores previstos com formato ``(N, C, H, W)``.
            distance_maps: Tensor do mapa de distância com formato ``(N, C, H, W)``.
            gt_masks: Tensor da máscara de ground truth com formato ``(N, C, H, W)``.

        Returns:
            Uma tupla ``(total_loss, loss_log)`` em que ``loss_log`` associa o
            :attr:`~LossTerm.name` de cada termo ao seu tensor escalar individual.
        """
        ctx: Dict[str, torch.Tensor] = {
            "markers": markers,
            "distance_maps": distance_maps,
            "gt_masks": gt_masks,
        }
        log: Dict[str, torch.Tensor] = {}
        active = []

        for term in self._terms:
            val = term(ctx)
            log[term.name] = val
            active.append(val)

        total = sum(active) if active else torch.tensor(0.0)
        return total, log
