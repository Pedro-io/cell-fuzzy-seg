"""Composição baseada em estratégia de instâncias de LossTerm."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .loss_term import LossTerm


class LossComposer(nn.Module):
    """Compõe uma lista arbitrária de instâncias de :class:`LossTerm` em uma única perda.

    Os termos são registrados como submódulos do PyTorch via ``nn.ModuleList``,
    então seus parâmetros e buffers são rastreados corretamente. O conjunto
    ativo é determinado inteiramente no momento da construção — basta trocar a
    lista para executar um experimento diferente sem alterar o código-fonte.

    O contexto compartilhado entregue aos termos contém **dois tensores de
    predição distintos** (investigação, C3/C4):

    - ``ctx[\"prediction\"]``: a predição supervisionada pelo ``Trainer`` — por
      padrão a **segmentação final** produzida pela rede congelada
      (``prediction_key=\"segmentation\"``). É o alvo dos termos que medem a
      qualidade da segmentação (Dice/RMSE/Size).
    - ``ctx[\"markers\"]``: os **marcadores produzidos pela MarkerNet**
      (``data[\"markers\"]``), quando disponíveis. É o alvo da supervisão direta
      da MarkerNet (ex.: ``DMapTerm``). Se a MarkerNet não estiver presente no
      pipeline, assume o valor de ``prediction`` para manter a compatibilidade.

    Os nomes dos termos devem ser únicos dentro do compositor; nomes duplicados
    sobrescrevem a entrada anterior no log.

    Exemplo::

        composer = LossComposer([DiceTerm(), SizeTerm(0.1), DMapTerm(0.1)])
        total, log = composer(segmentation, distance_maps, gt_masks, markers=markers)
        # log = {\"dice\": tensor, \"size\": tensor, \"dmap\": tensor}
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
        prediction: torch.Tensor,
        distance_maps: torch.Tensor,
        gt_masks: torch.Tensor,
        markers: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calcula a perda total e um log por termo.

        Args:
            prediction: Predição principal supervisionada (por padrão a
                segmentação final) com formato ``(N, C, H, W)``.
            distance_maps: Tensor do mapa de distância com formato ``(N, C, H, W)``.
            gt_masks: Tensor da máscara de ground truth com formato ``(N, C, H, W)``.
            markers: Marcadores produzidos pela MarkerNet com formato
                ``(N, C, H, W)``, usados pelos termos de supervisão direta
                (ex.: ``DMapTerm``). Se ``None``, ``ctx[\"markers\"]`` assume o
                valor de ``prediction``.

        Returns:
            Uma tupla ``(total_loss, loss_log)`` em que ``loss_log`` associa o
            :attr:`~LossTerm.name` de cada termo ao seu tensor escalar individual.
        """
        ctx: Dict[str, torch.Tensor] = {
            "prediction": prediction,
            "markers": markers if markers is not None else prediction,
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
