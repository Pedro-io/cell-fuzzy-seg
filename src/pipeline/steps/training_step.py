"""Pipeline step that computes the training loss via a LossComposer."""

from typing import Any, Dict

from .base_step import PipelineStep
from src.losses.loss_composer import LossComposer


class TrainingStep(PipelineStep):
    """Computes the composed training loss and stores it in the data dict.

    Consumes ``"markers"``, ``"distance_maps"``, and ``"gt_masks"`` from the
    data dict and delegates to the injected :class:`~src.losses.LossComposer`.
    The caller (training loop) is responsible for calling ``loss.backward()``
    and stepping the optimizer.

    Example::

        step = TrainingStep(LossComposer([SizeTerm(0.1), TVTerm(0.05)]))
        data = step(data)
        data["loss"].backward()

    Attributes:
        composer: The :class:`~src.losses.LossComposer` used to compute losses.
    """

    def __init__(self, composer: LossComposer, name: str = "TrainingStep") -> None:
        """Initializes TrainingStep.

        Args:
            composer: Composed loss to apply.
            name: Identifier for this step in the pipeline.
        """
        super().__init__(name=name)
        self.composer = composer

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the composer and stores the result in ``data``.

        Args:
            data: Dictionary containing at least:
                - ``"markers"``: predicted marker tensor ``(N, C, H, W)``.
                - ``"distance_maps"``: distance map tensor ``(N, C, H, W)``.
                - ``"gt_masks"``: ground truth mask tensor ``(N, C, H, W)``.

        Returns:
            The same ``data`` dictionary extended with:
                - ``"loss"``: total scalar loss tensor.
                - ``"loss_log"``: per-term dict mapping term name to its scalar.

        Raises:
            KeyError: If any required key is absent from ``data``.
        """
        loss, log = self.composer(
            data["markers"],
            data["distance_maps"],
            data["gt_masks"],
        )
        data["loss"] = loss
        data["loss_log"] = log
        return data
