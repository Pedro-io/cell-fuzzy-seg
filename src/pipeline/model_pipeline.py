from typing import Any, Dict, List

from src.utils.logger import logger

from .steps.base_step import PipelineStep


class ModelPipeline:
    """Orquestra a execução de uma sequência de passos do pipeline."""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def forward(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Executa todos os passos sequencialmente, passando ``data`` por cada um.

        A estrutura mínima necessária de ``data`` na entrada do pipeline depende
        dos passos configurados. O pipeline padrão espera:

        - ``"image"`` (``np.ndarray``): imagem de entrada com formato ``(H, W)`` ou
          ``(H, W, C)``. Necessária para ``CellposeStep``, ``MarkerStep`` e
          ``SegmentationStep``.

        Cada passo pode adicionar novas chaves das quais os passos seguintes dependem:

        - ``CellposeStep`` adiciona ``"segmentation"``, ``"flows"`` e ``"styles"``.
        - ``MarkerStep`` adiciona ``"markers"`` (quando implementado).
        - ``SegmentationStep`` atualiza ``"segmentation"`` (quando implementado).
        - ``RGBAStep`` adiciona ``"rgba"``.

        Args:
            data: Dicionário de dados de entrada. Deve ser um ``dict``; caso contrário,
                levanta ``TypeError``.
            verbose: Registra o nome de cada passo antes de executá-lo.

        Returns:
            Dicionário de dados atualizado após a aplicação de todos os passos.

        Raises:
            TypeError: Se ``data`` não for um ``dict``.
            Exception: Repassa qualquer exceção lançada por um passo, após registrar
                qual passo falhou.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Pipeline input must be a dict, got {type(data).__name__}.")

        for step in self.steps:
            if verbose:
                logger.info(f"[Pipeline] Running: {step.name}")
            try:
                data = step(data)
            except Exception as e:
                logger.error(f"[Pipeline] Step '{step.name}' failed: {e}")
                raise

        return data
