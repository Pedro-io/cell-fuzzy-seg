"""Pipeline de pré-processamento: executa uma sequência de passos não treináveis uma vez por amostra.

Este módulo fornece :class:`PreprocessingPipeline`, o orquestrador canônico
para passos de pré-processamento (por exemplo, Cellpose e conversão para RGBA)
que são executados **uma vez** por imagem, fora do loop de treino. Os resultados
são persistidos por um :class:`SaveResultsStep` e lidos depois do disco durante o
treino (:mod:`src.data.load.preprocessed_dataset`), evitando recomputar passos
caros (como o Cellpose) a cada época.
"""

from typing import Any, Dict, List

from src.pipeline.steps.base_step import PipelineStep
from src.utils.logger import logger


class PreprocessingPipeline:
    """Orquestra uma sequência de passos de pré-processamento não treináveis.

    Cada passo é executado em ordem, passando o mesmo dicionário de dados pela cadeia.
    Os passos devem **adicionar** chaves ao dicionário, nunca removê-las.

    O ponto de entrada canônico é :meth:`run`.

    Uso típico::

        pipeline = PreprocessingPipeline([
            CellposeStep(),
            RGBAStep(),
            SaveResultsStep(output_dir="data/preprocessed"),
        ])
        data = pipeline.run({"image": image, "ground_truth": mask})

    Args:
        steps: Lista ordenada de instâncias de :class:`PipelineStep` de pré-processamento.

    Raises:
        TypeError: Se ``data`` passado para :meth:`run` não for um ``dict``.
    """

    def __init__(self, steps: List[PipelineStep]) -> None:
        self.steps = steps

    def run(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Executa todos os passos sequencialmente no dicionário de dados.

        Cada passo recebe o dicionário atual e retorna uma versão atualizada.
        A saída do passo *i* torna-se a entrada do passo *i+1*.

        Args:
            data: Dicionário de dados de entrada. Deve ser um ``dict``.
            verbose: Se ``True``, registra o nome de cada passo antes da execução.

        Returns:
            Dicionário de dados atualizado após a aplicação de todos os passos.

        Raises:
            TypeError: Se ``data`` não for um ``dict``.
            Exception: Repassa qualquer exceção lançada por um passo, após registrar
                o nome do passo que falhou.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Pipeline input must be a dict, got {type(data).__name__}.")

        for step in self.steps:
            if verbose:
                logger.info(f"[PreprocessingPipeline] Running: {step.name}")
            try:
                data = step(data)
            except Exception as e:
                logger.error(f"[PreprocessingPipeline] Step '{step.name}' failed: {e}")
                raise

        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias conveniente para :meth:`run` (sem verbosidade).

        Permite usar o pipeline como objeto chamável, sem exigir ``.run()`` explicitamente.
        """
        return self.run(data, verbose=False)
