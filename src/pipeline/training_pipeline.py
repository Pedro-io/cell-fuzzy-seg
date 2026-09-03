"""Pipeline de treinamento: encadeia o forward das redes treináveis durante o treinamento.

Este módulo fornece :class:`TrainingPipeline`, o orquestrador canônico para os
passos de inferência (por exemplo, :class:`MarkerStep` e
:class:`FrozenSegmentationStep`) que executam o *forward* das redes treináveis
(MarkerNet e rede final de segmentação). O pipeline **não treina**: não calcula
loss, não executa ``backward()`` e não conhece o otimizador — sua única
responsabilidade é produzir a segmentação final mantendo o grafo computacional
diferenciável intacto para permitir a backpropagation posterior pelo Trainer.
"""

from typing import Any, Dict, List

from src.pipeline.steps.base_step import PipelineStep
from src.utils.logger import logger


class TrainingPipeline:
    """Orquestra uma sequência de passos de inferência treináveis.

    Cada passo é executado em ordem, passando o mesmo dicionário de dados pela
    cadeia. Os passos devem **adicionar** chaves ao dicionário, nunca removê-las.

    O ponto de entrada canônico é :meth:`run`, que corresponde à interface
    esperada por quem consome o pipeline durante o treinamento.

    Uso típico::

        pipeline = TrainingPipeline([
            MarkerStep(marker_net),
            FrozenSegmentationStep(final_network),
        ])
        data = pipeline.run(data)
        # data agora contém data["segmentation"]

    Args:
        steps: Lista ordenada de instâncias de :class:`PipelineStep` de inferência.

    Raises:
        TypeError: Se ``data`` passado para :meth:`run` não for um ``dict``.
    """

    def __init__(self, steps: List[PipelineStep]) -> None:
        self.steps = steps

    def run(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Executa todos os passos sequencialmente no dicionário de dados.

        Cada passo recebe o dicionário atual e retorna uma versão atualizada.
        A saída do passo *i* torna-se a entrada do passo *i+1*.

        O pipeline é sempre executado com gradientes habilitados durante o
        treinamento, pois as redes envolvidas (MarkerNet e rede final) são
        treináveis. Nenhum passo desta cadeia deve desabilitar gradientes de
        forma permanente.

        Args:
            data: Dicionário de dados de entrada. Deve ser um ``dict``.
            verbose: Se ``True``, registra o nome de cada passo antes da execução.

        Returns:
            Dicionário de dados atualizado após a aplicação de todos os passos.

        Raises:
            TypeError: Se ``data`` não for um ``dict``.
            Exception: Repassa qualquer exceção lançada por um passo, após
                registrar o nome do passo que falhou.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Pipeline input must be a dict, got {type(data).__name__}.")

        for step in self.steps:
            if verbose:
                logger.info(f"[TrainingPipeline] Running: {step.name}")
            try:
                data = step(data)
            except Exception as e:
                logger.error(f"[TrainingPipeline] Step '{step.name}' failed: {e}")
                raise

        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias conveniente para :meth:`run` (sem verbosidade).

        Permite passar o pipeline como objeto chamável, mantendo a mesma
        interface dos demais pipelines do projeto.
        """
        return self.run(data, verbose=False)
