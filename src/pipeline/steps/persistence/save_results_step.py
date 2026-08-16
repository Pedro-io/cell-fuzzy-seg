"""Step de persistência que grava os resultados do pré-processamento em disco.

Este passo recebe o dicionário de dados já enriquecido pelos Steps anteriores
(ex. Cellpose, RGBA, DistanceMap) e delega a gravação em disco ao
:class:`~src.io.output_writer.OutputWriter`. Ele é o ponto de entrada canônico
para persistir o pré-processamento **uma única vez** por imagem, permitindo que
os notebooks de treino (ex. ``experiment_3.ipynb``) carreguem os resultados já
processados sem recomputar o Cellpose a cada experimento.

Contrato de persistência:

- cada chave persistida vira uma pasta ``<output_dir>/<chave>/`` e cada amostra
  é salva como ``<output_dir>/<chave>/<id>.npy`` (formato NumPy, que preserva a
  precisão float32 de ``rgba`` e ``distance_map``);
- os nomes das chaves persistidas são **idênticos** às chaves usadas em memória
  (regra 5 do contrato de dados da arquitetura), para que o carregamento
  reconstrua o dicionário completo;
- o passo exige a chave ``"id"`` no dicionário para nomear os arquivos.

O passo **nunca** participa do ``TrainingPipeline`` — pertence exclusivamente ao
:class:`~src.pipeline.preprocessing_pipeline.PreprocessingPipeline` (regra 10 da
arquitetura).
"""

from typing import Any, Dict, List, Optional

from src.io.output_writer import OutputWriter
from src.utils.logger import logger

from ..base_step import PipelineStep


class SaveResultsStep(PipelineStep):
    """Persiste as chaves do dicionário de dados em disco via ``OutputWriter``.

    Atributos:
        output_dir: Diretório base onde as chaves serão persistidas
            (ex.: ``data_source/MoNuSegPreprocessed/train``).
        keys: Lista de chaves a persistir. Padrão: as chaves produzidas pelo
            pré-processamento (``image``, ``segmentation``, ``rgba``,
            ``ground_truth`` e ``distance_map``).
        writer: Instância de :class:`~src.io.output_writer.OutputWriter` que
            executa a gravação em disco.
        name: Identificador deste passo no pipeline.
    """

    DEFAULT_KEYS = ["image", "segmentation", "rgba", "ground_truth", "distance_map"]

    def __init__(
        self,
        output_dir: str,
        keys: Optional[List[str]] = None,
        name: str = "SaveResultsStep",
    ) -> None:
        """Inicializa SaveResultsStep.

        Args:
            output_dir: Diretório base onde os resultados serão persistidos.
            keys: Chaves do dicionário a persistir. Se ``None``, usa
                :attr:`DEFAULT_KEYS` (as chaves produzidas pelos Steps de
                pré-processamento do projeto).
            name: Identificador deste passo no pipeline.
        """
        super().__init__(name=name)
        self.output_dir = output_dir
        self.keys = list(keys) if keys is not None else list(self.DEFAULT_KEYS)
        self.writer = OutputWriter(output_dir)
        logger.info(f"[{self.name}] Persistindo em: {output_dir}")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Persiste as chaves configuradas do dicionário e devolve o dicionário intacto.

        Args:
            data: Dicionário contendo pelo menos ``"id"`` e as chaves listadas
                em :attr:`keys`.

        Returns:
            O mesmo dicionário ``data`` (o passo não altera o conteúdo em memória).

        Raises:
            KeyError: Se ``"id"`` estiver ausente ou se alguma chave de
                :attr:`keys` não estiver presente em ``data``.
        """
        image_id = data.get("id")
        if image_id is None:
            raise KeyError(
                f"[{self.name}] Missing 'id' in data — necessário para nomear os arquivos persistidos."
            )

        self.writer.save_preprocessed(image_id, data, self.keys)
        logger.debug(f"[{self.name}] Sample '{image_id}' persistida em {self.output_dir}")
        return data
