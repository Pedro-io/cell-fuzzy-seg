import copy
from typing import Any, Dict, List, Optional, Protocol

from src.data.load.base_dataset import BaseDataset


class PreprocessingPipelineProtocol(Protocol):
    """Protocolo que define a interface esperada para pipelines de pré-processamento.

    Um pipeline de pré-processamento deve expor ``run(data)``, ``forward(data)``
    ou ser diretamente chamável com um dicionário de dados.
    """

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa todos os passos de pré-processamento no dicionário de dados."""
        ...

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa todos os passos de pré-processamento no dicionário de dados."""
        ...


class MonusegPreprocessedDataset(BaseDataset):
    """Envolve um dataset base e enriquece cada amostra com os resultados do pré-processamento.

    Segue o padrão de composição: recebe um ``BaseDataset`` e um pipeline de
    pré-processamento (``PreprocessingPipeline`` ou objeto compatível),
    delegando o carregamento de dados ao primeiro e o pré-processamento ao
    segundo.

    A classe herda de ``BaseDataset`` para permanecer compatível com a
    interface de datasets do projeto, mas **não carrega sua própria
    configuração a partir do YAML** — ela depende do dataset encapsulado para
    obter metadados.

    Não precisa conhecer a implementação específica do dataset encapsulado
    além do contrato padrão de ``BaseDataset``, nem os passos específicos do
    pipeline de pré-processamento (Cellpose, RGBA, etc.).
    """

    def __init__(
        self,
        base_dataset: BaseDataset,
        preprocessing_pipeline: PreprocessingPipelineProtocol,
        transform: Optional[Any] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.preprocessing_pipeline = preprocessing_pipeline
        self.transform = transform

        # Copy metadata from the wrapped dataset to satisfy the BaseDataset
        # interface without loading its own YAML config or validating disk
        # paths. The wrapper delegates all file I/O to ``base_dataset``.
        self.dataset_name = getattr(base_dataset, "dataset_name", "wrapper")
        self.config_key = getattr(base_dataset, "config_key", None)
        self.config = getattr(base_dataset, "config", {})
        self.loader_config = getattr(base_dataset, "loader_config", {})
        self.preprocessing = getattr(base_dataset, "preprocessing", {})
        self.root_dir = getattr(base_dataset, "root_dir", "")
        self.yaml_path = getattr(base_dataset, "yaml_path", "")

    def __len__(self) -> int:
        """Retorna o número de amostras no dataset encapsulado."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retorna uma amostra pré-processada a partir do dataset encapsulado."""
        sample = self.base_dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a dict sample, got {type(sample).__name__}")

        processed_sample = copy.deepcopy(sample)
        processed_sample = self._run_pipeline(processed_sample)

        if self.transform is not None:
            processed_sample = self.transform(processed_sample)

        return processed_sample

    def _run_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa o pipeline de pré-processamento usando a interface suportada.

        Verifica, em ordem, os seguintes casos:
        1. ``.run(data)`` — a interface canônica de ``PreprocessingPipeline``
        2. ``.forward(data)`` — compatível com ``ModelPipeline``
        3. ``__call__(data)`` — compatível com qualquer objeto chamável
        """
        if hasattr(self.preprocessing_pipeline, "run"):
            return self.preprocessing_pipeline.run(data)

        if hasattr(self.preprocessing_pipeline, "forward"):
            return self.preprocessing_pipeline.forward(data)

        if callable(self.preprocessing_pipeline):
            return self.preprocessing_pipeline(data)

        raise TypeError(
            "preprocessing_pipeline must be callable, provide a 'run' method, "
            "or provide a 'forward' method"
        )

    def _load_image(self, image_path: str) -> Any:
        """Delegue o carregamento de imagem ao dataset encapsulado."""
        return self.base_dataset._load_image(image_path)

    def _load_mask(self, mask_path: str, image_shape: Optional[Any] = None) -> Any:
        """Delegue o carregamento de máscara ao dataset encapsulado."""
        return self.base_dataset._load_mask(mask_path, image_shape=image_shape)

    def _get_file_pairs(self) -> List[tuple]:
        """Delegue a resolução dos pares de arquivos ao dataset encapsulado."""
        return self.base_dataset._get_file_pairs()
