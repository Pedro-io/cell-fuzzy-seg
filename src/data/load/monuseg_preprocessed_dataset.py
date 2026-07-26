import copy
from typing import Any, Dict, List, Optional, Protocol

from src.data.load.base_dataset import BaseDataset


class PreprocessingPipelineProtocol(Protocol):
    """Protocol defining the expected interface for preprocessing pipelines.

    A preprocessing pipeline must expose a ``run(data)``, ``forward(data)``,
    or be directly callable with a data dictionary.
    """

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all preprocessing steps on the data dictionary."""
        ...

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all preprocessing steps on the data dictionary."""
        ...


class MonusegPreprocessedDataset(BaseDataset):
    """Wrap a base dataset and enrich each sample with preprocessing outputs.

    Follows the Composition pattern: receives a ``BaseDataset`` and a
    preprocessing pipeline (``PreprocessingPipeline`` or compatible object),
    delegating data loading to the former and preprocessing to the latter.

    The class inherits from ``BaseDataset`` to remain compatible with the
    project's dataset interface, but **does not load its own configuration
    from YAML** — it relies on the wrapped dataset for metadata.

    It does not need to know the specific implementation of the wrapped
    dataset beyond the standard ``BaseDataset`` contract, nor the specific
    steps of the preprocessing pipeline (Cellpose, RGBA, etc.).
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
        """Return the number of samples in the wrapped dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a preprocessed sample from the wrapped dataset."""
        sample = self.base_dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a dict sample, got {type(sample).__name__}")

        processed_sample = copy.deepcopy(sample)
        processed_sample = self._run_pipeline(processed_sample)

        if self.transform is not None:
            processed_sample = self.transform(processed_sample)

        return processed_sample

    def _run_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the preprocessing pipeline using the supported interface.

        Checks, in order, for:
        1. ``.run(data)`` — the canonical ``PreprocessingPipeline`` interface
        2. ``.forward(data)`` — compatible with ``ModelPipeline``
        3. ``__call__(data)`` — compatible with any callable
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
        """Delegate image loading to the wrapped dataset."""
        return self.base_dataset._load_image(image_path)

    def _load_mask(self, mask_path: str, image_shape: Optional[Any] = None) -> Any:
        """Delegate mask loading to the wrapped dataset."""
        return self.base_dataset._load_mask(mask_path, image_shape=image_shape)

    def _get_file_pairs(self) -> List[tuple]:
        """Delegate file pair resolution to the wrapped dataset."""
        return self.base_dataset._get_file_pairs()
