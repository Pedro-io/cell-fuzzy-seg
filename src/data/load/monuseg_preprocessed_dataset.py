import copy
from typing import Any, Dict, List, Optional

from src.data.load.monuseg_dataset import MonusegDataset


class MonusegPreprocessedDataset(MonusegDataset):
    """Wrap a base dataset and enrich each sample with preprocessing outputs.

    The class inherits from MonusegDataset to remain compatible with the
    project's dataset interface while composing a preprocessing pipeline at
    runtime. It does not need to know the specific implementation of the
    wrapped dataset beyond the standard sample contract.
    """

    def __init__(
        self,
        base_dataset: MonusegDataset,
        preprocessing_pipeline: Any,
        transform: Optional[Any] = None,
        yaml_path: Optional[str] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.preprocessing_pipeline = preprocessing_pipeline
        self.transform = transform

        if isinstance(base_dataset, MonusegDataset):
            super().__init__(
                dataset_name=getattr(base_dataset, "dataset_name", "monuseg"),
                config_key=getattr(base_dataset, "config_key", "monuseg_training"),
                transform=transform,
                yaml_path=yaml_path or getattr(base_dataset, "yaml_path", MonusegDataset.YAML_CONFIG_PATH),
            )
            self.image_dir = getattr(base_dataset, "image_dir", self.image_dir)
            self.mask_dir = getattr(base_dataset, "mask_dir", self.mask_dir)
            self.file_pairs = list(getattr(base_dataset, "file_pairs", self.file_pairs))
        else:
            self.dataset_name = getattr(base_dataset, "dataset_name", "monuseg")
            self.config_key = getattr(base_dataset, "config_key", None)
            self.config = getattr(base_dataset, "config", {})
            self.loader_config = getattr(base_dataset, "loader_config", {})
            self.preprocessing = getattr(base_dataset, "preprocessing", {})
            self.root_dir = getattr(base_dataset, "root_dir", "")
            self.yaml_path = yaml_path or getattr(base_dataset, "yaml_path", MonusegDataset.YAML_CONFIG_PATH)
            self.image_dir = getattr(base_dataset, "image_dir", None)
            self.mask_dir = getattr(base_dataset, "mask_dir", None)
            self.file_pairs = list(getattr(base_dataset, "file_pairs", []))

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
        """Execute the preprocessing pipeline using the supported interface."""
        if hasattr(self.preprocessing_pipeline, "run"):
            return self.preprocessing_pipeline.run(data)

        if hasattr(self.preprocessing_pipeline, "forward"):
            return self.preprocessing_pipeline.forward(data)

        if callable(self.preprocessing_pipeline):
            return self.preprocessing_pipeline(data)

        raise TypeError(
            "preprocessing_pipeline must be callable, provide a 'run' method, or provide a 'forward' method"
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
