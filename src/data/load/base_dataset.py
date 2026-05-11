import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import yaml
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for datasets.

    Defines the standard interface for all datasets in the project, allowing
    different data sources (MoNuSeg, custom, etc.) to be interchangeable while
    maintaining the same interface. Loads configurations from a centralized YAML
    file, enabling easy dataset switching.

    Attributes:
        dataset_name (str): Name of the dataset in YAML (e.g., 'monuseg').
        config_key (str): Specific configuration key within the dataset
            (e.g., 'monuseg_training').
        config (dict): Specific configuration loaded from YAML.
        loader_config (dict): DataLoader configuration.
        preprocessing (dict): Preprocessing configuration.
        root_dir (str): Root directory of the dataset.
        transform: Transformations to apply to data.
    """

    YAML_CONFIG_PATH = 'configs/datasets.yml'

    def __init__(
      self,
      dataset_name: str,
      config_key: str,
      transform: Optional[Any] = None,
      yaml_path: str = YAML_CONFIG_PATH
      ) -> None:
        """Initialize the dataset by loading configuration from YAML.

        Args:
            dataset_name (str): Name of the dataset in YAML (e.g., 'monuseg').
            config_key (str): Specific configuration key within the dataset
                (e.g., 'monuseg_training' for training, 'monuseg_test' for testing).
            transform: Transformations to apply. Default: None.
            yaml_path (str): Path to the YAML configuration file.
                Default: 'configs/datasets.yml'.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If dataset_name or config_key do not exist in YAML.
            ValueError: If root_dir does not exist or config is incomplete.
        """
        self.dataset_name = dataset_name
        self.config_key = config_key
        self.transform = transform
        self.yaml_path = yaml_path

        # Load configuration from YAML
        self._load_config_from_yaml()

        # Validate and extract information
        self._validate_config()
        self.root_dir = self.config.get('root_dir')

        if not self.root_dir or not os.path.exists(self.root_dir):
            raise ValueError(f"Invalid or non-existent root_dir: {self.root_dir}")

    def _load_config_from_yaml(self) -> None:
        """Load configurations from YAML file.

        Raises:
            FileNotFoundError: If YAML file does not exist.
            KeyError: If dataset_name or config_key are not in YAML.
        """
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")

        with open(self.yaml_path, encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)

        if self.dataset_name not in all_configs:
            raise KeyError(f"Dataset '{self.dataset_name}' not found in {self.yaml_path}")

        dataset_config = all_configs[self.dataset_name]

        if self.config_key not in dataset_config:
            raise KeyError(f"Config '{self.config_key}' not found in {self.yaml_path}[{self.dataset_name}]")

        # Extract specific configuration
        self.config = dataset_config[self.config_key]

        # Extract global dataset configurations (loader_config, preprocessing)
        self.loader_config = dataset_config.get('loader_config', {})
        self.preprocessing = dataset_config.get('preprocessing', {})

    def _validate_config(self) -> None:
        """Validate if configuration has all required fields.

        Raises:
            KeyError: If any required field is missing.
        """
        required_keys = ['root_dir', 'image_dir', 'mask_dir', 'image_extension',
                         'mask_extension']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Required config field missing for '{self.config_key}': {key}")

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing:
                - 'image': Loaded image.
                - 'mask': Corresponding mask.
                - 'image_name': Image filename (optional).
        """
        pass

    @abstractmethod
    def _load_image(self, image_path: str) -> Any:
        """Load an image from file.

        Args:
            image_path (str): Path to the image.

        Returns:
            Loaded image (dataset-specific format).
        """
        pass

    @abstractmethod
    def _load_mask(self, mask_path: str) -> Any:
        """Load a mask from file.

        Args:
            mask_path (str): Path to the mask.

        Returns:
            Loaded mask (dataset-specific format).
        """
        pass

    @abstractmethod
    def _get_file_pairs(self) -> list:
        """Return list of (image, mask) pairs.

        Returns:
            list: List of tuples (image_path, mask_path).
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return the specific configuration loaded.

        Returns:
            dict: Dataset configuration (e.g., monuseg_training).
        """
        return self.config

    def get_loader_config(self) -> Dict[str, Any]:
        """Return the DataLoader configuration (batch_size, num_workers, etc).

        Returns:
            dict: Loader configuration.
        """
        return self.loader_config

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return the preprocessing configuration (image_size, normalize, etc).

        Returns:
            dict: Preprocessing configuration.
        """
        return self.preprocessing

    def get_dataset_name(self) -> str:
        """Return the name of the dataset.

        Returns:
            str: Dataset name (e.g., 'monuseg').
        """
        return self.dataset_name

    def get_config_key(self) -> str:
        """Return the specific configuration key.

        Returns:
            str: Config key (e.g., 'monuseg_training').
        """
        return self.config_key

    def set_transform(self, transform: Any) -> None:
        """Set the transformations to be applied.

        Args:
            transform: Transformation object.
        """
        self.transform = transform

    def get_image_dir(self) -> str:
        """Return the full path to the images directory.

        Returns:
            str: Path to the images directory.
        """
        return os.path.join(self.root_dir, self.config['image_dir'])

    def get_mask_dir(self) -> str:
        """Return the full path to the masks directory.

        Returns:
            str: Path to the masks directory.
        """
        return os.path.join(self.root_dir, self.config['mask_dir'])

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics.

        Returns:
            dict: Dictionary with statistics (size, configuration, paths, etc).
        """
        return {
            'dataset_name': self.dataset_name,
            'config_key': self.config_key,
            'size': len(self),
            'config': self.config,
            'loader_config': self.loader_config,
            'preprocessing': self.preprocessing,
            'image_dir': self.get_image_dir(),
            'mask_dir': self.get_mask_dir(),
        }
