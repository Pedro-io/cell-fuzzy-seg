import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import yaml
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Classe base abstrata para datasets.

    Define a interface padrão para todos os datasets do projeto, permitindo
    que diferentes fontes de dados (MoNuSeg, personalizadas, etc.) sejam
    intercambiáveis, mantendo a mesma interface. Carrega configurações a partir
    de um arquivo YAML centralizado, facilitando a troca de datasets.

    Atributos:
        dataset_name (str): Nome do dataset no YAML (por exemplo, 'monuseg').
        config_key (str): Chave específica da configuração dentro do dataset
            (por exemplo, 'monuseg_training').
        config (dict): Configuração específica carregada do YAML.
        loader_config (dict): Configuração do DataLoader.
        preprocessing (dict): Configuração de pré-processamento.
        root_dir (str): Diretório raiz do dataset.
        transform: Transformações a serem aplicadas aos dados.
    """

    YAML_CONFIG_PATH = 'configs/datasets.yml'

    def __init__(
      self,
      dataset_name: str,
      config_key: str,
      transform: Optional[Any] = None,
      yaml_path: str = YAML_CONFIG_PATH
      ) -> None:
        """Inicializa o dataset carregando a configuração a partir do YAML.

        Args:
            dataset_name (str): Nome do dataset no YAML (por exemplo, 'monuseg').
            config_key (str): Chave específica da configuração dentro do dataset
                (por exemplo, 'monuseg_training' para treino, 'monuseg_test' para teste).
            transform: Transformações a serem aplicadas. Padrão: None.
            yaml_path (str): Caminho para o arquivo de configuração YAML.
                Padrão: 'configs/datasets.yml'.

        Raises:
            FileNotFoundError: Se o arquivo YAML não existir.
            KeyError: Se dataset_name ou config_key não existirem no YAML.
            ValueError: Se root_dir não existir ou a configuração estiver incompleta.
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
        """Carrega as configurações a partir do arquivo YAML.

        Raises:
            FileNotFoundError: Se o arquivo YAML não existir.
            KeyError: Se dataset_name ou config_key não estiverem no YAML.
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
        """Valida se a configuração possui todos os campos obrigatórios.

        Raises:
            KeyError: Se algum campo obrigatório estiver ausente.
        """
        required_keys = ['root_dir', 'image_dir', 'mask_dir', 'image_extension',
                         'mask_extension']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Required config field missing for '{self.config_key}': {key}")

    @abstractmethod
    def __len__(self) -> int:
        """Retorna o número de amostras no dataset.

        Returns:
            int: Total de amostras.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retorna uma amostra do dataset pelo índice.

        Args:
            idx (int): Índice da amostra.

        Returns:
            dict: Dicionário contendo:
                - 'image': Imagem carregada.
                - 'mask': Máscara correspondente.
                - 'image_name': Nome do arquivo da imagem (opcional).
        """
        pass

    @abstractmethod
    def _load_image(self, image_path: str) -> Any:
        """Carrega uma imagem a partir do arquivo.

        Args:
            image_path (str): Caminho para a imagem.

        Returns:
            Imagem carregada (no formato específico do dataset).
        """
        pass

    @abstractmethod
    def _load_mask(self, mask_path: str) -> Any:
        """Carrega uma máscara a partir do arquivo.

        Args:
            mask_path (str): Caminho para a máscara.

        Returns:
            Máscara carregada (no formato específico do dataset).
        """
        pass

    @abstractmethod
    def _get_file_pairs(self) -> list:
        """Retorna a lista de pares (imagem, máscara).

        Returns:
            list: Lista de tuplas (image_path, mask_path).
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Retorna a configuração específica carregada.

        Returns:
            dict: Configuração do dataset (por exemplo, monuseg_training).
        """
        return self.config

    def get_loader_config(self) -> Dict[str, Any]:
        """Retorna a configuração do DataLoader (batch_size, num_workers, etc).

        Returns:
            dict: Configuração do loader.
        """
        return self.loader_config

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Retorna a configuração de pré-processamento (image_size, normalize, etc).

        Returns:
            dict: Configuração de pré-processamento.
        """
        return self.preprocessing

    def get_dataset_name(self) -> str:
        """Retorna o nome do dataset.

        Returns:
            str: Nome do dataset (por exemplo, 'monuseg').
        """
        return self.dataset_name

    def get_config_key(self) -> str:
        """Retorna a chave específica da configuração.

        Returns:
            str: Chave da configuração (por exemplo, 'monuseg_training').
        """
        return self.config_key

    def set_transform(self, transform: Any) -> None:
        """Define as transformações a serem aplicadas.

        Args:
            transform: Objeto de transformação.
        """
        self.transform = transform

    def get_image_dir(self) -> str:
        """Retorna o caminho completo para o diretório de imagens.

        Returns:
            str: Caminho para o diretório de imagens.
        """
        return os.path.join(self.root_dir, self.config['image_dir'])

    def get_mask_dir(self) -> str:
        """Retorna o caminho completo para o diretório de máscaras.

        Returns:
            str: Caminho para o diretório de máscaras.
        """
        return os.path.join(self.root_dir, self.config['mask_dir'])

    def get_stats(self) -> Dict[str, Any]:
        """Retorna as estatísticas do dataset.

        Returns:
            dict: Dicionário com estatísticas (tamanho, configuração, caminhos, etc).
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
