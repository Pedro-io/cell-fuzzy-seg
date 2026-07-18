from abc import ABC, abstractmethod

import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """Classe base abstrata para redes neurais do projeto.

    Esta classe define o contrato comum que toda rede neural deve seguir,
    garantindo uma interface padronizada para inferência e utilização da rede.
    As subclasses são responsáveis por implementar os comportamentos essenciais
    de inicialização, passagem de dados pela rede e geração de previsões.

    Por herdar de nn.Module, todas as subclasses ganham automaticamente:
    - .to(device), .cuda(), .cpu()
    - .parameters() (necessário para o optimizer)
    - .state_dict() / .load_state_dict() no nível correto
    - .eval() / .train() chamados diretamente na instância
    - suporte a nn.DataParallel, hooks, torch.compile, etc.
    """

    def __init__(self, config):
        """Inicializa a rede neural com a configuração fornecida.

        Args:
            config (dict): Dicionário contendo configurações da rede, como
                hiperparâmetros, arquitetura e outras opções específicas.
        """
        super().__init__()
        self._config = config

    @abstractmethod
    def forward(self, x):
        """Executa o fluxo de dados pela rede (inferência básica).

        Args:
            x: Entrada da rede (ex.: tensor de imagem ou dados).

        Returns:
            Saída bruta da rede (ex.: logits ou features).
        """
        pass

    @abstractmethod
    def predict(self, x):
        """Realiza previsões finais com pós-processamento.

        Args:
            x: Entrada da rede (ex.: tensor de imagem ou dados).

        Returns:
            Previsões processadas (ex.: classes preditas ou máscaras segmentadas).
        """
        pass
    
    def get_config(self):
        """Retorna a configuração da rede.

        Returns:
            dict: Dicionário contendo a configuração da rede.
        """
        return self._config

