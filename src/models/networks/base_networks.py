from abc import ABC, abstractmethod


class BaseNetwork(ABC):
    """Classe base abstrata para redes neurais.

    Esta classe define a interface comum para todas as redes neurais no projeto,
    garantindo que subclasses implementem métodos essenciais para inicialização,
    inferência, treinamento, avaliação e persistência.
    """

    @abstractmethod
    def __init__(self, config):
        """Inicializa a rede neural com a configuração fornecida.

        Args:
            config (dict): Dicionário contendo configurações da rede, como
                hiperparâmetros, arquitetura e outras opções específicas.
        """
        pass

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

    @abstractmethod
    def train_step(self, batch, optimizer, loss_fn):
        """Executa um passo de treinamento.

        Args:
            batch: Lote de dados de treinamento (ex.: dicionário com 'input' e 'target').
            optimizer: Otimizador para atualizar os pesos.
            loss_fn: Função de perda para calcular o erro.
        """
        pass

    @abstractmethod
    def validation_step(self, batch, loss_fn):
        """Executa um passo de validação.

        Args:
            batch: Lote de dados de validação (ex.: dicionário com 'input' e 'target').
            loss_fn: Função de perda para calcular o erro.

        Returns:
            Valor da perda para o lote.
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader, metrics):
        """Avalia a rede em um conjunto de dados completo.

        Args:
            data_loader: Carregador de dados para iteração.
            metrics (list): Lista de métricas a serem calculadas (ex.: accuracy, IoU).

        Returns:
            dict: Dicionário com os valores das métricas calculadas.
        """
        pass

    @abstractmethod
    def save(self, path):
        """Salva o estado da rede em um arquivo.

        Args:
            path (str): Caminho do arquivo onde salvar o modelo.
        """
        pass

    @abstractmethod
    def load(self, path):
        """Carrega o estado da rede de um arquivo.

        Args:
            path (str): Caminho do arquivo de onde carregar o modelo.
        """
        pass

    @abstractmethod
    def get_config(self):
        """Retorna a configuração atual da rede.

        Returns:
            dict: Dicionário com a configuração da rede.
        """
        pass
