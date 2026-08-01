from abc import ABC, abstractmethod
from typing import Any, Dict


class PipelineStep(ABC):
    """Classe base para todos os passos do pipeline.

    Cada passo:
    - Recebe um dicionário de dados
    - Processa esse dicionário
    - Retorna o dicionário atualizado
    """
    def __init__(self, name: str = "PipelineStep"):
        self.name = name

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Permite que o passo seja chamado como uma função."""
        return self.forward(data)

    @abstractmethod
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Lógica principal do passo.

        Deve ser implementada pelas subclasses.
        """
        pass
