"""Módulo de configuração do logger.

Fornece uma instância compartilhada de `logger` (do Loguru) para ser importada
em diferentes partes do projeto, garantindo logging consistente. Novos sinks
ou formatos podem ser configurados centralmente aqui no futuro, se necessário.
"""

from loguru import logger

__all__ = ["logger"]
