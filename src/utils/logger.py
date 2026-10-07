"""Logger compartilhado do projeto (Loguru), com nível INFO e formato curto por padrão.

O nível pode ser trocado sem mexer no código, pela variável de ambiente ``CELL_FUZZY_LOG_LEVEL``
(ex.: ``DEBUG`` para ver os detalhes por imagem e por batch), ou em tempo de execução com
:func:`set_log_level`.
"""

import os
import sys
from typing import Any, Dict, Optional

from loguru import logger

LOG_LEVEL_ENV = "CELL_FUZZY_LOG_LEVEL"
DEFAULT_LEVEL = "INFO"

_handler_id: Optional[int] = None


def _format(record: Dict[str, Any]) -> str:
    """Hora e mensagem; o nível só aparece a partir de WARNING."""
    level = "" if record["level"].no < logger.level("WARNING").no else "{level} | "
    return "{time:HH:mm:ss} | " + level + "{message}\n{exception}"


def set_log_level(level: str) -> None:
    """Troca o nível mínimo das mensagens do projeto (ex.: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

    Substitui só o destino criado por este módulo; destinos adicionados por quem usa o logger
    continuam ativos.
    """
    global _handler_id
    if _handler_id is not None:
        logger.remove(_handler_id)
    else:
        logger.remove()  # remove o destino padrão do Loguru, que mostra tudo desde DEBUG
    _handler_id = logger.add(sys.stderr, level=level.upper(), format=_format)


set_log_level(os.environ.get(LOG_LEVEL_ENV, DEFAULT_LEVEL))

__all__ = ["logger", "set_log_level", "LOG_LEVEL_ENV"]
