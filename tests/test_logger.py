"""Testes da configuração do logger: nível padrão INFO, formato curto e troca de nível."""

import importlib
import io
import re
import sys

import src.utils.logger as logger_module
from src.utils.logger import logger, set_log_level


def _capture(monkeypatch, emit, level=None):
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    try:
        if level is not None:
            set_log_level(level)
        emit()
    finally:
        monkeypatch.undo()
        set_log_level("INFO")
    return buffer.getvalue()


def test_info_level_hides_debug_and_uses_short_format(monkeypatch):
    def emit():
        logger.debug("detalhe interno")
        logger.info("progresso")
        logger.warning("cuidado")

    out = _capture(monkeypatch, emit, level="INFO")

    assert "detalhe interno" not in out
    assert re.search(r"^\d\d:\d\d:\d\d \| progresso$", out, re.MULTILINE)
    assert re.search(r"^\d\d:\d\d:\d\d \| WARNING \| cuidado$", out, re.MULTILINE)


def test_debug_level_shows_details(monkeypatch):
    out = _capture(monkeypatch, lambda: logger.debug("detalhe interno"), level="debug")

    assert "detalhe interno" in out


def test_environment_variable_sets_the_initial_level(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    monkeypatch.setenv(logger_module.LOG_LEVEL_ENV, "WARNING")
    try:
        importlib.reload(logger_module)
        logger.info("progresso")
        logger.warning("cuidado")
    finally:
        monkeypatch.undo()
        importlib.reload(logger_module)

    assert "progresso" not in buffer.getvalue()
    assert "cuidado" in buffer.getvalue()
