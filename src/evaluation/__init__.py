"""Métricas, avaliação e registro de resultados de segmentação."""

from .metrics import METRIC_NAMES, binary_metrics, evaluate
from .report import provenance, save_run, summarize

__all__ = ["METRIC_NAMES", "binary_metrics", "evaluate", "provenance", "save_run", "summarize"]
