"""Métricas e avaliação de segmentação."""

from .metrics import METRIC_NAMES, binary_metrics, evaluate

__all__ = ["METRIC_NAMES", "binary_metrics", "evaluate"]
