"""Gravação e leitura dos modelos treinados (a MarkerUNet de cada *fold* e a do treino final).

Cada modelo é gravado como ``<nome>.pt`` (o ``state_dict``, com ``torch.save``) e ``<nome>.json`` ao
lado (de onde ele veio: *fold*, imagens de treino e de validação, melhor época, configuração,
procedência). Só os pesos da rede treinável são gravados; a rede final (o ScribblePrompt) é
congelada e vem do checkpoint dela.

A leitura usa ``torch.load(..., weights_only=True)``, que só aceita tensores e tipos simples: um
arquivo adulterado não consegue executar código ao ser carregado.
"""

import json
import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch

FINAL_NAME = "final"


def fold_name(fold: int) -> str:
    """Nome dos arquivos do modelo de um *fold* (``fold0``, ``fold1``...)."""
    return f"fold{fold}"


def checkpoint_paths(models_dir: str, name: str) -> Tuple[str, str]:
    """Caminhos ``(<models_dir>/<name>.pt, <models_dir>/<name>.json)``."""
    base = os.path.join(models_dir, name)
    return base + ".pt", base + ".json"


def save_checkpoint(model: torch.nn.Module, models_dir: str, name: str,
                    metadata: Optional[Mapping[str, Any]] = None) -> str:
    """Grava o ``state_dict`` de ``model`` e os metadados em ``models_dir``.

    Args:
        model: Rede cujos pesos são gravados (a MarkerUNet).
        models_dir: Pasta dos modelos do experimento (criada se não existir).
        name: Nome dos arquivos, sem extensão (ver :func:`fold_name` e ``FINAL_NAME``).
        metadata: Informações gravadas no ``.json`` ao lado (precisam ser serializáveis em JSON).

    Returns:
        Caminho do ``.pt`` gravado.
    """
    os.makedirs(models_dir, exist_ok=True)
    weights_path, meta_path = checkpoint_paths(models_dir, name)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, weights_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dict(metadata or {}), f, indent=2, ensure_ascii=False)
    return weights_path


def load_checkpoint(model: torch.nn.Module, models_dir: str, name: str) -> Dict[str, Any]:
    """Carrega em ``model`` os pesos gravados por :func:`save_checkpoint` e devolve os metadados.

    Args:
        model: Rede com a mesma arquitetura da que foi gravada.
        models_dir: Pasta dos modelos do experimento.
        name: Nome dos arquivos, sem extensão.

    Returns:
        Os metadados do ``.json`` (``{}`` se ele não existir).

    Raises:
        FileNotFoundError: Se o ``.pt`` não existir.
    """
    weights_path, meta_path = checkpoint_paths(models_dir, name)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Modelo não encontrado: {weights_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_pipeline(factory: Callable[[], Tuple[torch.nn.Module, Any, Any]], models_dir: str,
                  name: str) -> Tuple[Any, Dict[str, Any]]:
    """Monta um experimento com ``factory``, carrega os pesos e põe o pipeline em modo de avaliação.

    Args:
        factory: A função que montou o experimento no treino (devolve modelo, pipeline e perdas).
        models_dir: Pasta dos modelos do experimento.
        name: Nome dos arquivos, sem extensão.

    Returns:
        ``(pipeline, metadados)``.
    """
    model, pipeline, _ = factory()
    metadata = load_checkpoint(model, models_dir, name)
    for step in getattr(pipeline, "steps", []):
        set_training = getattr(step, "set_training", None)
        if set_training is not None:
            set_training(False)
    return pipeline, metadata


def ensure_new_models_dir(models_dir: str, names, overwrite: bool) -> None:
    """Recusa, antes de treinar, gravar por cima de modelos que já existem.

    Raises:
        FileExistsError: Se algum ``<name>.pt`` já existir em ``models_dir`` e ``overwrite`` for
            ``False``.
    """
    if overwrite:
        return
    existing = [checkpoint_paths(models_dir, n)[0] for n in names]
    existing = [p for p in existing if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            f"Já existem modelos em {models_dir} ({len(existing)}); "
            "use outra pasta ou overwrite=True."
        )
