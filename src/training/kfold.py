"""Validação cruzada k-fold nas imagens de treino e treino final com avaliação única no teste.

Cada *fold* treina um modelo novo nas imagens de treino do *fold*, escolhe a época de melhor Dice de
validação (com parada antecipada opcional) e avalia, com esse modelo, as imagens de validação. O
treino final usa todas as imagens de treino pelo número de épocas da mediana das melhores épocas dos
*folds* e é avaliado **uma vez** no teste.

Com ``models_dir``, os modelos são gravados (:mod:`src.training.checkpoints`): o da melhor época de
cada *fold* como ``fold<k>.pt`` e o do treino final como ``final.pt``, cada um com um ``.json``.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.load.preprocessed_dataset import PreprocessedDataset, make_loader
from src.evaluation.metrics import evaluate
from src.evaluation.report import provenance
from src.training.callbacks.best_model_callback import BestModelCallback
from src.training.checkpoints import FINAL_NAME, ensure_new_models_dir, fold_name, save_checkpoint
from src.training.trainer import Trainer, TrainerCallback
from src.training.training_loop import TrainingLoop
from src.utils.logger import logger

# Monta, a cada chamada, um experimento novo: (modelo treinável, pipeline de treino, compositor de perdas).
ExperimentFactory = Callable[[], Tuple[torch.nn.Module, Any, torch.nn.Module]]


@dataclass
class TrainConfig:
    """Hiperparâmetros do treino de cada *fold* e do treino final.

    Atributos:
        k: Número de *folds*.
        batch_size: Imagens por batch.
        max_epochs: Máximo de épocas por *fold*; também fixa a duração do *cosine schedule*.
        patience: Validações sem melhora até a parada antecipada (``None``: sem parada).
        min_delta: Melhora mínima do Dice de validação para contar como melhora.
        lr: Taxa de aprendizado do Adam.
        grad_clip: Norma máxima do gradiente (``None``: sem *clipping*).
        threshold: Limiar de binarização nas métricas.
        seed: Semente base (divisão dos *folds*, inicialização, aumentação e embaralhamento).
        device: Dispositivo do treino.
        log_every: Intervalo, em épocas, dos logs do ``TrainingLoop``.
    """

    k: int = 5
    batch_size: int = 4
    max_epochs: int = 200
    patience: Optional[int] = 20
    min_delta: float = 0.0
    lr: float = 1e-4
    grad_clip: Optional[float] = 1.0
    threshold: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 10


@dataclass
class FoldResult:
    """Resultado de um *fold*: ids, melhor época, curvas e métricas por imagem da validação."""

    fold: int
    train_ids: List[str]
    val_ids: List[str]
    best_epoch: int
    best_dice: float
    epochs_run: int
    val_dice_history: List[Dict[str, float]]
    train_loss_history: List[float]
    rows: List[Dict[str, Any]] = field(default_factory=list)


def make_folds(ids: Sequence[str], k: int = 5, seed: int = 42) -> List[List[str]]:
    """Divide os ids em ``k`` *folds* disjuntos, de tamanhos o mais iguais possível.

    A divisão é por imagem, aleatória com ``seed``, sem estratificação.

    Raises:
        ValueError: Se ``k`` for menor que 2 ou maior que o número de ids, ou se houver ids repetidos.
    """
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Há ids repetidos.")
    if not 2 <= k <= len(ids):
        raise ValueError(f"k deve estar entre 2 e {len(ids)}, recebido {k}.")
    order = np.random.default_rng(seed).permutation(len(ids))
    return [[ids[i] for i in part] for part in np.array_split(order, k)]


def _train(
    factory: ExperimentFactory,
    samples: Mapping[str, Any],
    train_ids: Sequence[str],
    val_ids: Optional[Sequence[str]],
    config: TrainConfig,
    seed: int,
    num_epochs: int,
    extra_callbacks: Sequence[TrainerCallback],
) -> Tuple[torch.nn.Module, Any, Optional[BestModelCallback], Dict[str, Any], Optional[Any]]:
    """Treina um modelo novo.

    Returns:
        ``(modelo, pipeline, callback do melhor modelo, histórico, val_loader)``.
    """
    torch.manual_seed(seed)
    model, pipeline, composer = factory()

    train_loader = make_loader(
        PreprocessedDataset(samples, train_ids, augment=True, seed=seed),
        config.batch_size, shuffle=True, seed=seed,
    )
    val_loader = None
    if val_ids is not None:
        val_loader = make_loader(PreprocessedDataset(samples, val_ids), config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # O schedule cobre sempre max_epochs; o treino final só para antes (em num_epochs).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs * len(train_loader)
    )
    best = None
    callbacks = list(extra_callbacks)
    if val_loader is not None:
        best = BestModelCallback(model, patience=config.patience, min_delta=config.min_delta,
                                 threshold=config.threshold)
        callbacks.insert(0, best)
    trainer = Trainer(pipeline, composer, optimizer, scheduler=scheduler, callbacks=callbacks,
                      device=config.device, grad_clip=config.grad_clip)
    history = TrainingLoop(trainer, train_loader, val_loader, num_epochs=num_epochs,
                           log_every=config.log_every).run()
    return model, pipeline, best, history, val_loader


def run_kfold(
    factory: ExperimentFactory,
    samples: Mapping[str, Any],
    ids: Sequence[str],
    config: TrainConfig,
    extra_callbacks: Sequence[TrainerCallback] = (),
    models_dir: Optional[str] = None,
    overwrite: bool = False,
) -> List[FoldResult]:
    """Roda a validação cruzada: um modelo novo por *fold*, avaliado na época de melhor Dice.

    Args:
        factory: Função que monta um experimento novo a cada chamada.
        samples: Amostras de treino carregadas por ``load_preprocessed``.
        ids: Ids das imagens de treino que entram na validação cruzada.
        config: Hiperparâmetros.
        extra_callbacks: Callbacks adicionais (ex.: ``GradNormCallback``), repassados a todos os *folds*.
        models_dir: Se informado, grava o modelo da melhor época de cada *fold* nessa pasta
            (``fold<k>.pt`` + ``fold<k>.json``). ``None``: não grava.
        overwrite: Se ``False``, recusa (antes de treinar) gravar por cima de modelos existentes.

    Returns:
        Um :class:`FoldResult` por *fold*, com as métricas por imagem da validação (cada linha com
        ``"fold"``), calculadas com o modelo da melhor época.

    Raises:
        FileExistsError: Se ``models_dir`` já tiver modelos dos *folds* e ``overwrite`` for
            ``False``.
    """
    if models_dir is not None:
        ensure_new_models_dir(models_dir, [fold_name(k) for k in range(config.k)], overwrite)
    results = []
    folds = make_folds(ids, config.k, config.seed)
    for fold, val_ids in enumerate(folds):
        held_out = set(val_ids)
        train_ids = [i for i in ids if i not in held_out]
        logger.info(f"[kfold] fold {fold + 1}/{config.k}: {len(train_ids)} treino, {len(val_ids)} validação")
        model, pipeline, best, history, val_loader = _train(
            factory, samples, train_ids, val_ids, config, seed=config.seed + fold,
            num_epochs=config.max_epochs, extra_callbacks=extra_callbacks,
        )
        best.restore_best()
        rows = evaluate(pipeline, val_loader, threshold=config.threshold, device=config.device)
        for row in rows:
            row["fold"] = fold
        if models_dir is not None:
            save_checkpoint(model, models_dir, fold_name(fold), {
                "fold": fold, "train_ids": train_ids, "val_ids": list(val_ids),
                "best_epoch": best.best_epoch, "best_dice": best.best_dice,
                "epochs_run": history["epochs_run"], "config": _config_dict(config),
                "procedencia": provenance(),
            })
        results.append(FoldResult(
            fold=fold, train_ids=train_ids, val_ids=list(val_ids), best_epoch=best.best_epoch,
            best_dice=best.best_dice, epochs_run=history["epochs_run"],
            val_dice_history=list(best.history), train_loss_history=list(history["train_loss"]),
            rows=rows,
        ))
        logger.info(f"[kfold] fold {fold + 1}: melhor Dice {best.best_dice:.4f} na época {best.best_epoch}")
    return results


def _config_dict(config: Any) -> Dict[str, Any]:
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)


def final_epochs(folds: Sequence[FoldResult]) -> int:
    """Número de épocas do treino final: a mediana das melhores épocas dos *folds*, arredondada.

    O arredondamento é o do Python (para o par mais próximo: 10,5 → 10).
    """
    return int(round(float(np.median([f.best_epoch for f in folds]))))


def run_final(
    factory: ExperimentFactory,
    train_samples: Mapping[str, Any],
    test_samples: Mapping[str, Any],
    config: TrainConfig,
    num_epochs: int,
    extra_callbacks: Sequence[TrainerCallback] = (),
    models_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Any]:
    """Treina com todas as imagens de treino por ``num_epochs`` épocas e avalia o teste uma vez.

    Args:
        factory: Função que monta um experimento novo.
        train_samples: Amostras de treino (todas entram no treino).
        test_samples: Amostras de teste.
        config: Hiperparâmetros.
        num_epochs: Épocas do treino (ex.: :func:`final_epochs` da validação cruzada).
        extra_callbacks: Callbacks adicionais.
        models_dir: Se informado, grava o modelo treinado como ``final.pt`` + ``final.json``.
        overwrite: Se ``False``, recusa (antes de treinar) gravar por cima de um ``final.pt``.

    Returns:
        ``(linhas, histórico, pipeline)``: as métricas por imagem do teste (com a linha de base do
        Cellpose), o histórico do treino e o pipeline treinado (para visualizar as predições).

    Raises:
        FileExistsError: Se ``models_dir`` já tiver ``final.pt`` e ``overwrite`` for ``False``.
    """
    if models_dir is not None:
        ensure_new_models_dir(models_dir, [FINAL_NAME], overwrite)
    model, pipeline, _, history, _ = _train(
        factory, train_samples, list(train_samples), None, config, seed=config.seed + config.k,
        num_epochs=num_epochs, extra_callbacks=extra_callbacks,
    )
    if models_dir is not None:
        save_checkpoint(model, models_dir, FINAL_NAME, {
            "train_ids": list(train_samples), "num_epochs": num_epochs,
            "config": _config_dict(config), "procedencia": provenance(),
        })
    test_loader = make_loader(PreprocessedDataset(test_samples), config.batch_size, shuffle=False)
    rows = evaluate(pipeline, test_loader, threshold=config.threshold, device=config.device)
    return rows, history, pipeline
