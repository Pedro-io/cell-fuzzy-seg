"""Loss and regularization modules for cell segmentation."""

from .border_loss import BorderLoss
from .distance_map_loss import DistanceMapLoss
from .loss_composer import LossComposer
from .loss_term import LossTerm
from .not_too_thin_loss import NotTooThinLoss
from .object_size_loss import ObjectSizeLoss
from .rmse_accuracy import RMSEAccuracy
from .rmse_loss import RMSELoss
from .soft_dice_loss import SoftDiceLoss
from .terms import (
    BorderTerm,
    DiceTerm,
    DMapTerm,
    NotTooThinTerm,
    RMSETerm,
    SizeTerm,
    TVTerm,
)
from .total_variation_loss import TotalVariationLoss

__all__ = [
    # Primitives
    "BorderLoss",
    "DistanceMapLoss",
    "NotTooThinLoss",
    "ObjectSizeLoss",
    "RMSEAccuracy",
    "RMSELoss",
    "SoftDiceLoss",
    "TotalVariationLoss",
    # Strategy pattern
    "LossTerm",
    "LossComposer",
    "BorderTerm",
    "DiceTerm",
    "DMapTerm",
    "NotTooThinTerm",
    "RMSETerm",
    "SizeTerm",
    "TVTerm",
]
