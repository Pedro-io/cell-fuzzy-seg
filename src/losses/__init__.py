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
    TopologyTerm,
    TVTerm,
)
from .total_variation_loss import TotalVariationLoss
from .topology import (
    ComponentTree,
    ComponentTreeFunction,
    TopologyLoss,
    attribute_max_altitudes,
    attribute_saddle_nodes,
)

__all__ = [
    # Primitives
    "BorderLoss",
    "ComponentTree",
    "ComponentTreeFunction",
    "DistanceMapLoss",
    "NotTooThinLoss",
    "ObjectSizeLoss",
    "RMSEAccuracy",
    "RMSELoss",
    "SoftDiceLoss",
    "TopologyLoss",
    "TotalVariationLoss",
    "attribute_max_altitudes",
    "attribute_saddle_nodes",
    # Strategy pattern
    "LossTerm",
    "LossComposer",
    "BorderTerm",
    "DiceTerm",
    "DMapTerm",
    "NotTooThinTerm",
    "RMSETerm",
    "SizeTerm",
    "TopologyTerm",
    "TVTerm",
]
