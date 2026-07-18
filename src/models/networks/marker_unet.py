import segmentation_models_pytorch as smp
import torch

from ..configs.marker_unet_config import UNET_CONFIG
from .base_networks import BaseNetwork


class MarkerUNet(BaseNetwork):
    def __init__(self, config=UNET_CONFIG):
        super().__init__(config)
        self.model = smp.Unet(**config)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)

        return probs
