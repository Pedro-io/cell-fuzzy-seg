import segmentation_models_pytorch as smp
from .base_networks import BaseNetwork


class MarkerNet(BaseNetwork):
    """Segmentation model that predicts cell markers from input images.

    This network is designed to produce marker seeds for downstream watershed or
    fuzzy segmentation. It can be configured with different backbones and
    pretrained weights via the ``segmentation_models_pytorch`` library.

    Attributes:
        model: The underlying segmentation model from ``smp``.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        name: str = "MarkerNet",
        aux_params: dict = None,
    ):
        """Initializes MarkerNet.

        Args:
            backbone: Name of the backbone architecture to use (e.g., "resnet34").
            pretrained: Whether to initialize the backbone with ImageNet weights.
            name: Identifier for this network.
            aux_params: Parameters for the auxiliary segmentation head. 
        """
        super().__init__(name=name)
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=4,
            classes=1,
            aux_params=aux_params,
        )

    def forward(self, x):
        """Runs a forward pass through the model.

        Args:
            x: Input tensor of shape (N, 4, H, W).

        Returns:
            Output tensor of shape (N, 1, H, W) representing predicted markers.
        """
        return self.model(x)