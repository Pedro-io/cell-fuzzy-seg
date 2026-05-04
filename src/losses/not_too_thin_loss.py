"""Not-too-thin morphological regularization loss module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _morpho(image: torch.Tensor, kernel: torch.Tensor, operation: str) -> torch.Tensor:
    """Differentiable morphological dilation or erosion via unfolding.

    Args:
        image: 2-D image tensor of shape ``(H, W)``.
        kernel: 2-D structuring element tensor of shape ``(Kh, Kw)``.
        operation: Either ``"dilate"`` or ``"erode"``.

    Returns:
        Morphologically processed tensor of the same shape as ``image``.

    Raises:
        ValueError: If ``operation`` is not ``"dilate"`` or ``"erode"``.
    """
    p1 = kernel.shape[0] // 2
    p2 = kernel.shape[1] // 2
    im = image.reshape(1, 1, *image.shape)
    k = kernel.reshape(1, -1, 1)

    if operation == "dilate":
        im = F.pad(im, (p1, p1, p2, p2), mode="constant", value=0)
        folds = F.unfold(im, kernel.shape)
        k = (1 - k) * -9_999_999_999
        folds_proc, _ = torch.max(folds + k, dim=1)
    elif operation == "erode":
        im = F.pad(im, (p1, p1, p2, p2), mode="constant", value=1)
        folds = F.unfold(im, kernel.shape)
        k = (1 - k) * 9_999_999_999
        folds_proc, _ = torch.min(folds + k, dim=1)
    else:
        raise ValueError(f"Unsupported operation '{operation}'. Use 'dilate' or 'erode'.")

    return folds_proc.reshape(image.shape)


class NotTooThinLoss(nn.Module):
    """Regularization loss that penalizes thin, filament-like predictions.

    Applies a morphological opening (erosion followed by dilation) to identify
    thin structures removed by the opening, then penalizes their presence in
    the prediction.

    Expected input: a 2-D image tensor of shape ``(H, W)``.
    """

    def __init__(self, kernel: torch.Tensor, weight: float = 0.5) -> None:
        """Initializes NotTooThinLoss.

        Args:
            kernel: 2-D structuring element used for the morphological opening.
            weight: Scalar multiplier applied to the loss.
        """
        super().__init__()
        self.register_buffer("kernel", kernel)
        self.weight = weight

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the not-too-thin loss.

        Args:
            image: 2-D image tensor of shape ``(H, W)``.

        Returns:
            Scalar weighted loss value.
        """
        opened = _morpho(_morpho(image, self.kernel, "erode"), self.kernel, "dilate")
        removed = ((image - opened) > 0).float() * image
        removed = _morpho(removed, self.kernel, "dilate")
        target = torch.max(removed, image)
        return self.weight * torch.mean(target - image)
