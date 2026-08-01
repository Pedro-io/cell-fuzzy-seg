"""Módulo de perda de regularização morfológica "não muito fina"."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _morpho(image: torch.Tensor, kernel: torch.Tensor, operation: str) -> torch.Tensor:
    """Dilatação ou erosão morfológica diferenciável via unfolding.

    Args:
        image: Tensor de imagem 2-D com formato ``(H, W)``.
        kernel: Tensor do elemento estruturante 2-D com formato ``(Kh, Kw)``.
        operation: Pode ser ``"dilate"`` ou ``"erode"``.

    Returns:
        Tensor processado morfologicamente com o mesmo formato de ``image``.

    Raises:
        ValueError: Se ``operation`` não for ``"dilate"`` nem ``"erode"``.
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
    """Perda de regularização que penaliza previsões finas, semelhantes a filamentos.

    Aplica uma abertura morfológica (erosão seguida de dilatação) para
    identificar estruturas finas removidas pela abertura e, em seguida,
    penaliza sua presença na previsão.

    Entrada esperada: tensor de imagem 2-D com formato ``(H, W)``.
    """

    def __init__(self, kernel: torch.Tensor, weight: float = 0.5) -> None:
        """Inicializa NotTooThinLoss.

        Args:
            kernel: Elemento estruturante 2-D usado para a abertura morfológica.
            weight: Multiplicador escalar aplicado à perda.
        """
        super().__init__()
        self.register_buffer("kernel", kernel)
        self.weight = weight

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Calcula a perda "não muito fina".

        Args:
            image: Tensor de imagem 2-D com formato ``(H, W)``.

        Returns:
            Valor escalar da perda ponderada.
        """
        opened = _morpho(_morpho(image, self.kernel, "erode"), self.kernel, "dilate")
        removed = ((image - opened) > 0).float() * image
        removed = _morpho(removed, self.kernel, "dilate")
        target = torch.max(removed, image)
        return self.weight * torch.mean(target - image)
