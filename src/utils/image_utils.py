import numpy as np


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Converte uma imagem para um array uint8 de 3 canais normalizado para [0, 255].

    Imagens em tons de cinza (H, W) são expandidas para (H, W, 3) duplicando o
    único canal. Arrays que não são uint8 são escalonados para que o valor máximo
    seja mapeado para 255; uma imagem composta apenas por zeros é mantida inalterada.

    Args:
        image: Array de entrada com formato ``(H, W)`` ou ``(H, W, C)``.

    Returns:
        Array uint8 com formato ``(H, W, 3)``.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() > 0:
            image = image / image.max() * 255
        image = image.astype(np.uint8)

    return image


def to_float32_rgb(image: np.ndarray) -> np.ndarray:
    """Converte uma imagem para um array float32 de 3 canais normalizado para [0, 1].

    Imagens em tons de cinza (H, W) são expandidas para (H, W, 3) duplicando o
    único canal. Imagens de um canal (H, W, 1) são expandidas para 3 canais por concatenação.

    Args:
        image: Array de entrada com formato ``(H, W)`` ou ``(H, W, C)``.

    Returns:
        Array float32 com formato ``(H, W, 3)`` e valores em [0, 1].
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    image = image[:, :, :3].astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image
