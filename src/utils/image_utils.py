import numpy as np


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Converts an image to a 3-channel uint8 array normalised to [0, 255].

    Grayscale images (H, W) are expanded to (H, W, 3) by duplicating the
    single channel. Non-uint8 arrays are scaled so that the maximum value
    maps to 255; an all-zero image is left unchanged.

    Args:
        image: Input array of shape ``(H, W)`` or ``(H, W, C)``.

    Returns:
        uint8 array of shape ``(H, W, 3)``.
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
    """Converts an image to a 3-channel float32 array normalized to [0, 1].

    Grayscale images (H, W) are expanded to (H, W, 3) by duplicating the
    single channel. Single-channel images (H, W, 1) are expanded to 3
    channels by concatenation.

    Args:
        image: Input array of shape ``(H, W)`` or ``(H, W, C)``.

    Returns:
        float32 array of shape ``(H, W, 3)`` with values in [0, 1].
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    image = image[:, :, :3].astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image
