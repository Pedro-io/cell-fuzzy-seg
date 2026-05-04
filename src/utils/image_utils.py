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
