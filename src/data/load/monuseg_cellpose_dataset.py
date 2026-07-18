from typing import Any, Dict, Optional

import numpy as np

from src.pipeline.steps.cellpose_step import CellposeStep
from src.utils.image_utils import to_float32_rgb

from .monuseg_dataset import MonusegDataset


class MonusegCellposeDataset(MonusegDataset):
    """Dataset wrapper that augments Monuseg samples with Cellpose output.

    This dataset subclass wraps the original MonusegDataset and runs Cellpose
    inference on each sample image. It preserves the original ground truth
    (`ground_truth`) for metric and loss computation but provides an
    additional key, `cellpose_segmentation`, containing the segmentation
    produced by Cellpose. Optionally caches Cellpose outputs to avoid
    repeated inference.

    Pipeline (conceptual):
        1. load data
        2. normalization
        3. cellpose inference
        4. distance map computation

    Args:
        dataset_name: Name of the dataset to load (passed to MonusegDataset).
        config_key: Configuration key used by the base dataset loader.
        transform: Optional transform applied by the base dataset.
        yaml_path: Path to dataset YAML configuration (defaults to base class
            YAML_CONFIG_PATH).
        cellpose_step: Optional preconstructed `CellposeStep`. If not provided,
            a new `CellposeStep` is created with `cellpose_batch_size`.
        cellpose_batch_size: Batch size to use when instantiating
            `CellposeStep` (ignored if `cellpose_step` is provided).
        cache: If True, Cellpose outputs are cached in-memory keyed by index.

    Attributes:
        cache: Whether outputs are cached.
        _cache: Internal dict used when caching is enabled.
        cellpose_step: The `CellposeStep` instance used to run inference.
    """

    def __init__(
        self,
        dataset_name: str,
        config_key: str,
        transform: Optional[Any] = None,
        yaml_path: str = MonusegDataset.YAML_CONFIG_PATH,
        cellpose_step: Optional[CellposeStep] = None,
        cellpose_batch_size: int = 8,
        cache: bool = False,
    ) -> None:
        super().__init__(dataset_name, config_key, transform, yaml_path)
        self.cache = cache
        self._cache: Dict[int, np.ndarray] = {} if cache else {}

        if cellpose_step is None:
            self.cellpose_step = CellposeStep(batch_size=cellpose_batch_size)
        else:
            self.cellpose_step = cellpose_step

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a dataset sample augmented with Cellpose segmentation.

        The returned dict contains at least the keys produced by the base
        `MonusegDataset` implementation plus the `cellpose_segmentation` key.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary with dataset fields and an additional
            `cellpose_segmentation` numpy array with dtype `uint8`.
        """
        sample = super().__getitem__(idx)

        if self.cache and idx in self._cache:
            sample['cellpose_segmentation'] = self._cache[idx]
            return sample

        cellpose_mask = self._compute_cellpose_mask(sample['image'])
        sample['cellpose_segmentation'] = cellpose_mask

        if self.cache:
            self._cache[idx] = cellpose_mask

        return sample

    def _compute_cellpose_mask(self, image: np.ndarray) -> np.ndarray:
        """Run Cellpose inference on a normalized image and return the mask.

        This helper converts the input image to a float32 RGB representation
        in the [0, 1] range using `to_float32_rgb`, runs the configured
        `CellposeStep`, and returns the segmentation mask as an 8-bit
        numpy array.

        Args:
            image: Input image array (expected to be already normalized
                according to the dataset pipeline).

        Returns:
            Segmentation mask produced by Cellpose with dtype `uint8`.
        """
        # Normalize image before Cellpose
        normalized_image = to_float32_rgb(image)
        data = {'image': normalized_image}
        output = self.cellpose_step.forward(data)
        mask = output['segmentation']

        return mask.astype(np.uint8)


def prepare_image(image: np.ndarray) -> np.ndarray:
    """Convert image to RGB float32 in range [0, 1].

    This is a thin wrapper around `to_float32_rgb` kept for readability in
    the data collation pipeline. The function does not perform additional
    normalization; the input image is expected to be normalized by the
    dataset pipeline already.

    Args:
        image: Input image array.

    Returns:
        RGB image as float32 numpy array with values in [0, 1].
    """
    return to_float32_rgb(image)


def compute_distance_map(binary_mask: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Compute a normalized distance map for a binary mask.

    Computes the Euclidean distance transform of the foreground in
    `binary_mask`, applies an optional power transform, normalizes the
    result to [0, 1], and returns a distance map inverted so that object
    centers have lower values (useful for some loss formulations).

    Args:
        binary_mask: Binary mask array where foreground is non-zero.
        power: Exponent applied to the distance transform before
            normalization. Defaults to 1.0 (no change).

    Returns:
        Float32 numpy array with values in [0, 1] representing the
        inverted, normalized distance map.
    """
    from scipy.ndimage import distance_transform_edt

    dt = distance_transform_edt(binary_mask.astype(np.uint8))
    dt = np.power(dt, power)
    if dt.max() > 0:
        dt = dt / dt.max()
    dmap = 1.0 - dt
    return dmap.astype(np.float32)


def collate_fn_with_cellpose(samples):
    """Collate function that uses Cellpose segmentation as the 4th channel.

    This function prepares a batch for training by stacking images where
    the Cellpose segmentation is appended as a 4th channel (RGBA layout
    where the 4th channel is the Cellpose binary mask). It also computes
    distance maps from the ground truth masks.

    Args:
        samples: Iterable of samples produced by a dataset `__getitem__`.

    Returns:
        A dictionary with the following keys:
            - `image`: Tensor of shape (B, 4, H, W) float32
            - `distance_map`: Tensor of shape (B, 1, H, W) float32
            - `ground_truth`: Tensor of shape (B, 1, H, W) float32
            - `cellpose_segmentation`: Tensor of shape (B, H, W) float32
            - `id`: List of sample ids
    """
    import torch

    images_list, dmaps_list, masks_list, ids = [], [], [], []

    for s in samples:
        img = prepare_image(s['image'])
        cp_mask = (s['cellpose_segmentation'] > 0).astype(np.float32)

        rgba = np.concatenate([img, cp_mask[:, :, None]], axis=-1)
        rgba = rgba.transpose(2, 0, 1)

        mask = s['ground_truth'].astype(np.float32)
        dmap = compute_distance_map(mask)

        images_list.append(rgba)
        dmaps_list.append(dmap[None])
        masks_list.append(mask[None])
        ids.append(s['id'])

    return {
        'image': torch.tensor(np.stack(images_list), dtype=torch.float32),
        'distance_map': torch.tensor(np.stack(dmaps_list), dtype=torch.float32),
        'ground_truth': torch.tensor(np.stack(masks_list), dtype=torch.float32),
        'segmentation': torch.tensor(np.stack([s['cellpose_segmentation'] for s in samples]), dtype=torch.float32),
        'id': ids,
    }
