from typing import Any, Dict, Optional, Tuple

import numpy as np

from .monuseg_dataset import MonusegDataset
from src.pipeline.steps.cellpose_step import CellposeStep


class MonusegCellposeDataset(MonusegDataset):
    """Monuseg dataset wrapper that augments samples with Cellpose output.

    This wrapper keeps the original ground truth for loss/metric computation,
    but replaces the training signal used by MarkerNet with the actual
    Cellpose segmentation instead of the ground truth mask.
    """

    def __init__(
        self,
        dataset_name: str,
        config_key: str,
        transform: Optional[Any] = None,
        yaml_path: str = MonusegDataset.YAML_CONFIG_PATH,
        cellpose_step: Optional[CellposeStep] = None,
        cellpose_batch_size: int = 1,
        cache: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(dataset_name, config_key, transform, yaml_path)
        self.target_size = target_size
        self.cache = cache
        self._cache: Dict[int, np.ndarray] = {} if cache else {}

        if cellpose_step is None:
            self.cellpose_step = CellposeStep(batch_size=cellpose_batch_size)
        else:
            self.cellpose_step = cellpose_step

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        """Run Cellpose inference on a single image and return the mask."""
        data = {'image': image}
        output = self.cellpose_step.forward(data)
        mask = output['segmentation']

        if self.target_size is not None and mask.shape != self.target_size:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), self.target_size, interpolation=cv2.INTER_NEAREST)

        return mask.astype(np.uint8)


def prepare_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convert image to RGB float32 in range [0, 1] and resize to target size."""
    import cv2

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    image = image[:, :, :3]
    image = cv2.resize(image, size)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image


def compute_distance_map(binary_mask: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Compute a normalized distance map for a binary mask."""
    from scipy.ndimage import distance_transform_edt

    dt = distance_transform_edt(binary_mask.astype(np.uint8))
    dt = np.power(dt, power)
    if dt.max() > 0:
        dt = dt / dt.max()
    dmap = 1.0 - dt
    return dmap.astype(np.float32)


def collate_fn_with_cellpose(samples, target_size: Tuple[int, int] = (256, 256)):
    """Collate function that uses Cellpose segmentation as the 4th channel."""
    import cv2
    import torch

    images_list, dmaps_list, masks_list, ids = [], [], [], []

    for s in samples:
        img = prepare_image(s['image'], target_size)
        cp_mask = cv2.resize(
            s['cellpose_segmentation'].astype(np.float32),
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )

        rgba = np.concatenate([img, cp_mask[:, :, None]], axis=-1)
        rgba = rgba.transpose(2, 0, 1)

        mask = cv2.resize(
            s['ground_truth'].astype(np.float32),
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )
        dmap = compute_distance_map(mask)

        images_list.append(rgba)
        dmaps_list.append(dmap[None])
        masks_list.append(mask[None])
        ids.append(s['id'])

    return {
        'image': torch.tensor(np.stack(images_list), dtype=torch.float32),
        'distance_map': torch.tensor(np.stack(dmaps_list), dtype=torch.float32),
        'ground_truth': torch.tensor(np.stack(masks_list), dtype=torch.float32),
        'cellpose_segmentation': torch.tensor(np.stack([s['cellpose_segmentation'] for s in samples]), dtype=torch.float32),
        'id': ids,
    }
