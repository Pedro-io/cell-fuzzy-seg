import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from cellpose import io

from src.utils.logger import logger

from .base_dataset import BaseDataset


class MonusegDataset(BaseDataset):
    def __init__(
      self,
      dataset_name: str,
      config_key: str,
      transform: Optional[Any] = None,
      yaml_path: str = BaseDataset.YAML_CONFIG_PATH
    ) -> None:
        super().__init__(dataset_name, config_key, transform, yaml_path)
        self.image_dir = self.get_image_dir()
        self.mask_dir = self.get_mask_dir()
        self.file_pairs = self._get_file_pairs()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the sample (image and mask) at the specified index."""
        image_path, mask_path = self.file_pairs[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, image_shape=image.shape[:2])

        sample = {
            'id': os.path.splitext(os.path.basename(image_path))[0],
            'image': image,
            'ground_truth': mask,
            'meta': {
                'image_path': image_path,
                'mask_path': mask_path
            }
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from the path using the same reader as Cellpose."""
        logger.debug(f"Loading image: {image_path}")
        image = io.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return image

    def _xml_to_mask(self, xml_path, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for region in root.iter("Region"):
            points = []
            for vertex in region.iter("Vertex"):
                x = float(vertex.attrib["X"])
                y = float(vertex.attrib["Y"])
                points.append([int(x), int(y)])
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask

    def _load_mask(self, mask_path: str, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Loads a mask from disk. Supports .xml, .npy and image-based formats."""
        logger.debug(f"Loading mask: {mask_path}")
        extension = os.path.splitext(mask_path)[1].lower()

        if extension == '.npy':
            mask = np.load(mask_path)
        elif extension == '.xml':
            if image_shape is None:
                stem = os.path.splitext(os.path.basename(mask_path))[0]
                image_path = os.path.join(self.image_dir, stem + self.config['image_extension'])
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Corresponding image not found for {mask_path}: expected {image_path}")
                image_shape = self._load_image(image_path).shape[:2]
            mask = self._xml_to_mask(mask_path, image_shape)
        else:
            mask = io.imread(mask_path)

        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        return mask

    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        """Returns the list of (image, mask) pairs from the configured folders."""
        logger.debug(f"Getting file pairs from: image_dir={self.image_dir}, mask_dir={self.mask_dir}")
        image_ext = self.config['image_extension'].lower()
        mask_ext = self.config['mask_extension'].lower()

        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(image_ext)]
        )

        if not image_files:
            raise FileNotFoundError(
                f"No images found in {self.image_dir} with extension {image_ext}"
            )

        pairs: List[Tuple[str, str]] = []
        for image_name in image_files:
            image_path = os.path.join(self.image_dir, image_name)
            stem = os.path.splitext(image_name)[0]
            mask_name = stem + mask_ext
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask not found for {image_name}: expected {mask_path}"
                )

            pairs.append((image_path, mask_path))

        return pairs
