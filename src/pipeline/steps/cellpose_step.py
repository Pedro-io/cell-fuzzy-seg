import numpy as np
from cellpose import models, core
from .base_step import PipelineStep
from src.utils.logger import logger


class CellposeStep(PipelineStep):
    """Pipeline step that runs Cellpose cell segmentation on an input image.

    Requires a CUDA-capable GPU. Initializes a CellposeModel and stores it
    for repeated use across multiple forward calls.

    Attributes:
        model: Loaded CellposeModel instance running on GPU.
        batch_size: Number of images processed per Cellpose batch.
    """

    def __init__(self, batch_size: int = 10, name: str = "CellposeStep"):
        """Initializes CellposeStep and verifies GPU availability.

        Args:
            batch_size: Number of images per inference batch.
            name: Identifier for this step in the pipeline.

        Raises:
            RuntimeError: If no GPU is detected by Cellpose.
        """
        super().__init__(name=name)
        if not core.use_gpu():
            raise RuntimeError("GPU is required but not available.")

        self.model = models.CellposeModel(gpu=True)
        self.batch_size = batch_size

        logger.info("[CellposeStep] Running on GPU")

    def forward(self, data: dict) -> dict:
        """Runs Cellpose segmentation and stores results back into the data dict.

        Expects a single image array. Converts grayscale inputs to 3-channel and
        normalises any non-uint8 array to the [0, 255] range before inference.

        Args:
            data: Dictionary containing at least the key ``"image"`` with a
                NumPy array of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            The same ``data`` dictionary, extended with:
                - ``"segmentation"``: integer mask array of shape ``(H, W)``.
                - ``"flows"``: Cellpose flow outputs.
                - ``"styles"``: Cellpose style vectors.

        Raises:
            KeyError: If ``"image"`` is absent from ``data``.
        """
        if "image" not in data:
            raise KeyError("Input data must contain 'image'")

        image = data["image"]

        # Cellpose expects a 3-channel image; duplicate the single channel.
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        # Normalise to uint8 [0, 255] required by Cellpose.
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.max() > 0:
                image = image / image.max() * 255
            image = image.astype(np.uint8)

        masks, flows, styles = self.model.eval(
            image,
            batch_size=self.batch_size
        )

        data["segmentation"] = masks
        data["flows"] = flows
        data["styles"] = styles

        return data