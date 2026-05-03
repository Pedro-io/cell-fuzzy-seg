import numpy as np
from cellpose import models, core
from .base_step import PipelineStep
from src.utils.logger import logger


class CellposeStep(PipelineStep):
    def __init__(self, batch_size: int = 10):
        if not core.use_gpu():
            raise RuntimeError("GPU is required but not available.")

        self.model = models.CellposeModel(gpu=True)
        self.batch_size = batch_size

        logger.info("[CellposeStep] Running on GPU")

    def forward(self, data: dict) -> dict:
        if "image" not in data:
            raise KeyError("Input data must contain 'image'")

        image = data["image"]

        #Ensure 3 chanels for Cellpose
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        #uint8 (0–255)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.max() > 0:
                image = image / image.max() * 255
            image = image.astype(np.uint8)

        #Run Cellpose
        masks, flows, styles = self.model.eval(
            image,
            batch_size=self.batch_size
        )

        data["segmentation"] = masks
        data["flows"] = flows
        data["styles"] = styles

        return data