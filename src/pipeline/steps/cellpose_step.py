from typing import Any, Dict

from cellpose import core, models

from src.utils.logger import logger

from .base_step import PipelineStep


class CellposeStep(PipelineStep):
    """Pipeline step that applies Cellpose segmentation to an input image.

    This step initializes a single CellposeModel on the GPU and reuses it for
    all subsequent calls to ``forward``.

    Attributes:
        model: Loaded ``CellposeModel`` instance using GPU inference.
        batch_size: Number of images processed per inference batch.
        diam_mean: Estimated object diameter passed to Cellpose.
        cellprob_threshold: Cell probability threshold for mask selection.
        flow_threshold: Flow threshold for Cellpose tracking.
        min_size: Minimum object size to keep in the final mask.
    """

    def __init__(
        self, batch_size: int = 8,
        name: str = "CellposeStep",
        pretreined_model: str = "cpsam_v2",
        diam_mean: float = 30.0,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.2,
        min_size: int = 4
        ) -> None:
        """Create a CellposeStep and verify GPU availability.

        Args:
            batch_size: Number of images per inference batch.
            name: Identifier for this step in the pipeline.
            pretreined_model: Name of the pretrained Cellpose model to load.
            diam_mean: Mean diameter of cells for segmentation.
            cellprob_threshold: Threshold applied to Cellpose cell probability.
            flow_threshold: Threshold applied to Cellpose flow outputs.
            min_size: Minimum instance size to keep in the segmentation mask.

        Raises:
            RuntimeError: If a CUDA-capable GPU is not available.
        """
        super().__init__(name=name)
        if not core.use_gpu():
            raise RuntimeError("GPU is required but not available.")

        self.model = models.CellposeModel(gpu=True, pretrained_model=pretreined_model, diam_mean=diam_mean)
        self.batch_size = batch_size
        self.diam_mean = diam_mean
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.min_size = min_size
        logger.info("[CellposeStep] Running on GPU")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Cellpose segmentation and attach results to the input data.

        Args:
            data: Dictionary containing at least the ``"image"`` key with a
                NumPy array of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            The same ``data`` dictionary with added keys:
                - ``"segmentation"``: instance mask array of shape ``(H, W)``.
                - ``"flows"``: Cellpose flow field outputs.
                - ``"styles"``: Cellpose style vectors.

        Raises:
            KeyError: If the ``"image"`` key is missing from ``data``.
        """
        if "image" not in data:
            raise KeyError("Input data must contain 'image'")

        image = data["image"]

        masks, flows, styles = self.model.eval(
            image,
            batch_size=self.batch_size,
            diameter=self.diam_mean,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            min_size=self.min_size,
        )

        data["segmentation"] = masks
        data["flows"] = flows
        data["styles"] = styles

        return data
