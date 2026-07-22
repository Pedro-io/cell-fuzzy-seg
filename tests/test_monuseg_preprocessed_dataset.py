from typing import Any, Dict, List

from src.data.load.base_dataset import BaseDataset
from src.data.load.monuseg_preprocessed_dataset import MonusegPreprocessedDataset


class DummyDataset(BaseDataset):
    def __init__(self):
        self.samples = [
            {"id": "img1", "image": "img-1", "ground_truth": "mask-1", "meta": {"source": "dummy"}},
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def _load_image(self, image_path: str) -> Any:
        return image_path

    def _load_mask(self, mask_path: str) -> Any:
        return mask_path

    def _get_file_pairs(self) -> List[tuple]:
        return []


class DummyPipeline:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["processed"] = True
        return data


def test_preprocessed_dataset_runs_pipeline_and_preserves_base_fields():
    base_dataset = DummyDataset()
    dataset = MonusegPreprocessedDataset(base_dataset, DummyPipeline())

    sample = dataset[0]

    assert sample["id"] == "img1"
    assert sample["image"] == "img-1"
    assert sample["ground_truth"] == "mask-1"
    assert sample["processed"] is True
