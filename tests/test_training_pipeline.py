from src.pipeline.steps.base_step import PipelineStep
from src.pipeline.training_pipeline import TrainingPipeline


class AddKeyStep(PipelineStep):
    """Step dummy que adiciona uma chave fixa ao dicionário de dados."""

    def __init__(self, key: str, value, name: str = "AddKeyStep"):
        super().__init__(name=name)
        self.key = key
        self.value = value

    def forward(self, data):
        data[self.key] = self.value
        return data


def test_training_pipeline_runs_steps_in_order():
    pipeline = TrainingPipeline([
        AddKeyStep("markers", "markers-value"),
        AddKeyStep("segmentation", "seg-value"),
    ])

    data = pipeline.run({"image": "img"}, verbose=False)

    assert data == {"image": "img", "markers": "markers-value", "segmentation": "seg-value"}


def test_training_pipeline_rejects_non_dict():
    pipeline = TrainingPipeline([])

    try:
        pipeline.run("not-a-dict")
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-dict input")


def test_training_pipeline_call_alias():
    pipeline = TrainingPipeline([AddKeyStep("segmentation", 42)])

    data = pipeline({"image": "img"})

    assert data["segmentation"] == 42


def test_training_pipeline_exposes_steps():
    steps = [AddKeyStep("markers", 1)]
    pipeline = TrainingPipeline(steps)

    assert pipeline.steps is steps
