import torch
import segmentation_models_pytorch as smp
from .base_networks import BaseNetwork


class MarkerNet(BaseNetwork):
    def __init__(self, config: dict):
        encoder_name = config.get("encoder_name", "resnet34")
        pretrained = config.get("pretrained", True)
        in_channels = config.get("in_channels", 4)
        self.threshold = config.get("threshold", 0.5)
        self._config = config

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return (probs >= self.threshold).float()

    def train_step(self, batch, optimizer, loss_fn):
        self.model.train()
        images = batch["image"]
        distance_maps = batch["distance_map"]
        gt_masks = batch["ground_truth"]

        optimizer.zero_grad()
        markers = self.forward(images)
        loss, loss_log = loss_fn(markers, distance_maps, gt_masks)
        loss.backward()
        optimizer.step()
        return loss.item(), loss_log

    def validation_step(self, batch, loss_fn):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"]
            distance_maps = batch["distance_map"]
            gt_masks = batch["ground_truth"]
            markers = self.forward(images)
            loss, loss_log = loss_fn(markers, distance_maps, gt_masks)
        return loss.item(), loss_log

    def evaluate(self, data_loader, metrics):
        self.model.eval()
        results = {m.__class__.__name__: 0.0 for m in metrics}
        n = 0
        with torch.no_grad():
            for batch in data_loader:
                preds = self.predict(batch["image"])
                targets = batch["ground_truth"]
                for metric in metrics:
                    results[metric.__class__.__name__] += metric(preds, targets).item()
                n += 1
        return {k: v / n for k, v in results.items()}

    def save(self, path):
        torch.save({"model_state_dict": self.model.state_dict(), "config": self._config}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_config(self):
        return self._config
