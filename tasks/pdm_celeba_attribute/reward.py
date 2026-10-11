import math

import torch
import torch.nn.functional as F


def select_best_faces(faces, batch_size):
    scores = faces["scores"]
    detected = torch.zeros(batch_size, dtype=torch.bool, device=scores.device)
    if scores.numel() == 0:
        return faces, detected
    image_ids = torch.arange(batch_size, device=scores.device)
    belongs_to_image = image_ids[:, None] == faces["image_ids"].long()[None, :]
    detected = belongs_to_image.any(dim=1)
    image_scores = scores.expand(batch_size, -1).masked_fill(~belongs_to_image, -torch.inf)
    best_indices = image_scores.argmax(dim=1)[detected]
    selected = {
        key: value[best_indices]
        for key, value in faces.items()
        if torch.is_tensor(value) and value.shape[:1] == scores.shape[:1]
    }
    return selected, detected


class Reward:
    def __init__(self, assets, device, config):
        import facer
        from .reward_network import FaceAttribute

        self.device = torch.device(device)
        self.detector = facer.face_detector(
            "retinaface/mobilenet", device=self.device, model_path=assets["detector_checkpoint"]
        ).eval().requires_grad_(False)
        self.attribute = FaceAttribute(
            assets["attribute_checkpoint"], self.device
        ).requires_grad_(False)
        self.labels = tuple(self.attribute.labels)
        self.upscale_size = config.get("upscale_size", 256)
        self.batch_size = config["batch_size"]
        self.temperature = config["temperature"]
        self.epsilon = config.get("epsilon", 1e-6)
        self.no_face_probability = config.get("no_face_probability", 1e-4)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Reward temperature must be positive and finite")
        if self.batch_size < 1 or self.upscale_size < 1 or not 0 < self.epsilon < 0.5 or not 0 < self.no_face_probability < 1:
            raise ValueError("Invalid reward size or probability")
        self.indices = None
        self.calls = 0

    def set_prompt(self, prompt):
        labels = {name: index for index, name in enumerate(self.labels)}
        attributes = prompt["attributes"]
        self.indices = [labels[attr["name"]] for attr in attributes]
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("Target attributes must be nonempty and unique")
        self.targets = torch.tensor(
            [attr["target"] for attr in attributes], device=self.device, dtype=torch.bool
        )

    def probabilities(self, images):
        if self.indices is None:
            raise RuntimeError("Set the attribute prompt before scoring")
        values, detections = [], []
        for batch in images.split(self.batch_size):
            pixels = ((batch.to(self.device).float() + 1) / 2).clamp(0, 1)
            pixels = F.interpolate(pixels, size=(self.upscale_size, self.upscale_size),
                                   mode="bicubic", align_corners=False).clamp(0, 1) * 255
            with torch.no_grad():
                faces, detected = select_best_faces(self.detector(pixels.detach()), len(batch))
            selected = pixels.new_full((len(batch), len(self.indices)), self.no_face_probability)
            if detected.any():
                faces = self.attribute(pixels, faces)
                probabilities = faces["attrs"][:, self.indices]
                probabilities = torch.where(self.targets, probabilities, 1 - probabilities)
                selected = selected.index_copy(0, faces["image_ids"].long(), probabilities)
            # Keep a zero input gradient when no face is detected.
            selected = selected + pixels.flatten(1).sum(-1, keepdim=True) * 0
            values.append(selected.clamp(self.epsilon, 1 - self.epsilon))
            detections.append(detected)
        self.calls += len(images)
        return torch.cat(values), torch.cat(detections)

    def __call__(self, images):
        probabilities, _ = self.probabilities(images)
        return probabilities.log().sum(-1) / self.temperature
