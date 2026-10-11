import math

import torch
from torch.utils.checkpoint import checkpoint

from .data import load_calibration
from .reward_network import HPSv2


class Reward:
    def __init__(self, assets, model, config):
        self.model = model
        self.batch_size = config["batch_size"]
        self.temperature = config["temperature"]
        self.checkpoint_grad = config["checkpoint_grad"]
        if self.batch_size < 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Use positive batch_size and finite positive temperature")
        self.calibration = load_calibration(assets["calibration"])
        self.scorer = HPSv2(assets["reward_checkpoint"], model.device)
        self.prompt = None
        self.calls = 0

    def set_prompt(self, condition):
        prompt = condition["prompt"]
        row = self.calibration[prompt]
        self.mean, self.std = row["mean"]["hpsv2"], row["std"]["hpsv2"]
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("Calibration requires finite mean and positive finite std")
        self.scorer.set_prompt(prompt)
        self.prompt = prompt

    def _score_batch(self, states):
        return self.scorer(self.model.decode(states))

    def scores(self, states):
        if self.prompt is None:
            raise RuntimeError("Set the prompt before scoring")
        values = []
        recompute = (self.checkpoint_grad and torch.is_grad_enabled()
                     and states.requires_grad and len(states) > self.batch_size)
        for batch in states.split(self.batch_size):
            value = (checkpoint(self._score_batch, batch, use_reentrant=False)
                     if recompute else self._score_batch(batch))
            values.append(value)
        self.calls += len(states)
        return torch.cat(values)

    def __call__(self, states):
        return (self.scores(states) - self.mean) / (self.std * self.temperature)
