import torch

from .network import load_planning_diffusion


class Model:
    def __init__(self, assets, condition, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Planning inference requires CUDA")
        settings, normalizer = assets["settings"], assets["normalizer"]
        self.low = torch.tensor(normalizer["mins"], device=self.device)
        self.high = torch.tensor(normalizer["maxs"], device=self.device)
        self.horizon, self.dimension = normalizer["horizon"], len(self.low)
        if not torch.all(self.high > self.low):
            raise ValueError("Normalizer must have positive ranges")
        if settings["include_velocity"] != (self.dimension == 4):
            raise ValueError("Normalizer does not match the checkpoint state representation")
        network, schedule = load_planning_diffusion(assets["checkpoint"], settings, normalizer)
        self.network = network.to(self.device)
        self.schedule = {key: value.to(self.device) for key, value in schedule.items()}
        self.betas = self.schedule["betas"]
        self.steps = settings["n_diffusion_steps"]
        self.predict_epsilon = settings["predict_epsilon"]
        self.set_condition(condition)
        self.reset_counts()

    def set_condition(self, condition):
        endpoints = torch.tensor([condition["starts"], condition["goals"]],
                                 device=self.device, dtype=self.low.dtype)
        if endpoints.ndim != 3 or endpoints.shape[-1] != self.dimension:
            raise ValueError("Endpoints must have shape [2, robots, state dimension]")
        self.physical_endpoints = endpoints
        self.endpoints = self.normalize(endpoints)
        self.robots = endpoints.shape[1]

    def reset_counts(self):
        self.forward_nfe = self.backward_nfe = 0

    @property
    def equivalent_nfe(self):
        return self.forward_nfe + 2 * self.backward_nfe

    def _count_backward(self, gradient):
        self.backward_nfe += len(gradient)
        return gradient

    def normalize(self, states):
        return 2 * (states - self.low) / (self.high - self.low) - 1

    def physical(self, states):
        # Preserve the reference clamp rule independently per joint candidate.
        axes = tuple(range(1, states.ndim))
        outside = ((states.amax(axes, keepdim=True) > 1.0001)
                   | (states.amin(axes, keepdim=True) < -1.0001))
        bounded = torch.where(outside, states.clamp(-1, 1), states)
        return (bounded + 1) * .5 * (self.high - self.low) + self.low

    def condition(self, states):
        states = states.clone()
        states[:, :, 0] = self.endpoints[0]
        states[:, :, -1] = self.endpoints[1]
        return states

    def noise(self, count, generator):
        states = torch.randn(count, self.robots, self.horizon, self.dimension,
                             device=self.device, generator=generator)
        return self.condition(states)

    def prediction(self, states, level):
        # Native levels: 0 is clean; network indices run from 0 to steps - 1.
        if not isinstance(level, int) or not 1 <= level <= self.steps:
            raise ValueError("Prediction requires a native level in [1, steps]")
        flat = states.flatten(0, 1)
        times = torch.full((len(flat),), level - 1, device=self.device, dtype=torch.long)
        prediction = self.network(flat, times)
        self.forward_nfe += len(flat)
        if prediction.requires_grad:
            prediction.register_hook(self._count_backward)
        return prediction.reshape_as(states)

    def _clean(self, states, prediction, level):
        index = level - 1
        if self.predict_epsilon:
            prediction = (self.schedule["sqrt_recip_alphas_cumprod"][index] * states
                          - self.schedule["sqrt_recipm1_alphas_cumprod"][index] * prediction)
        return prediction.clamp(-1, 1)

    def clean_prediction(self, states, level):
        if level == 0:
            return states
        clean = self._clean(states, self.prediction(states, level), level)
        return self.condition(clean)

    def ddpm(self, states, level):
        clean = self._clean(states, self.prediction(states, level), level)
        index = level - 1
        mean = (self.schedule["posterior_mean_coef1"][index] * clean
                + self.schedule["posterior_mean_coef2"][index] * states)
        variance = self.schedule["posterior_log_variance_clipped"][index].exp()
        return mean, variance.expand(len(states), self.robots, 1, 1)
