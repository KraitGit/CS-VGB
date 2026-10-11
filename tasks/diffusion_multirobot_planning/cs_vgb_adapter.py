import math

import torch


class Adapter:
    def __init__(self, model, reward, metric, noise_std_scale=1., guidance=None):
        if not math.isfinite(noise_std_scale) or noise_std_scale < 0:
            raise ValueError("noise_std_scale must be finite and nonnegative")
        self.model, self.reward, self.metric = model, reward, metric
        self.noise_std_scale = noise_std_scale
        self.guidance = guidance
        if guidance is not None and not 0 <= guidance["start_fraction"] <= 1:
            raise ValueError("Guidance start_fraction must be in [0, 1]")
        self.alpha_bar = torch.cat((model.betas.new_ones(1), (1 - model.betas).cumprod(0)))
        self.geometry_gradient_samples = 0
        self.reset_archive()

    @property
    def equivalent_nfe(self):
        return self.model.equivalent_nfe

    @torch.no_grad()
    def initial(self, count, level, generator, guided=False):
        # Denoise to the starting search checkpoint.
        if not isinstance(count, int) or count < 1:
            raise ValueError("Particle count must be a positive integer")
        if not isinstance(level, int) or not 1 <= level <= self.model.steps:
            raise ValueError("Initial level must be an integer in [1, model.steps]")
        states = self.model.noise(count, generator)
        if level < self.model.steps:
            states = self.forward(states, self.model.steps, level, generator, guided=guided)
        return states

    def reset_archive(self):
        self.best_candidate = None
        self.best_rank = None
        self.best_log_reward = None
        self.first_feasible_nfe = None

    @torch.no_grad()
    def _record(self, states, cost, log_rewards):
        # Planning archive selects clean candidates by feasibility-first rank, not log reward.
        metrics = self.metric.score(states)
        ranks = self.metric.archive_rank(cost, metrics)
        index = ranks.argmax().reshape(1)
        candidate, rank, log_reward = states[index], ranks[index], log_rewards[index]
        if self.best_candidate is not None:
            candidates = torch.cat((self.best_candidate, candidate))
            ranks = torch.cat((self.best_rank, rank))
            rewards = torch.cat((self.best_log_reward, log_reward))
            index = ranks.argmax().reshape(1)
            candidate, rank, log_reward = candidates[index], ranks[index], rewards[index]
        self.best_candidate = candidate.detach()
        self.best_rank = rank.detach()
        self.best_log_reward = log_reward.detach()
        if self.first_feasible_nfe is None and bool(metrics["success"].any()):
            self.first_feasible_nfe = self.equivalent_nfe

    @torch.no_grad()
    def log_value(self, states, level):
        if not isinstance(level, int) or not 0 <= level <= self.model.steps:
            raise ValueError("Value estimation requires an integer level in [0, model.steps]")
        if level > 0:
            # One clean prediction, not a full DDPM rollout or posterior average.
            return self.reward(self.model.clean_prediction(states, level))
        cost = self.reward.cost(states)
        log_rewards = -cost / self.reward.config["temperature"]
        self._record(states, cost, log_rewards)
        return log_rewards

    @torch.no_grad()
    def guide(self, states, steps, scale, max_grad_norm):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Guidance steps must be a nonnegative integer")
        if any(not math.isfinite(value) or value < 0 for value in (scale, max_grad_norm)):
            raise ValueError("Guidance scale and max_grad_norm must be finite and nonnegative")
        if scale == 0 or max_grad_norm == 0:
            return states
        for _ in range(steps):
            # Reference guidance differentiates physical costs, not the denoiser or log reward.
            with torch.enable_grad():
                physical = self.model.physical(states).detach().requires_grad_(True)
                costs = self.reward.physical_costs(physical)
                gradients = [torch.autograd.grad(cost.sum(), physical,
                             retain_graph=n < len(costs) - 1)[0]
                             for n, cost in enumerate(costs.values())]
            combined = torch.zeros_like(states)
            for name, gradient in zip(costs, gradients):
                norm = (gradient + 1e-6).norm(dim=-1, keepdim=True)
                gradient = gradient * norm.clamp(max=max_grad_norm) / norm
                gradient[:, :, 0] = 0
                gradient[:, :, -1] = 0
                combined.add_(gradient, alpha=self.reward.config["weights"][name])
            # Preserve the reference physical-gradient update in normalized coordinates.
            states = self.model.condition(states - scale * combined)
            self.geometry_gradient_samples += len(states)
        return states

    @torch.no_grad()
    def forward(self, states, t, s, generator, guided=False):
        # Native DDPM levels: 0 is clean, model.steps is noise.
        if not isinstance(t, int) or not isinstance(s, int) or not 0 <= s < t <= self.model.steps:
            raise ValueError("Denoising requires integer levels 0 <= s < t <= model.steps")
        if guided and self.guidance is None:
            raise ValueError("Guided denoising requires guidance settings")
        threshold = math.ceil(self.guidance["start_fraction"] * self.model.steps) if guided else 0
        for level in range(t, s, -1):
            mean, variance = self.model.ddpm(states, level)
            if level > 1:
                noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
                mean = mean + self.noise_std_scale * variance.sqrt() * noise
            states = self.model.condition(mean)
            if guided and level - 1 < threshold:
                for stage in self.guidance["stages"]:
                    states = self.guide(states, stage["steps"], stage["scale"],
                                        self.guidance["max_grad_norm"])
        return states

    @torch.no_grad()
    def propose_forward(self, states, t, s, generator, count=1, guided=False):
        # count is the number of new children per slot, excluding retained entries.
        if not isinstance(count, int) or count < 1:
            raise ValueError("Proposal count must be a positive integer")
        parents = torch.arange(len(states), device=states.device).repeat_interleave(count)
        return self.forward(states[parents], t, s, generator, guided=guided), parents

    @torch.no_grad()
    def guided_forward(self, states, t, s, generator, enabled=True, count=1):
        # Planning uses stage scales; the direction setting is an on/off switch.
        if not isinstance(enabled, bool):
            raise ValueError("Planning guidance_forward must be boolean")
        return self.propose_forward(states, t, s, generator, count, guided=enabled)

    @torch.no_grad()
    def guided_renoise(self, states, t, s, generator, enabled=False, count=1):
        if not isinstance(enabled, bool):
            raise ValueError("Planning guidance_backward must be boolean")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Proposal count must be a positive integer")
        if enabled and self.guidance is None:
            raise ValueError("Guided renoising requires guidance settings")
        parents = torch.arange(len(states), device=states.device).repeat_interleave(count)
        samples = self.renoise(states[parents], t, s, generator)
        if enabled:
            # Reference backward guidance runs once after the complete renoise jump.
            for stage in self.guidance["stages"]:
                samples = self.guide(samples, stage["steps"], stage["scale"],
                                     self.guidance["max_grad_norm"])
        return samples, parents

    @torch.no_grad()
    def renoise(self, states, t, s, generator):
        if not isinstance(t, int) or not isinstance(s, int) or not 0 <= t < s <= self.model.steps:
            raise ValueError("Renoising requires integer levels 0 <= t < s <= model.steps")
        ratio = self.alpha_bar[s] / self.alpha_bar[t]
        mean = self.model.condition(ratio.sqrt() * states)
        noise = torch.randn(states.shape, device=states.device, dtype=states.dtype, generator=generator)
        # VP renoising does not use the DDPM noise_std_scale.
        return self.model.condition(mean + (1 - ratio).sqrt() * noise)
