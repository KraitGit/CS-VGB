import math

import torch


class Adapter:
    def __init__(self, model, reward=None):
        self.model = model
        self.reward = reward
        self.reset_archive()

    @property
    def equivalent_nfe(self):
        return self.model.equivalent_nfe

    def reset_archive(self):
        """Clear evaluated endpoints before starting a new reward condition."""
        self.best_endpoint = None
        self.best_log_reward = None

    @torch.no_grad()
    def _record(self, clean, log_rewards):
        """Keep the best evaluated clean endpoint without retaining the batch."""
        scores = log_rewards.detach().reshape(-1)
        best = scores.argmax().reshape(1)
        sample, score = clean.detach()[best], scores[best]
        if self.best_endpoint is not None:
            samples = torch.cat((self.best_endpoint, sample))
            scores = torch.cat((self.best_log_reward, score))
            best = scores.argmax().reshape(1)
            sample, score = samples[best], scores[best]
        self.best_endpoint, self.best_log_reward = sample, score

    @torch.no_grad()
    def log_value(self, states, t, generator, mc_samples=1):
        if not 0 <= t <= 1 or not isinstance(mc_samples, int) or mc_samples < 1:
            raise ValueError("Value estimation requires 0 <= t <= 1 and positive integer mc_samples")
        if t == 0:
            log_rewards = self.reward(states)
            self._record(states, log_rewards)
            return log_rewards

        clean = self.model.posterior(states.repeat_interleave(mc_samples, 0), t, generator)
        log_rewards = self.reward(clean).reshape(len(states), mc_samples)
        self._record(clean, log_rewards)
        return torch.logsumexp(log_rewards, dim=1) - math.log(mc_samples)

    @torch.enable_grad()
    def posterior_gradient(self, states, t, generator, guidance_samples=8):
        """Return endpoints [N,m_g,...], gradients [N,...], and log rewards [N,m_g]."""
        if not 0 <= t <= 1 or not isinstance(guidance_samples, int) or guidance_samples < 1:
            raise ValueError("Guidance requires 0 <= t <= 1 and positive integer guidance_samples")
        source = states.detach().requires_grad_(True)
        count = guidance_samples if t > 0 else 1
        clean = (self.model.posterior(source.repeat_interleave(count, 0), t, generator)
                 if t > 0 else source)
        log_rewards = self.reward(clean).reshape(len(states), count)
        log_value = torch.logsumexp(log_rewards, dim=1) - math.log(count)
        gradient, = torch.autograd.grad(log_value.sum(), source)
        self._record(clean, log_rewards)
        endpoints = clean.detach().reshape(len(states), count, *states.shape[1:])
        return endpoints, gradient.detach(), log_rewards.detach()

    @torch.no_grad()
    def select_endpoint(self, states, t, generator, endpoint_samples=8):
        if not 0 < t <= 1 or not isinstance(endpoint_samples, int) or endpoint_samples < 1:
            raise ValueError("Endpoint selection requires 0 < t <= 1 and positive integer endpoint_samples")

        clean = self.model.posterior(states.repeat_interleave(endpoint_samples, 0), t, generator)
        log_rewards = self.reward(clean)
        self._record(clean, log_rewards)
        best = log_rewards.argmax().reshape(1)
        parent = best // endpoint_samples
        return clean[best], parent, log_rewards[best]

    @torch.no_grad()
    def forward(self, states, t, s, generator):
        # Paper time: 0 is clean, 1 is noise.
        if not 0 <= s < t <= 1:
            raise ValueError("Denoising requires 0 <= s < t <= 1")

        clean = self.model.posterior(states, t, generator)
        return self._bridge(states, clean, t, s, generator)

    @torch.no_grad()
    def renoise(self, states, t, s, generator):
        if not 0 <= t < s <= 1:
            raise ValueError("Renoising requires 0 <= t < s <= 1")

        a = (1 - s) / (1 - t)
        variance = s * s - a * a * t * t
        noise = torch.randn(
            states.shape, device=states.device, dtype=states.dtype, generator=generator
        )
        return a * states + variance**0.5 * noise


    @torch.no_grad()
    def propose_forward(self, states, t, s, generator, count=1, endpoint_samples=8):
        # count is the number of new children per slot (l_F), excluding retained entries.
        if not 0 <= s < t <= 1:
            raise ValueError("Denoising requires 0 <= s < t <= 1")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Proposal count must be a positive integer")

        clean, parent, _ = self.select_endpoint(states, t, generator, endpoint_samples)
        # Select globally over states; flatten [slots, count, ...] for the bridge.
        parents = parent.expand(len(states) * count)
        condition = states.index_select(0, parents)
        clean = clean.expand_as(condition)
        return self._bridge(condition, clean, t, s, generator), parents

    @torch.no_grad()
    def guided_forward(self, states, t, s, generator, strength, count=1, guidance_samples=8,
                       endpoint_mode="reuse"):
        """Per-parent bridge proposals with the F.1 forward mean shift."""
        if not 0 <= s < t <= 1:
            raise ValueError("Denoising requires 0 <= s < t <= 1")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Proposal count must be a positive integer")
        if endpoint_mode not in ("reuse", "fresh", "best"):
            raise ValueError("endpoint_mode must be reuse, fresh, or best")
        parents = torch.arange(len(states), device=states.device).repeat_interleave(count)
        condition = states[parents]
        if strength == 0:
            return self.forward(condition, t, s, generator), parents

        endpoints, gradient, log_rewards = self.posterior_gradient(
            states, t, generator, guidance_samples)

        if endpoint_mode == "fresh":
            clean = self.model.posterior(condition, t, generator)
        elif endpoint_mode == "reuse":
            clean = endpoints[:, 0][parents]
        else:
            best = log_rewards.argmax(dim=1)
            rows = torch.arange(len(states), device=states.device)
            clean = endpoints[rows, best][parents]

        velocity_norm = ((endpoints.mean(1) - states) / t).flatten(1).norm(dim=1)
        gradient_norm = gradient.flatten(1).norm(dim=1)
        denominator = torch.where(gradient_norm > 0, gradient_norm, torch.ones_like(gradient_norm))
        shape = (-1,) + (1,) * (states.ndim - 1)
        shift = strength * (t - s) * gradient * (velocity_norm / denominator).reshape(shape)
        return self._bridge(condition, clean, t, s, generator) + shift[parents], parents

    @torch.no_grad()
    def guided_renoise(self, states, t, s, generator, strength, count=1, guidance_samples=8):
        """Per-parent renoising proposals with the F.1 backward mean shift."""
        if not 0 <= t < s <= 1:
            raise ValueError("Renoising requires 0 <= t < s <= 1")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Proposal count must be a positive integer")
        parents = torch.arange(len(states), device=states.device).repeat_interleave(count)
        if strength == 0:
            return self.renoise(states[parents], t, s, generator), parents

        _, gradient, _ = self.posterior_gradient(states, t, generator, guidance_samples)
        a = (1 - s) / (1 - t)
        variance = s * s - a * a * t * t
        rms = gradient.square().flatten(1).mean(dim=1).sqrt()
        denominator = torch.where(rms > 0, rms, torch.ones_like(rms))
        shape = (-1,) + (1,) * (states.ndim - 1)
        shift = strength * variance**0.5 * gradient / denominator.reshape(shape)
        return self.renoise(states[parents], t, s, generator) + shift[parents], parents

    def _bridge(self, states, clean, t, s, generator):
        if s == 0:
            return clean

        # Gaussian bridge conditioned on the current state and clean endpoint.
        a = (1 - t) / (1 - s)
        sigma2 = t * t - a * a * s * s
        mean = a * s * s / (t * t) * states + (1 - s) * sigma2 / (t * t) * clean
        variance = s * s * sigma2 / (t * t)
        noise = torch.randn(
            states.shape, device=states.device, dtype=states.dtype, generator=generator
        )
        return mean + variance**0.5 * noise
