import math

import torch

from .pools import Pool, select_local
from .weights import backward_log_weights, forward_log_weights, population_action


class Sampler:
    """Guided or unguided search population, initialized at the highest checkpoint."""

    @torch.no_grad()
    def __init__(self, adapter, states, times, generator, forward_size=2,
                 backward_size=2, depth_alpha=.3, persistence=1., lazy=True,
                 skip_stays=True, endpoint_samples=8, value_samples=1,
                 guided=False, guidance_forward=.1, guidance_backward=0.,
                 guidance_samples=8, endpoint_mode="fresh", local_child_argmax=None):
        self.times = tuple(times)
        if len(self.times) < 2 or self.times[0] != 0:
            raise ValueError("The checkpoint grid must start at 0 and contain at least two levels")
        if any(not math.isfinite(t) for t in self.times) or any(s >= t for s, t in zip(self.times, self.times[1:])):
            raise ValueError("Checkpoint times must increase strictly")
        self.adapter = adapter
        self.generator = generator
        self.sizes = {"F": forward_size, "B": backward_size}
        self.depth_alpha = depth_alpha
        self.persistence = persistence
        self.lazy = lazy
        self.skip_stays = skip_stays
        self.endpoint_samples = endpoint_samples
        self.value_samples = value_samples
        self.guided = guided
        self.guidance_forward = guidance_forward
        self.guidance_backward = guidance_backward
        self.guidance_samples = guidance_samples
        self.endpoint_mode = endpoint_mode
        self.local_child_argmax = not guided if local_child_argmax is None else local_child_argmax
        self.states = states.detach()
        self.level = len(self.times) - 1
        self.direction = "F"
        self.forced_renoise_level = 0
        self.log_values = self._log_value(states, self.times[-1]).detach()
        self.pools = {"F": Pool(forward_size)}

    def _log_value(self, states, t):
        if self.value_samples is None:
            return self.adapter.log_value(states, t)
        return self.adapter.log_value(states, t, self.generator, self.value_samples)

    def _propose_forward(self, states, t, s, count):
        if self.endpoint_samples is None:
            return self.adapter.propose_forward(states, t, s, self.generator, count)
        return self.adapter.propose_forward(states, t, s, self.generator, count, self.endpoint_samples)

    def _guided_proposal(self, states, t, s, count, direction):
        method = self.adapter.guided_forward if direction == "F" else self.adapter.guided_renoise
        strength = self.guidance_forward if direction == "F" else self.guidance_backward
        if self.guidance_samples is None:
            return method(states, t, s, self.generator, strength, count)
        options = {"endpoint_mode": self.endpoint_mode} if direction == "F" else {}
        return method(states, t, s, self.generator, strength, count, self.guidance_samples, **options)

    @torch.no_grad()
    def _fill(self, direction):
        pool = self.pools[direction]
        if pool.samples is not None:
            return
        count = pool.remaining
        n = len(self.states)
        destination = self.level + (-1 if direction == "F" else 1)
        t, s = self.times[self.level], self.times[destination]
        if count == 0:
            samples = self.states.new_empty((n, 0, *self.states.shape[1:]))
            values = self.log_values.new_empty((n, 0))
            parents = torch.empty((n, 0), dtype=torch.long, device=self.states.device)
        else:
            if self.guided:
                samples, parents = self._guided_proposal(self.states, t, s, count, direction)
            elif direction == "F":
                samples, parents = self._propose_forward(self.states, t, s, count)
            else:
                parents = torch.arange(n, device=self.states.device).repeat_interleave(count)
                samples = self.adapter.renoise(self.states[parents], t, s, self.generator)
            values = self._log_value(samples, s)
            samples = samples.reshape(n, count, *self.states.shape[1:])
            values, parents = values.reshape(n, count), parents.reshape(n, count)
        pool.fill(samples, values, parents)

    def _weights(self, direction, marked=False):
        pool = self.pools[direction]
        if marked:
            values = pool.marked_values()
            parents = (torch.arange(len(self.states), device=self.states.device)
                       if pool.parents is None else
                       pool.parents.gather(1, pool.mark[:, None]).squeeze(1))
        else:
            values, parents = pool.log_values, pool.parents
        weights = forward_log_weights if direction == "F" else backward_log_weights
        return weights(values, self.log_values[parents], self.level, self.depth_alpha)

    @torch.no_grad()
    def step(self, force_backward=False):
        """Apply one population action; return move, stay, or flip."""
        if force_backward:
            self.direction = "B"
        direction = self.direction
        opposite = "B" if direction == "F" else "F"
        self._fill(direction)
        active = self._weights(direction)
        if force_backward or self.level in (0, len(self.times) - 1):
            action, indices = "move", None
        else:
            if not self.lazy:
                self._fill(opposite)
            action, indices = population_action(
                active, self._weights(opposite, marked=self.lazy), self.generator,
                self.persistence, self.lazy, self.skip_stays)
        if action == "flip":
            if self.lazy:
                self.pools[direction].mark = indices
                self._fill(opposite)
            self.direction = opposite
        elif action == "move":
            pool = self.pools[direction]
            states, values, parents, _ = select_local(
                pool.samples, pool.log_values, active, pool.parents,
                argmax=self.local_child_argmax, generator=self.generator)
            reverse = Pool(self.sizes[opposite], self.states[parents],
                           self.log_values[parents], self.generator)
            self.level += -1 if direction == "F" else 1
            self.pools = {opposite: reverse}
            if self.level in (0, len(self.times) - 1):
                self.direction = "B" if self.level == 0 else "F"
            else:
                self.pools[direction] = Pool(self.sizes[direction])
            self.states, self.log_values = states, values
        return action

    @torch.no_grad()
    def _warp(self, level):
        """Renoise from clean and restart with fresh pools."""
        self.states = self.adapter.renoise(
            self.states, self.times[0], self.times[level], self.generator).detach()
        self.level, self.direction = level, "F"
        self.log_values = self._log_value(self.states, self.times[level]).detach()
        self.pools = {"F": Pool(self.sizes["F"])}
        if level < len(self.times) - 1:
            if self.lazy:
                # A new reverse mark comes from the adjacent checkpoint.
                marked = self.adapter.renoise(
                    self.states, self.times[level], self.times[level + 1], self.generator)
                values = self._log_value(marked, self.times[level + 1])
                self.pools["B"] = Pool(self.sizes["B"], marked, values, self.generator)
            else:
                self.pools["B"] = Pool(self.sizes["B"])

    @torch.no_grad()
    def run(self, num_updates, fallback_f_only=False, nfe_budget=None, warp_beta=0.,
            direct_warp=True, progress=None, stop_on_feasible=False):
        """Check cumulative NFE before each update; fallback may add compute."""
        if not isinstance(num_updates, int) or num_updates < 0:
            raise ValueError("num_updates must be a nonnegative integer")
        if not 0 <= warp_beta <= 1:
            raise ValueError("warp_beta must be in [0, 1]")
        if stop_on_feasible and not hasattr(self.adapter, "first_feasible_nfe"):
            raise ValueError("Feasibility stopping requires adapter.first_feasible_nfe")
        warp_level = math.ceil(warp_beta * (len(self.times) - 1))
        warps = 0
        forced_renoise_steps = 0
        if direct_warp or not warp_level:
            self.forced_renoise_level = 0
        if nfe_budget is not None:
            if not isinstance(nfe_budget, int) or nfe_budget < 0:
                raise ValueError("nfe_budget must be a nonnegative integer or None")
            if getattr(self.adapter, "equivalent_nfe", None) is None:
                raise ValueError("NFE budgets require adapter.equivalent_nfe")
        archive, rewards = [], []
        counts = {"move": 0, "stay": 0, "flip": 0}
        stop_reason = "num_updates"
        if self.level == 0:
            archive.append(self.states)
            rewards.append(self.log_values)
        for _ in range(num_updates):
            if stop_on_feasible and self.adapter.first_feasible_nfe is not None:
                break
            if nfe_budget is not None and self.adapter.equivalent_nfe >= nfe_budget:
                stop_reason = "nfe_budget"
                break
            if self.level == 0 and warp_level and direct_warp:
                self._warp(warp_level)
                warps += 1
                if progress is not None:
                    progress("warp")
                if nfe_budget is not None and self.adapter.equivalent_nfe >= nfe_budget:
                    stop_reason = "nfe_budget"
                    break
            elif self.level == 0 and warp_level:
                self.forced_renoise_level = warp_level
            force_backward = self.level < self.forced_renoise_level
            action = self.step(force_backward=force_backward)
            forced_renoise_steps += int(force_backward)
            if self.level >= self.forced_renoise_level:
                self.forced_renoise_level = 0
            counts[action] += 1
            if action == "move" and self.level == 0:
                archive.append(self.states)
                rewards.append(self.log_values)
            if progress is not None:
                progress(action)

        before_fallback_nfe = getattr(self.adapter, "equivalent_nfe", None)
        fallback_steps = 0
        feasible_stop = stop_on_feasible and self.adapter.first_feasible_nfe is not None
        if feasible_stop:
            stop_reason = "feasible_candidate"
        if fallback_f_only and not archive and not feasible_stop:
            # Complete missing output with one fresh forward child per particle.
            fallback_steps = self.level
            while self.level > 0:
                t, s = self.times[self.level], self.times[self.level - 1]
                if self.guided:
                    self.states, _ = self._guided_proposal(self.states, t, s, 1, "F")
                else:
                    self.states, _ = self._propose_forward(self.states, t, s, 1)
                self.level -= 1
                if progress is not None:
                    progress("fallback")
            self.states = self.states.detach()
            self.log_values = self._log_value(self.states, self.times[0]).detach()
            self.direction = "B"
            self.forced_renoise_level = 0
            self.pools = {"B": Pool(self.sizes["B"])}
            archive.append(self.states)
            rewards.append(self.log_values)

        clean_visits = len(archive)
        samples = (torch.cat(archive) if archive else
                   self.states.new_empty((0, *self.states.shape[1:])))
        log_rewards = torch.cat(rewards) if rewards else self.log_values.new_empty(0)
        best = best_log_reward = None
        if archive:
            index = log_rewards.argmax().reshape(1)
            best, best_log_reward = samples[index], log_rewards[index]
        # Endpoint tracking spans this adapter's current reward condition.
        endpoint = getattr(self.adapter, "best_endpoint", None)
        if endpoint is not None:
            endpoint_reward = self.adapter.best_log_reward
            if best is None:
                best, best_log_reward = endpoint, endpoint_reward
            else:
                candidates = torch.cat((best, endpoint))
                scores = torch.cat((best_log_reward, endpoint_reward))
                index = scores.argmax().reshape(1)
                best, best_log_reward = candidates[index], scores[index]
        # Task archives may prioritize feasibility over log reward (for planning).
        candidate = getattr(self.adapter, "best_candidate", None)
        if candidate is not None:
            best, best_log_reward = candidate, self.adapter.best_log_reward
        total_nfe = getattr(self.adapter, "equivalent_nfe", None)
        return {
            "best": best,
            "best_log_reward": best_log_reward,
            "archive": samples,
            "log_rewards": log_rewards,
            "clean_visits": clean_visits,
            "counts": counts,
            "fallback_steps": fallback_steps,
            "warps": warps,
            "warp_level": warp_level,
            "direct_warp": direct_warp,
            "forced_renoise_steps": forced_renoise_steps,
            "updates": sum(counts.values()),
            "stop_reason": stop_reason,
            "nfe_budget": nfe_budget,
            "before_fallback_nfe": before_fallback_nfe,
            "equivalent_nfe": total_nfe,
            "fallback_nfe": None if total_nfe is None else total_nfe - before_fallback_nfe,
            "budget_overshoot": None if nfe_budget is None else max(0, total_nfe - nfe_budget),
            "checkpoint": self.level,
            "direction": self.direction,
            "sample": self.states,
        }
