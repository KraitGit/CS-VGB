import math

import torch

from .geometry import (segment_collision_cost, segment_box_clearance,
                       segment_sphere_clearance)


def gp_trajectory_cost(states, dt, sigma=1.0):
    """MMD constant-velocity GP energy, summed over robots and edges."""
    if states.shape[-1] != 4 or states.shape[-2] < 2:
        raise ValueError("Expected physical [q_x,q_y,v_x,v_y] trajectories")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("GP dt and sigma must be finite and positive")
    q, v = states[..., :2], states[..., 2:]
    # e^T Q^-1 e = 12/dt^3 ||dq-dt*(v0+v1)/2||^2 + ||dv||^2/dt.
    residual = q[..., 1:, :] - q[..., :-1, :] - .5*dt*(v[..., 1:, :] + v[..., :-1, :])
    velocity_change = v[..., 1:, :] - v[..., :-1, :]
    return (12*residual.square()/dt**3 + velocity_change.square()/dt).sum((-1, -2, -3))/sigma**2


class Reward:
    """Shared trajectory cost for scoring and guidance; feasibility is evaluated separately."""

    def __init__(self, model, config, benchmark, condition):
        self.model, self.config, self.benchmark = model, config, benchmark
        self.boxes = torch.as_tensor(condition["boxes"], device=model.device, dtype=torch.float32).reshape(-1, 4)
        self.spheres = torch.as_tensor(condition["spheres"], device=model.device, dtype=torch.float32).reshape(-1, 3)
        for name in ("temperature", "obstacle_cost_scale"):
            if not math.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"Invalid reward.{name}")
        self.calls = 0

    def physical_costs(self, physical):
        paths = physical[..., :2]
        robots = self.model.robots
        safe = self.benchmark["safe_distance"]
        dt = self.benchmark["duration"] / physical.shape[-2]
        collision = segment_collision_cost(paths, safe) / robots
        smoothness = gp_trajectory_cost(physical, dt) * dt**3 / (12 * safe**2 * robots)
        margin = self.benchmark["obstacle_margin"]
        distances = []
        if len(self.boxes):
            distances.append(segment_box_clearance(paths, self.boxes))
        if len(self.spheres):
            distances.append(segment_sphere_clearance(paths, self.spheres))
        if distances:
            distance = torch.cat(distances, -1).amin(-1)
            obstacle = (margin - distance).clamp_min(0).mean((-1, -2))
            obstacle = obstacle * self.config["obstacle_cost_scale"]
        else:
            obstacle = paths.sum((-1, -2, -3)) * 0
        return dict(collision=collision, smoothness=smoothness, obstacle=obstacle)

    def cost(self, x):
        self.calls += len(x)
        components = self.physical_costs(self.model.physical(x))
        return sum(self.config["weights"][key] * value for key, value in components.items())

    def __call__(self, x):
        return -self.cost(x) / self.config["temperature"]
