from functools import lru_cache

import torch


def _segment_closest_distance(origin, velocity):
    squared_speed = velocity.square().sum(-1)
    fraction = (-(origin*velocity).sum(-1)/squared_speed.clamp_min(1e-20)).clamp(0, 1)
    return (origin+fraction[..., None]*velocity).norm(dim=-1)


def segment_robot_distances(paths):
    """Closest center distances per robot pair and synchronized segment."""
    n = paths.shape[-3]
    i, j = torch.triu_indices(n, n, offset=1, device=paths.device)
    delta = paths[..., i, :, :] - paths[..., j, :, :]
    origin = delta[..., :-1, :]
    velocity = delta[..., 1:, :] - origin
    return _segment_closest_distance(origin, velocity)


def segment_collision_cost(paths, safe_distance):
    """Soft collision penalty between synchronized robot segments."""
    closest = segment_robot_distances(paths)
    return ((safe_distance-closest).clamp_min(0)/safe_distance).square().sum((-1, -2))


def trajectory_metrics(paths, starts, goals, benchmark, safe_distance):
    """Robot/endpoint metrics; obstacle and wall-radius checks are separate."""
    # paths: [joint candidates, robots, synchronized waypoints, xy].
    if paths.ndim != 4 or paths.shape[-1] != 2 or paths.shape[2] < 3:
        raise ValueError("Expected [B,N,H,2] trajectories with H>=3 for acceleration metrics")
    if not torch.isfinite(paths).all():
        raise ValueError("Nonfinite trajectories")
    n, horizon = paths.shape[1:3]
    i, j = torch.triu_indices(n, n, offset=1, device=paths.device)
    delta = paths[:, i] - paths[:, j]
    distances = delta.norm(dim=-1)
    collision_cost = ((safe_distance-distances).clamp_min(0)/safe_distance).square().sum((1, 2))
    closest = _segment_closest_distance(delta[:, :, :-1], delta[:, :, 1:]-delta[:, :, :-1])
    segment_cost = ((safe_distance-closest).clamp_min(0)/safe_distance).square().sum((1, 2))
    if n == 1:
        clearance = paths.new_full((len(paths),), torch.inf)
    else:
        clearance = closest.amin((1, 2)) - 2*benchmark["robot_radius"]
    endpoint_error = torch.maximum((paths[:, :, 0]-starts).norm(dim=-1).amax(1),
                                   (paths[:, :, -1]-goals).norm(dim=-1).amax(1))
    valid = (endpoint_error <= benchmark["endpoint_tolerance"]) & (
        paths.abs().amax((1, 2, 3)) <= benchmark["workspace_limit"]+1e-6)
    increments = paths[:, :, 1:]-paths[:, :, :-1]
    dt = benchmark["duration"]/(horizon-1)
    acceleration = (increments[:, :, 1:]-increments[:, :, :-1])/dt**2
    return {"collision_cost": collision_cost,
            "segment_collision_cost": segment_cost,
            "success": valid & (clearance > 0),
            "valid": valid, "minimum_clearance": clearance,
            "endpoint_error": endpoint_error,
            "path_length": increments.norm(dim=-1).sum(-1).mean(1),
            "mean_squared_acceleration": acceleration.square().sum(-1).mean((1, 2))}


@lru_cache(maxsize=16)
def _box_signs(device, dtype):
    return torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], device=device, dtype=dtype)


def segment_box_clearance(paths, boxes):
    """Exact minimum signed box distance along each linear segment."""
    # nearest distance to square (outer) / distance to side (inner)
    center, half = boxes[:, :2], boxes[:, 2:]/2
    origin = paths[..., :-1, None, :]-center
    velocity = (paths[..., 1:, :]-paths[..., :-1, :])[..., None, :].expand_as(origin)
    signs = _box_signs(paths.device, paths.dtype)
    corners = half[:, None, :]*signs
    speed2 = velocity.square().sum(-1, keepdim=True).clamp_min(1e-20)
    corner_t = ((corners-origin[..., None, :])*velocity[..., None, :]).sum(-1)/speed2
    intercept = torch.cat((origin-half, -origin-half), -1)
    slope = torch.cat((velocity, -velocity), -1)

    def roots(a, b):
        return torch.where(b.abs() > 1e-12, -a/torch.where(b.abs() > 1e-12, b, 1.), 0.)

    i, j = torch.triu_indices(4, 4, offset=1, device=paths.device)
    times = torch.cat((torch.zeros_like(corner_t[..., :1]), torch.ones_like(corner_t[..., :1]),
                       corner_t, roots(intercept, slope),
                       roots(intercept[..., i]-intercept[..., j], slope[..., i]-slope[..., j])), -1).clamp(0, 1)
    points = origin[..., None, :]+times[..., None]*velocity[..., None, :]
    residual = points.abs()-half[:, None, :]
    signed = residual.clamp_min(0).norm(dim=-1)+residual.amax(-1).clamp_max(0)
    return signed.amin(-1)


def segment_sphere_clearance(paths, spheres):
    """Signed segment-to-circle distance, before subtracting robot radius."""
    center, radius = spheres[:, :2], spheres[:, 2]
    origin = paths[..., :-1, None, :]-center
    velocity = (paths[..., 1:, None, :]-center)-origin
    return _segment_closest_distance(origin, velocity)-radius
