import torch

from .geometry import segment_box_clearance, segment_sphere_clearance, trajectory_metrics


class Metric:
    def __init__(self, model, benchmark, condition):
        self.model, self.benchmark = model, benchmark
        self.boxes = torch.as_tensor(condition["boxes"], device=model.device, dtype=torch.float32).reshape(-1, 4)
        self.spheres = torch.as_tensor(condition["spheres"], device=model.device, dtype=torch.float32).reshape(-1, 3)

    @torch.no_grad()
    def score(self, x):
        paths = self.model.physical(x)[..., :2]
        starts, goals = self.model.physical_endpoints[..., :2]
        metrics = trajectory_metrics(paths, starts, goals, self.benchmark,
                                     self.benchmark["safe_distance"])
        clearance = paths.new_full((len(paths),), torch.inf)
        for objects, function in ((self.boxes, segment_box_clearance),
                                  (self.spheres, segment_sphere_clearance)):
            if len(objects):
                clearance = torch.minimum(clearance, function(paths, objects).amin((-1, -2, -3)))
        radius = self.benchmark["robot_radius"]
        metrics["obstacle_clearance"] = clearance - radius
        metrics["wall_clearance"] = self.benchmark["workspace_limit"] - paths.abs().amax((1, 2, 3)) - radius
        metrics["success"] &= (metrics["obstacle_clearance"] > 0) & (metrics["wall_clearance"] > 0)
        return metrics

    @torch.no_grad()
    def archive_rank(self, cost, metrics):
        violation = sum((-metrics[key]).clamp_min(0) for key in
                        ("minimum_clearance", "obstacle_clearance", "wall_clearance"))
        violation += (metrics["endpoint_error"] - self.benchmark["endpoint_tolerance"]).clamp_min(0)
        # Ranking criterion: success > failure; higher ranks are better.
        # Success: lower cost -> higher positive rank; failure: lower violation -> higher negative rank.
        return torch.where(metrics["success"], 1 / (1 + cost.clamp_min(0)), -1 - violation)
