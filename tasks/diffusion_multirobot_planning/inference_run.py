import argparse
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import time

import yaml
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def save_figure(paths, condition, benchmark, output, title):
    """Plot one joint trajectory in physical coordinates without a display server."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    paths = paths.detach().cpu().numpy()
    figure = Figure(figsize=(7, 7), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    limit = benchmark["workspace_limit"]
    radius = benchmark["robot_radius"]
    ax.add_patch(Rectangle((-limit, -limit), 2 * limit, 2 * limit,
                           fill=False, edgecolor="#475569", linewidth=1.2))
    for cx, cy, width, height in condition["boxes"]:
        ax.add_patch(Rectangle((cx - width / 2, cy - height / 2), width, height,
                               facecolor="#cbd5e1", edgecolor="#64748b", zorder=2))
    for cx, cy, obstacle_radius in condition["spheres"]:
        ax.add_patch(Circle((cx, cy), obstacle_radius, facecolor="#cbd5e1",
                            edgecolor="#64748b", zorder=2))
    colors = colormaps["tab10" if len(paths) <= 10 else "tab20"]
    for index, (path, start, goal) in enumerate(zip(paths, condition["starts"], condition["goals"])):
        color = colors(index % colors.N)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=.9, zorder=3)
        ax.add_patch(Circle(start[:2], radius, facecolor=color, edgecolor=color, alpha=.2, zorder=4))
        ax.scatter(*start[:2], s=35, facecolors="white", edgecolors=[color], linewidths=1.5, zorder=5)
        ax.scatter(*goal[:2], marker="*", s=110, color=color, edgecolors="white", linewidths=.5, zorder=5)
        ax.annotate(str(index + 1), start[:2], xytext=(5, 5), textcoords="offset points",
                    color=color, fontsize=8, fontweight="bold", zorder=6)
    ax.legend(handles=[
        Line2D([], [], marker="o", color="#475569", markerfacecolor="white", linestyle="none", label="Start"),
        Line2D([], [], marker="*", color="#475569", markersize=10, linestyle="none", label="Goal"),
        Line2D([], [], color="#475569", label="Trajectory"),
    ], loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3, frameon=False)
    padding = radius * 2
    ax.set(xlim=(-limit - padding, limit + padding), ylim=(-limit - padding, limit + padding),
           xlabel="x", ylabel="y", title=title, aspect="equal")
    ax.set_axisbelow(True)
    ax.grid(alpha=.15)
    figure.savefig(output / "best.png", dpi=180)
    figure.savefig(output / "best.pdf")
    figure.clear()


def load_model(config, condition):
    """Load the generator once for inference within one protocol."""
    import torch

    from tasks.diffusion_multirobot_planning import data
    from tasks.diffusion_multirobot_planning.model import Model

    common = config["inference_common"]
    device = torch.device(common["device"])
    if device.type != "cuda":
        raise ValueError("Inference requires a CUDA device, e.g. cuda:0")
    logger.info("Preparing data and checkpoint...")
    assets = data.prepare(config)
    logger.info("Loading planning diffusion model...")
    model = Model(assets, condition, device)
    if common["compile_model"]:
        logger.info("Enabling torch.compile; first inference may take longer.")
        model.network = torch.compile(model.network)
    return model


def run(config, model=None):
    """Run one map; a supplied model must belong to the selected protocol."""
    if config["task"] != "diffusion_multirobot_planning":
        raise ValueError("Only diffusion_multirobot_planning is implemented")
    logger.info("Loading libraries...")
    import torch

    from src.sampler import Sampler
    from tasks.diffusion_multirobot_planning import data
    from tasks.diffusion_multirobot_planning.cs_vgb_adapter import Adapter
    from tasks.diffusion_multirobot_planning.metric import Metric
    from tasks.diffusion_multirobot_planning.reward import Reward

    common = config["inference_common"]
    protocol = config["data"]["protocol"]
    conditions = data.load_conditions(config["data"]["conditions"], protocol)
    index = common["condition_index"]
    if not 0 <= index < len(conditions):
        raise ValueError("condition_index is outside the selected protocol")
    condition = conditions[index]
    device = torch.device(common["device"])
    if device.type != "cuda":
        raise ValueError("Inference requires a CUDA device, e.g. cuda:0")
    seed = common["seed"]
    options = config["cs_vgb"]
    logger.info("Planning | %s | condition=%s | seed=%s | %s | %s",
                protocol, index, seed, device, "guided" if options["guided"] else "unguided")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = load_model(config, condition) if model is None else model
    model.set_condition(condition)
    times = common["checkpoint_time_grid"]
    if (len(times) < 2 or times[0] != 0 or any(type(t) is not int for t in times)
            or any(s >= t for s, t in zip(times, times[1:])) or times[-1] > model.steps):
        raise ValueError("Require increasing integer checkpoints 0 = k_0 < ... <= model.steps")
    reward = Reward(model, config["reward"], common, condition)
    metric = Metric(model, common, condition)
    adapter = Adapter(model, reward, metric, noise_std_scale=common["noise_std_scale"],
                      guidance=config["guidance"])
    model.reset_counts()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    logger.info("Initializing %s particles: native level %s -> %s...",
                common["num_particles"], model.steps, times[-1])
    states = adapter.initial(common["num_particles"], times[-1], generator,
                             guided=options["guided"] and options["guidance_forward"])
    prefix_nfe = model.equivalent_nfe
    sampler = Sampler(adapter, states, times, generator, **options)
    budget = common["nfe_budget"]
    last_display = 0.
    with tqdm(total=budget if budget is not None else common["num_updates"],
              desc="Inference", unit="NFE" if budget is not None else "update",
              dynamic_ncols=True, mininterval=.5, file=sys.stdout) as bar:
        if budget is not None:
            bar.update(min(model.equivalent_nfe, budget))

        def progress(action):
            nonlocal last_display
            now = time.perf_counter()
            if now - last_display >= .5 or action in ("warp", "fallback"):
                score = adapter.best_log_reward
                bar.set_description("Fallback" if action == "fallback" else "Inference", refresh=False)
                bar.set_postfix({
                    "level": sampler.times[sampler.level], "dir": sampler.direction,
                    "action": action, "NFE": model.equivalent_nfe,
                    "success": "--" if adapter.best_rank is None else bool(adapter.best_rank.item() > 0),
                    "cost": "--" if score is None else f"{-score.item() * reward.config['temperature']:.3f}",
                }, refresh=False)
                last_display = now
            if budget is not None:
                bar.update(min(model.equivalent_nfe, budget) - bar.n)
            elif action in ("move", "stay", "flip"):
                bar.update(1)
            elif action in ("warp", "fallback"):
                bar.refresh()

        result = sampler.run(
            common["num_updates"], nfe_budget=budget, fallback_f_only=common["fallback_f_only"],
            warp_beta=common["warp_beta"], direct_warp=common["direct_warp"], progress=progress,
            stop_on_feasible=common["stop_on_feasible"])
        if budget is not None:
            bar.update(min(model.equivalent_nfe, budget) - bar.n)
        score = result["best_log_reward"]
        best_cost = None if score is None else -score.item() * reward.config["temperature"]
        bar.set_postfix(stop=result["stop_reason"], NFE=model.equivalent_nfe,
                        cost="--" if best_cost is None else f"{best_cost:.3f}")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    logger.info("Saving results...")
    output_root = Path(common["output_dir"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{protocol}_{index:03d}_seed_{seed}_", dir=output_root))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "condition.json").write_text(json.dumps(condition, indent=2) + "\n")
    saved = {key: value.detach().cpu() if torch.is_tensor(value) else value
             for key, value in result.items()}
    evaluation = {}
    if result["best"] is not None:
        logger.info("Evaluating final trajectories...")
        with torch.no_grad():
            saved["paths"] = model.physical(result["best"])[..., :2].cpu()
            saved["endpoints"] = model.physical_endpoints[..., :2].cpu()
            for key, value in metric.score(result["best"]).items():
                scalar = value.item()
                # No robots/obstacles to collide with can produce infinite clearance.
                evaluation[key] = scalar if math.isfinite(scalar) else None
        (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
    torch.save(saved, output / "result.pt")
    metadata = {key: value for key, value in saved.items() if not torch.is_tensor(value)}
    metadata.update({
        "task": config["task"], "protocol": protocol, "condition_index": index, "seed": seed,
        "status": "complete" if result["best"] is not None else "no_output",
        "best_log_reward": None if score is None else score.item(), "best_cost": best_cost,
        "prefix_nfe": prefix_nfe, "forward_nfe": model.forward_nfe, "backward_nfe": model.backward_nfe,
        "first_feasible_nfe": adapter.first_feasible_nfe,
        "geometry_gradient_samples": adapter.geometry_gradient_samples,
        "reward_samples": reward.calls, "seconds": elapsed, **evaluation,
    })
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if common["save_figure"] and result["best"] is not None:
        logger.info("Saving trajectory figure...")
        title = (f"{protocol} | map {index} | seed {seed}\n"
                 f"{'Success' if evaluation['success'] else 'Infeasible'} | cost {best_cost:.3f}")
        save_figure(saved["paths"][0], condition, common, output, title)
    logger.info("Finished | %.1fs | updates=%s | stop=%s | NFE=%s", elapsed,
                result["updates"], result["stop_reason"], model.equivalent_nfe)
    logger.info("success=%s | cost=%s | path length=%s", evaluation.get("success", "--"),
                "--" if best_cost is None else f"{best_cost:.3f}",
                "--" if not evaluation else f"{evaluation['path_length']:.3f}")
    logger.info("Output: %s", output)
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Run one multi-robot planning condition.")
    parser.add_argument("--config", default="configs/diffusion_multirobot_planning.yaml")
    parser.add_argument("--protocol", choices=("highways", "circle020", "conveyor_moderate", "conveyor_hard"))
    parser.add_argument("--condition-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).expanduser().read_text())
    if args.protocol is not None:
        config["data"]["protocol"] = args.protocol
    for key in ("condition_index", "seed", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config["inference_common"][key] = value
    run(config)


if __name__ == "__main__":
    main()
