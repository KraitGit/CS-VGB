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


def load_models(config):
    """Load the generator and face reward once for single or sweep inference."""
    logger.info("Loading libraries...")
    import torch

    from tasks.pdm_celeba_attribute import data
    from tasks.pdm_celeba_attribute.model import Model
    from tasks.pdm_celeba_attribute.reward import Reward

    common = config["inference_common"]
    device = torch.device(common["device"])
    if device.type != "cuda":
        raise ValueError("Inference requires a CUDA device, e.g. cuda:0")
    torch.manual_seed(common["seed"])
    logger.info("Preparing data and checkpoints...")
    assets = data.prepare(config)
    logger.info("Loading PDM generator...")
    model = Model(assets["checkpoint"], device, **config["model"])
    if common["compile_model"]:
        logger.info("Enabling torch.compile; first inference may take longer.")
        model.network = torch.compile(model.network)
    logger.info("Loading face detector and attribute reward...")
    reward = Reward(assets, device, config["reward"])
    return model, reward


def load_metric(config):
    from tasks.pdm_celeba_attribute.metric import Metric

    logger.info("Loading evaluation models...")
    return Metric(config["inference_common"]["device"], **config["metric"])


def run(config, models=None, metric=None):
    """Run one configured condition and save its output in a new directory."""
    if config["task"] != "pdm_celeba_attribute":
        raise ValueError("Only pdm_celeba_attribute is implemented")
    logger.info("Loading libraries...")
    import torch

    from src.sampler import Sampler
    from tasks.pdm_celeba_attribute import data
    from tasks.pdm_celeba_attribute.cs_vgb_adapter import Adapter

    common = config["inference_common"]
    conditions = data.load_conditions(config["data"]["conditions"])
    index = common["condition_index"]
    if not 0 <= index < len(conditions):
        raise ValueError("condition_index is outside the condition manifest")
    condition = conditions[index]
    device = torch.device(common["device"])
    if device.type != "cuda":
        raise ValueError("Inference requires a CUDA device, e.g. cuda:0")
    seed = common["seed"]
    mode = "guided" if config["cs_vgb"]["guided"] else "unguided"
    logger.info("PDM CelebA | condition=%s | seed=%s | %s | %s", index, seed, device, mode)
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model, reward = load_models(config) if models is None else models
    reward.set_prompt(condition)
    adapter = Adapter(model, reward)
    model.reset_counts()
    reward.calls = 0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    logger.info("Initializing %s particles and their values...", common["num_particles"])
    states = model.noise(common["num_particles"], generator)
    intervals = common["checkpoint_intervals"]
    times = [j / intervals for j in range(intervals + 1)]
    sampler = Sampler(adapter, states, times, generator, **config["cs_vgb"])
    score_scale = config["reward"]["temperature"] / len(condition["attributes"])
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
                    "level": f"{sampler.level}/{len(sampler.times)-1}",
                    "dir": sampler.direction, "action": action,
                    "NFE": str(model.equivalent_nfe) if budget is None else f"{model.equivalent_nfe}/{budget}",
                    "score": "--" if score is None else f"{math.exp(score.item() * score_scale):.3f}",
                }, refresh=False)
                last_display = now
            if budget is not None:
                bar.update(min(model.equivalent_nfe, budget) - bar.n)
            elif action in ("move", "stay", "flip"):
                bar.update(1)
            elif action in ("warp", "fallback"):
                bar.refresh()

        result = sampler.run(
            common["num_updates"], nfe_budget=common["nfe_budget"],
            fallback_f_only=common["fallback_f_only"],
            warp_beta=common["warp_beta"], direct_warp=common["direct_warp"],
            progress=progress)
        if budget is not None:
            bar.update(min(model.equivalent_nfe, budget) - bar.n)
        score = result["best_log_reward"]
        best_score = None if score is None else math.exp(score.item() * score_scale)
        bar.set_postfix(stop=result["stop_reason"], NFE=model.equivalent_nfe,
                        score="--" if best_score is None else f"{best_score:.3f}")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    logger.info("Saving results...")

    output_root = Path(common["output_dir"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"condition_{index:03d}_seed_{seed}_", dir=output_root))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "condition.json").write_text(json.dumps(condition, indent=2) + "\n")
    saved = {key: value.detach().cpu() if torch.is_tensor(value) else value
             for key, value in result.items()}
    torch.save(saved, output / "result.pt")
    if common["save_image"] and result["best"] is not None:
        from torchvision.utils import save_image
        save_image(model.decode(result["best"]).cpu(), output / "best.png")
    metadata = {key: value for key, value in saved.items() if not torch.is_tensor(value)}
    metadata.update({
        "task": config["task"], "condition_index": index, "seed": seed,
        "status": "complete" if result["best"] is not None else "no_output",
        "best_log_reward": None if result["best_log_reward"] is None else result["best_log_reward"].item(),
        "best_score": best_score,
        "forward_nfe": model.forward_nfe, "backward_nfe": model.backward_nfe,
        "reward_samples": reward.calls, "seconds": elapsed,
    })
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    logger.info("Finished | %.1fs | updates=%s | clean visits=%s | stop=%s",
                elapsed, result["updates"], result["clean_visits"], result["stop_reason"])
    logger.info("NFE=%s (forward=%s, VJP=%s) | best score=%s",
                model.equivalent_nfe, model.forward_nfe, model.backward_nfe,
                "--" if best_score is None else f"{best_score:.3f}")
    if result["best"] is not None:
        if metric is None:
            metric = load_metric(config)
        logger.info("Evaluating final image...")
        evaluation = metric.score(result["best"], condition, reward)
        metadata.update(evaluation)
        (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
        (output / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        logger.info("Metrics | %s", " | ".join(
            f"{key}={evaluation[key]:.3f}" for key in
            ("attribute_gm", "clip_iqa", "vqa")
            if key in evaluation))
    logger.info("Output: %s", output)
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Run one PDM CelebA condition.")
    parser.add_argument("--config", default="configs/pdm_celeba_attribute.yaml")
    parser.add_argument("--condition-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).expanduser().read_text())
    for key in ("condition_index", "seed", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config["inference_common"][key] = value
    run(config)


if __name__ == "__main__":
    main()
