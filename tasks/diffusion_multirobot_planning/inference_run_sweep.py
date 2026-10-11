import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import tempfile

import yaml

from tasks.diffusion_multirobot_planning import data
from tasks.diffusion_multirobot_planning.inference_run import load_model, run


logger = logging.getLogger(__name__)


def run_sweep(setting):
    config = yaml.safe_load(Path(setting["config"]).expanduser().read_text())
    if config["task"] != "diffusion_multirobot_planning":
        raise ValueError("This sweep requires diffusion_multirobot_planning")
    count, repeats = setting["num_conditions"], setting["num_repeats"]
    conditions = {}
    for protocol in setting["protocols"]:
        rows = data.load_conditions(config["data"]["conditions"], protocol)
        if not isinstance(count, int) or not 1 <= count <= len(rows):
            raise ValueError(f"num_conditions must be between 1 and {len(rows)} for {protocol}")
        if not isinstance(repeats, int) or not 1 <= repeats <= min(len(row["sampling_seeds"]) for row in rows[:count]):
            raise ValueError("num_repeats exceeds the available frozen sampling seeds")
        conditions[protocol] = rows[:count]

    output_root = Path(setting["output_dir"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sweep_", dir=output_root))
    (output / "sweep.yaml").write_text(yaml.safe_dump(setting, sort_keys=False))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    total = len(conditions) * count * repeats * sum(len(method["presets"]) for method in setting["methods"].values())
    completed = 0
    for protocol, rows in conditions.items():
        protocol_config = deepcopy(config)
        protocol_config["data"]["protocol"] = protocol
        protocol_config["inference_common"].update(setting.get("inference_common", {}))
        model = load_model(protocol_config, rows[0])
        for name, method in setting["methods"].items():
            for budget, preset in method["presets"].items():
                budget_config = deepcopy(protocol_config)
                for section in ("inference_common", "cs_vgb", "guidance"):
                    budget_config[section].update(setting.get(section, {}))
                    budget_config[section].update(method.get(section, {}))
                    budget_config[section].update(preset.get(section, {}))
                budget_config["inference_common"]["nfe_budget"] = budget
                for index, condition in enumerate(rows):
                    for repeat, seed in enumerate(condition["sampling_seeds"][:repeats]):
                        current = deepcopy(budget_config)
                        current["inference_common"].update(
                            condition_index=index, seed=seed,
                            output_dir=str(output / protocol / name / f"nfe_{budget:05d}" / f"repeat_{repeat}"))
                        logger.info("Run %s/%s | %s | %s | NFE=%s | map=%s/%s | repeat=%s/%s",
                                    completed + 1, total, protocol, name, budget, index + 1, count, repeat + 1, repeats)
                        result_dir = run(current, model=model)
                        metrics = json.loads((result_dir / "metrics.json").read_text())
                        metrics.update(method=name, nfe_budget=budget, repeat=repeat, output_dir=str(result_dir))
                        with (output / "summary.jsonl").open("a") as stream:
                            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                        completed += 1
        del model
    logger.info("Sweep finished | %s runs | %s", completed, output)
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Run the first N planning maps with frozen sampling seeds.")
    parser.add_argument("--config", default="configs/sweep/diffusion_multirobot_planning_sweep.yaml")
    args = parser.parse_args()
    setting = yaml.safe_load(Path(args.config).expanduser().read_text())
    run_sweep(setting)


if __name__ == "__main__":
    main()
