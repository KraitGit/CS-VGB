import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import tempfile

import yaml

from tasks.pdm_celeba_attribute import data
from tasks.pdm_celeba_attribute.inference_run import load_metric, load_models, run


logger = logging.getLogger(__name__)


def run_sweep(setting):
    config = yaml.safe_load(Path(setting["config"]).expanduser().read_text())
    if config["task"] != "pdm_celeba_attribute":
        raise ValueError("This sweep requires pdm_celeba_attribute")
    conditions = data.load_conditions(config["data"]["conditions"])
    count = setting["num_conditions"]
    if not isinstance(count, int) or not 1 <= count <= len(conditions):
        raise ValueError(f"num_conditions must be between 1 and {len(conditions)}")

    output_root = Path(setting["output_dir"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sweep_", dir=output_root))
    (output / "sweep.yaml").write_text(yaml.safe_dump(setting, sort_keys=False))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    models = load_models(config)
    metric = load_metric(config)
    base_seed = config["inference_common"]["seed"]
    for name, method in setting["methods"].items():
        for budget, preset in method["presets"].items():
            budget_config = deepcopy(config)
            for section in ("inference_common", "cs_vgb"):
                budget_config[section].update(setting.get(section, {}))
                budget_config[section].update(method.get(section, {}))
                budget_config[section].update(preset.get(section, {}))
            budget_config["inference_common"]["nfe_budget"] = budget
            for index in range(count):
                current = deepcopy(budget_config)
                current["inference_common"].update(
                    condition_index=index, seed=base_seed + index,
                    output_dir=str(output / name / f"nfe_{budget:05d}"))
                logger.info("%s | NFE=%s | condition %s/%s | index=%s", name, budget, index + 1, count, index)
                result_dir = run(current, models=models, metric=metric)
                metrics = json.loads((result_dir / "metrics.json").read_text())
                metrics["method"] = name
                metrics["nfe_budget"] = budget
                metrics["output_dir"] = str(result_dir)
                with (output / "summary.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
    logger.info("Sweep finished | %s budgets x %s conditions | %s",
                sum(len(method["presets"]) for method in setting["methods"].values()), count, output)
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Run the first N PDM CelebA conditions.")
    parser.add_argument("--config", default="configs/sweep/pdm_celeba_attribute_sweep.yaml")
    args = parser.parse_args()
    setting = yaml.safe_load(Path(args.config).expanduser().read_text())
    run_sweep(setting)


if __name__ == "__main__":
    main()
