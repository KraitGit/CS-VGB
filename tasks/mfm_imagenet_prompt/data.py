import argparse
import json
from pathlib import Path

import yaml


def load_conditions(path):
    with Path(path).expanduser().open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_calibration(path):
    return {row["prompt"]: row for row in load_conditions(path)}


def prepare(config):
    from huggingface_hub import hf_hub_download, snapshot_download

    conditions = load_conditions(config["data"]["conditions"])
    calibration = load_calibration(config["data"]["calibration"])
    for condition in conditions:
        row = calibration[condition["prompt"]]
        if not row["std"]["hpsv2"] > 0:
            raise ValueError("HPSv2 calibration requires positive standard deviation")
    models = json.loads(Path(config["data"]["models"]).expanduser().read_text())
    checkpoint = hf_hub_download(**models["model"])
    vae = snapshot_download(**models["vae"])
    reward = hf_hub_download(**models["reward"])
    return {"checkpoint": checkpoint, "vae": vae, "reward_checkpoint": reward,
            "calibration": str(Path(config["data"]["calibration"]).expanduser()),
            "condition_count": len(conditions)}


def main():
    parser = argparse.ArgumentParser(description="Prepare MFM, VAE and HPSv2.1 assets.")
    parser.add_argument("--config", default="configs/mfm_imagenet_prompt.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).expanduser().read_text())
    print(json.dumps(prepare(config), indent=2))


if __name__ == "__main__":
    main()
