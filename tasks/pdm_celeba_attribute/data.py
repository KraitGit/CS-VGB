import argparse
import json
import yaml

from pathlib import Path


def load_conditions(path):
    with Path(path).expanduser().open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(config):
    from huggingface_hub import hf_hub_download
    import torch

    conditions = load_conditions(config["data"]["conditions"])
    models = json.loads(Path(config["data"]["models"]).expanduser().read_text())
    checkpoint = hf_hub_download(**models["model"])
    assets = {"checkpoint": checkpoint, "condition_count": len(conditions)}
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in models["reward_urls"].items():
        path = cache / url.rsplit("/", 1)[-1]
        if not path.is_file():
            torch.hub.download_url_to_file(url, str(path))
        assets[name] = str(path)
    return assets


def main():
    parser = argparse.ArgumentParser(description="Prepare PDM, face reward checkpoints, and CelebA conditions.")
    parser.add_argument("--config", default="configs/pdm_celeba_attribute.yaml")
    args = parser.parse_args()
    with Path(args.config).expanduser().open() as stream:
        config = yaml.safe_load(stream)
    print(json.dumps(prepare(config), indent=2))


if __name__ == "__main__":
    main()
