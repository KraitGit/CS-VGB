import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import yaml


MODEL_ARCHIVE_ID = "1WO3tpvg-HU0m9RyDvGyfDamo7roBYMud"
CHECKPOINT_FILENAME = "checkpoints/ema_model_current_state_dict.pth"
CHECKPOINT_DIRECTORY = "~/.cache/cs_vgb/mmd/data_trained_models"


def load_conditions(path, protocol=None):
    with Path(path).expanduser().open() as stream:
        conditions = [json.loads(line) for line in stream if line.strip()]
    return [row for row in conditions if protocol is None or row["protocol"] == protocol]


def load_model_spec(path, protocol):
    return json.loads(Path(path).expanduser().read_text())[protocol]


def prepare(config):
    data = config["data"]
    protocol = data["protocol"]
    conditions = load_conditions(data["conditions"], protocol)
    if not conditions:
        raise ValueError(f"No conditions for {protocol}")
    spec = load_model_spec(data["models"], protocol)
    directory = Path(CHECKPOINT_DIRECTORY).expanduser()
    checkpoint = directory / spec["model_id"] / CHECKPOINT_FILENAME
    if not checkpoint.is_file():
        import gdown

        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "data_trained_models.tar.xz"
        if not archive.is_file():
            partial = archive.with_suffix(".part")
            gdown.download(id=MODEL_ARCHIVE_ID, output=str(partial), quiet=False)
            partial.replace(archive)
        # Extract only the selected weight file, never arbitrary archive paths.
        member_name = f"data_trained_models/{spec['model_id']}/{CHECKPOINT_FILENAME}"
        with tarfile.open(archive) as bundle:
            member = next(item for item in bundle if item.name.removeprefix("./") == member_name)
            if not member.isfile():
                raise ValueError("Checkpoint archive entry must be a regular file")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            partial = checkpoint.with_suffix(".part")
            with bundle.extractfile(member) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(checkpoint)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint differs from the frozen planning reference: {checkpoint}")
    return {"checkpoint": str(checkpoint), "settings": spec["settings"],
            "normalizer": spec["normalizer"], "condition_count": len(conditions)}


def main():
    parser = argparse.ArgumentParser(description="Prepare frozen MMD planning assets.")
    parser.add_argument("--config", default="configs/diffusion_multirobot_planning.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).expanduser().read_text())
    print(json.dumps(prepare(config), indent=2))


if __name__ == "__main__":
    main()
