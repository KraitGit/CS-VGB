# CS-VGB

PyTorch implementation of CS-VGB for search and guided inference with pretrained generative models.

## Contents

- Shared CS-VGB sampling and task-specific model adapters.
- Single-condition inference and NFE-budget sweeps for three tasks.
- Pretrained checkpoint preparation and task-specific evaluation.

| Task | Generative model | Evaluation |
| :--- | :--- | :--- |
| CelebA attributes | Posterior Diamond Map | Attribute GM · CLIP-IQA · VQA |
| ImageNet prompts | Meta Flow Map | HPSv2 · CLIP-IQA · VQA |
| Multi-robot planning | MMD diffusion | Feasibility · Trajectory costs |

## Repository Layout

```text
.
├── src/                     # shared CS-VGB sampler and proposal logic
├── tasks/                   # task-specific models, rewards, and entry points
│   ├── pdm_celeba_attribute/
│   ├── mfm_imagenet_prompt/
│   └── diffusion_multirobot_planning/
├── configs/                 # single-run settings
│   └── sweep/               # task-specific sweep presets
├── requirements.txt
└── outputs/                 # generated results (created at runtime)
```

## Installation

Create an environment and install the dependencies:

```bash
conda create -n cs_vgb python=3.11 -y
conda activate cs_vgb
pip install -r requirements.txt
```

## Data and Checkpoints

Run commands from the repository root. Choose a task and prepare its pretrained checkpoints:

```bash
TASK=pdm_celeba_attribute
# TASK=mfm_imagenet_prompt
# TASK=diffusion_multirobot_planning

python -m tasks.${TASK}.data
```

Conditions and model manifests are included in each task's `assets/` directory. Checkpoints download on first use and are cached for subsequent runs.

## Inference

Single condition:

```bash
python -m tasks.${TASK}.inference_run
```

Sweep over conditions and NFE budgets:

```bash
python -m tasks.${TASK}.inference_run_sweep
```

## Configuration

| Location | Purpose |
| :--- | :--- |
| `configs/<task>.yaml` | Single-run settings, guidance, and evaluation |
| `configs/sweep/<task>_sweep.yaml` | NFE presets and number of conditions |
| `outputs/` | Default location for results and metrics |

Set `num_conditions` in the sweep YAML to run the first N conditions. Sweep runs reuse loaded models and evaluators.

CLIP-IQA and VQA evaluation can be toggled in the task configuration.

## Acknowledgments

We thank the authors and maintainers of the following code and pretrained models:

- **CelebA:** [Posterior Diamond Maps](https://github.com/PeterHolderrieth/diamond_maps) / [weights](https://huggingface.co/MonkeyDoug/diamond-maps), [EDM2](https://github.com/NVlabs/edm2), and [Flow Maps](https://github.com/nmboffi/flow-maps).
- **ImageNet:** [Meta Flow Maps](https://github.com/adh1s/mfm) / [weights](https://huggingface.co/adh1s/mfm), and [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse).
- **Planning:** [MMD](https://github.com/yoraish/mmd), including its released diffusion checkpoints.
- **Rewards and evaluation:** [FACER](https://github.com/FacePerceiver/facer), [FaRL](https://github.com/FacePerceiver/FaRL), [RetinaFace weights](https://github.com/elliottzheng/face-detection), [HPSv2 weights](https://huggingface.co/xswu/HPSv2), [OpenCLIP](https://github.com/mlfoundations/open_clip), [CLIP-IQA](https://github.com/IceClear/CLIP-IQA), [TorchMetrics](https://github.com/Lightning-AI/torchmetrics), and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

## Citation

```bibtex
@article{jeon2026imperfect,
  title={Imperfect Verifiers, Exact Targets: Inference-Time Scaling for Continuous Generation},
  author={Jeon, Kijung and Tao, Molei},
  journal={arXiv preprint arXiv:2610.15281},
  year={2026},
  url={https://arxiv.org/abs/2610.15281}
}
```

## License

Original code in this repository is released under the [MIT License](LICENSE). Third-party code and pretrained weights retain their upstream licenses.
