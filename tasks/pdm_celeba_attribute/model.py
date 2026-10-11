import torch

from .network import load_celeba_diamond_map


class Model:
    def __init__(self, checkpoint, device, posterior_steps=1, use_bfloat16=False, batch_size=32):
        if posterior_steps < 1 or batch_size < 1:
            raise ValueError("posterior_steps and batch_size must be positive")
        self.device = torch.device(device)
        self.network = load_celeba_diamond_map(checkpoint).to(self.device)
        self.network.use_bfloat16 = use_bfloat16
        self.posterior_steps = posterior_steps
        self.batch_size = batch_size
        self.reset_counts()

    def reset_counts(self):
        self.forward_nfe = 0
        self.backward_nfe = 0

    @property
    def equivalent_nfe(self):
        """Generator compute convention: forward samples + 2 * VJP samples."""
        return self.forward_nfe + 2 * self.backward_nfe

    def _count_backward(self, gradient):
        self.backward_nfe += len(gradient)
        return gradient

    def noise(self, count, generator):
        return torch.randn(count, 3, 64, 64, device=self.device, generator=generator)

    def posterior(self, states, time, generator):
        # Paper time: zero is clean; the pretrained map uses the reverse convention.
        if not 0 <= time <= 1:
            raise ValueError("time must be in [0, 1]")
        if time == 0:
            return states
        inner = torch.randn(
            states.shape, device=states.device, dtype=states.dtype, generator=generator
        )
        for step in range(self.posterior_steps):
            outputs = []
            for batch, condition in zip(inner.split(self.batch_size), states.split(self.batch_size)):
                output = self.network(
                    batch, condition, 1 - time, 1.0,
                    step / self.posterior_steps, (step + 1) / self.posterior_steps,
                )
                self.forward_nfe += len(batch)
                if output.requires_grad:
                    output.register_hook(self._count_backward)
                outputs.append(output)
            inner = torch.cat(outputs)
        return inner

    @torch.inference_mode()
    def sample(self, count, generator):
        return self.posterior(self.noise(count, generator), 1.0, generator)

    @staticmethod
    def decode(states):
        return ((states.float() + 1) / 2).clamp(0, 1)
