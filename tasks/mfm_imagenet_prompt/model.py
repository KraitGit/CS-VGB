import torch

from .network import load_imagenet_flow_map


class Model:
    def __init__(self, checkpoint, vae, device, class_id=0, posterior_steps=1,
                 use_bfloat16=True, batch_size=16, cfg_scale=2.5):
        from diffusers import AutoencoderKL

        if posterior_steps < 1 or batch_size < 1 or cfg_scale not in (1., 2.5):
            raise ValueError("Use positive steps/batch size and released CFG 1 or 2.5")
        self.device = torch.device(device)
        self.posterior_steps = posterior_steps
        self.use_bfloat16 = use_bfloat16
        self.batch_size = batch_size
        self.cfg_scale = cfg_scale
        self.class_id = class_id
        network, scale, bias = load_imagenet_flow_map(checkpoint)
        self.network = network.to(self.device)
        self.scale, self.bias = scale.to(self.device), bias.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(vae).to(self.device).eval().requires_grad_(False)
        self.reset_counts()

    def reset_counts(self):
        self.forward_nfe = self.backward_nfe = 0

    @property
    def equivalent_nfe(self):
        """Generator compute convention: forward samples + 2 * VJP samples."""
        return self.forward_nfe + 2 * self.backward_nfe

    def _count_backward(self, gradient):
        self.backward_nfe += len(gradient)
        return gradient

    def noise(self, count, generator):
        return torch.randn(count, 4, 32, 32, device=self.device, generator=generator)

    def posterior(self, states, time, generator):
        # Paper time: zero is clean; the pretrained map uses the reverse convention.
        if not 0 <= time <= 1:
            raise ValueError("time must be in [0, 1]")
        if time == 0:
            return states
        inner = torch.randn(states.shape, device=states.device, dtype=states.dtype, generator=generator)
        for step in range(self.posterior_steps):
            start, end = step / self.posterior_steps, (step + 1) / self.posterior_steps
            outputs = []
            for batch, condition in zip(inner.split(self.batch_size), states.split(self.batch_size)):
                times, ends, conditions, scales = [batch.new_full((len(batch),), value)
                                                   for value in (start, end, 1 - time, self.cfg_scale)]
                labels = torch.full((len(batch),), self.class_id, device=self.device, dtype=torch.long)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_bfloat16):
                    velocity = self.network.v(times, ends, batch, conditions, condition,
                                              class_labels=labels, cfg_scale=scales)
                self.forward_nfe += len(batch)
                if velocity.requires_grad:
                    velocity.register_hook(self._count_backward)
                outputs.append(batch + (end - start) * velocity.float())
            inner = torch.cat(outputs)
        return inner

    @torch.inference_mode()
    def sample(self, count, generator):
        return self.posterior(self.noise(count, generator), 1., generator)

    def decode(self, states):
        outputs = []
        for batch in states.split(self.batch_size):
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_bfloat16):
                decoded = self.vae.decode((batch - self.bias) / self.scale).sample
            outputs.append(((decoded.float() + 1) / 2).clamp(0, 1))
        return torch.cat(outputs)
