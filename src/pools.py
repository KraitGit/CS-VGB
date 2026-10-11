import torch


class Pool:
    """One candidate pool per particle slot, stored together as [N, L, ...]."""

    def __init__(self, size, inserted=None, inserted_log_values=None, generator=None):
        if not isinstance(size, int) or size < 1:
            raise ValueError("Pool size must be a positive integer")
        if (inserted is None) != (inserted_log_values is None):
            raise ValueError("Inserted states and log values must be provided together")
        self.size = size
        self.samples = self.log_values = self.parents = None
        self.mark = None
        self.inserted = None if inserted is None else inserted.detach()
        self.inserted_log_values = None if inserted_log_values is None else inserted_log_values.detach()
        if inserted is not None:
            self.mark = torch.randint(size, (len(inserted),), device=inserted.device, generator=generator)

    @property
    def remaining(self):
        if self.samples is not None:
            return 0
        return self.size - int(self.inserted is not None)

    @torch.no_grad()
    def fill(self, samples, log_values, parents):
        if self.samples is not None:
            return
        shape = (len(samples), self.remaining)
        if samples.shape[:2] != shape or log_values.shape != shape or parents.shape != shape:
            raise ValueError("Fresh candidates, values, and parents must align as [N, remaining, ...]")
        if self.inserted is None:
            self.samples = samples.detach()
            self.log_values = log_values.detach()
            self.parents = parents.detach()
            return

        slots = torch.arange(len(samples), device=samples.device)
        indices = (self.mark[:, None] + 1 + torch.arange(self.remaining, device=samples.device)) % self.size
        self.samples = samples.new_empty((len(samples), self.size, *samples.shape[2:]))
        self.log_values = log_values.new_empty((len(samples), self.size))
        self.parents = parents.new_empty((len(samples), self.size))
        self.samples[slots[:, None], indices] = samples
        self.log_values[slots[:, None], indices] = log_values
        self.parents[slots[:, None], indices] = parents
        self.samples[slots, self.mark] = self.inserted
        self.log_values[slots, self.mark] = self.inserted_log_values
        # The reverse candidate originates at the current particle in this slot.
        self.parents[slots, self.mark] = slots
        self.inserted = self.inserted_log_values = None

    def marked_values(self):
        if self.mark is None:
            raise ValueError("This pool has no marked candidate")
        if self.log_values is None:
            return self.inserted_log_values
        return self.log_values.gather(1, self.mark[:, None]).squeeze(1)


@torch.no_grad()
def select_local(candidates, log_values, log_weights, parents, argmax=True, generator=None):
    """Select one child per complete local pool; all inputs start with [N, L]."""
    selected = (log_weights.argmax(dim=1) if argmax else
                torch.multinomial(log_weights.softmax(dim=1), 1, generator=generator).squeeze(1))
    slots = torch.arange(len(candidates), device=candidates.device)
    return (
        candidates[slots, selected],
        log_values[slots, selected],
        parents[slots, selected],
        selected,
    )
