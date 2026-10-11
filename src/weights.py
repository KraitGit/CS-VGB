import math

import torch


def forward_log_weights(log_values, log_parent_values, level, depth_alpha):
    """Forward log contributions for aligned [particles, pool size] tensors."""
    # e_j,i^F
    if level < 1 or not math.isfinite(depth_alpha) or depth_alpha <= 0:
        raise ValueError("Forward weights require level >= 1 and positive finite depth_alpha")
    if level == 1:
        return log_values
    return (level - 1) * math.log(depth_alpha) + 0.5 * (log_parent_values + log_values)


def backward_log_weights(log_values, log_parent_values, level, depth_alpha):
    """Backward log contributions; parent values align with candidate values."""
    if level < 0 or not math.isfinite(depth_alpha) or depth_alpha <= 0:
        raise ValueError("Backward weights require level >= 0 and positive finite depth_alpha")
    if level == 0:
        return log_parent_values.expand_as(log_values)
    return level * math.log(depth_alpha) + 0.5 * (log_parent_values + log_values)


@torch.no_grad()
def population_action(log_active, log_reverse, generator, persistence=1., lazy=True, skip_stays=True):
    """Interior action: active [N,L], reverse [N] if lazy, otherwise [N,L_other]."""
    if not 0 <= persistence <= 1:
        raise ValueError("Persistence must be in [0, 1]")
    log_move = (torch.logsumexp(log_active, dim=1) - math.log(log_active.shape[1])).sum()
    if lazy:
        log_other = log_reverse.sum()
    else:
        log_other = (torch.logsumexp(log_reverse, dim=1) - math.log(log_reverse.shape[1])).sum()
    scale = torch.maximum(log_move, log_other)
    move = (log_move - scale).exp()
    other = (log_other - scale).exp()

    # skipping stays implementation
    # Lazy skipping conditions on the drawn indices, unlike exact rejection below.
    indices = None
    if lazy:
        indices = torch.randint(log_active.shape[1], (len(log_active),),
                                device=log_active.device, generator=generator)
        log_selected = log_active.gather(1, indices[:, None]).sum()
        overlap = (torch.minimum(log_selected, log_other) - scale).exp()
    else:
        overlap = torch.minimum(move, other)
    stay = persistence * overlap
    flip = (other - stay).clamp_min(0)
    if skip_stays:
        stay = torch.zeros_like(stay)
    masses = torch.stack((move, stay, flip))
    action = torch.multinomial(masses, 1, generator=generator).item()
    return ("move", "stay", "flip")[action], indices

    # exact skipping stays mentioned in appendix
    # while True:
    #     indices = None
    #     if lazy:
    #         indices = torch.randint(log_active.shape[1], (len(log_active),),
    #                                 device=log_active.device, generator=generator)
    #         log_selected = log_active.gather(1, indices[:, None]).sum()
    #         overlap = (torch.minimum(log_selected, log_other) - scale).exp()
    #     else:
    #         overlap = torch.minimum(move, other)
    #     stay = persistence * overlap
    #     masses = torch.stack((move, stay, (other - stay).clamp_min(0)))
    #     action = torch.multinomial(masses, 1, generator=generator).item()
    #     # Retrying a lazy stay must also redraw the auxiliary indices.
    #     if action != 1 or not skip_stays:
    #         return ("move", "stay", "flip")[action], indices
