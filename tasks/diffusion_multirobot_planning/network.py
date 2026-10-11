# Adapted from MMD / MPD; see NOTICE.md.
import math

import torch
from torch import nn
from einops.layers.torch import Rearrange


class TimeEncoder(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim_out)
        )

    def forward(self, x):
        return self.encoder(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, padding=None, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, stride=1,
                      padding=padding if padding is not None else kernel_size // 2),
            Rearrange('batch channels n_support_points -> batch channels 1 n_support_points'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 n_support_points -> batch channels n_support_points'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, cond_embed_dim, n_support_points, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, n_groups=group_norm_n_groups(out_channels)),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=group_norm_n_groups(out_channels)),
        ])

        # Without context conditioning, cond_mlp handles only time embeddings
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, c):
        '''
            x : [ batch_size x inp_channels x n_support_points ]
            c : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x n_support_points ]
        '''
        h = self.blocks[0](x) + self.cond_mlp(c)
        h = self.blocks[1](h)
        res = self.residual_conv(x)
        out = h + res

        return out

def group_norm_n_groups(n_channels, target_n_groups=8):
    if n_channels < target_n_groups:
        return 1
    for n_groups in range(target_n_groups, target_n_groups + 10):
        if n_channels % n_groups == 0:
            return n_groups
    return 1

class TemporalUnet(nn.Module):
    def __init__(self, horizon, dimension, width, dim_mults):
        super().__init__()
        dims = [dimension, *[width * multiplier for multiplier in dim_mults]]
        pairs = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = TimeEncoder(32, 32)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(pairs):
            last = index == len(pairs) - 1
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, 32, horizon),
                ResidualTemporalBlock(dim_out, dim_out, 32, horizon),
                nn.Identity(), None,
                nn.Identity() if last else Downsample1d(dim_out),
            ]))
            if not last:
                horizon //= 2
        self.mid_block1 = ResidualTemporalBlock(dims[-1], dims[-1], 32, horizon)
        self.mid_attn = nn.Identity()
        self.mid_attention = nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(dims[-1], dims[-1], 32, horizon)
        for dim_in, dim_out in reversed(pairs[1:]):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, 32, horizon),
                ResidualTemporalBlock(dim_in, dim_in, 32, horizon),
                nn.Identity(), None, Upsample1d(dim_in),
            ]))
            horizon *= 2
        self.final_conv = nn.Sequential(
            Conv1dBlock(width, width, kernel_size=5, n_groups=group_norm_n_groups(width)),
            nn.Conv1d(width, dimension, 1),
        )

    def forward(self, states, time):
        embedding = self.time_mlp(time)
        states = states.transpose(1, 2)
        skips = []
        for first, second, _, _, downsample in self.downs:
            states = second(first(states, embedding), embedding)
            skips.append(states)
            states = downsample(states)
        states = self.mid_block2(self.mid_block1(states, embedding), embedding)
        for first, second, _, _, upsample in self.ups:
            states = torch.cat((states, skips.pop()), dim=1)
            states = upsample(second(first(states, embedding), embedding))
        return self.final_conv(states).transpose(1, 2)


def load_planning_diffusion(checkpoint, settings, normalizer):
    multipliers = {0: (1, 2, 4), 1: (1, 2, 4, 8)}
    network = TemporalUnet(
        normalizer["horizon"], len(normalizer["mins"]), settings["unet_input_dim"],
        multipliers[settings["unet_dim_mults_option"]],
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict({key.removeprefix("model."): value for key, value in state.items()
                             if key.startswith("model.")}, strict=True)
    names = (
        "betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod", "log_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
        "posterior_variance", "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    )
    schedule = {key: state[key] for key in names}
    if set(state) != set(names) | {"model." + key for key in network.state_dict()}:
        raise ValueError("Unexpected planning checkpoint entries")
    if any(value.shape != (settings["n_diffusion_steps"],) for value in schedule.values()):
        raise ValueError("Checkpoint schedule does not match diffusion step count")
    return network.eval().requires_grad_(False), schedule
