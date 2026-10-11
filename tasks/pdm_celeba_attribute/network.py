import math
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize(value, dim=None, eps=1e-4):
    if dim is None:
        dim = tuple(range(1, value.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    norm = value.float().square().sum(dim=dim, keepdim=True).sqrt()
    scale = math.sqrt(norm.numel() / value.numel())
    norm = eps + norm * scale
    return value / norm.to(value.dtype)


def resample(value, mode, resample_filter=(1, 1)):
    if mode == "keep":
        return value
    if tuple(resample_filter) != (1, 1):
        raise ValueError("The CelebA EDM2 checkpoint requires filter (1, 1)")

    filt = torch.tensor(
        resample_filter,
        device=value.device,
        dtype=torch.float32,
    )
    filt = torch.outer(filt, filt)
    filt = (filt / filt.sum()).to(value.dtype)
    weight = filt[None, None].expand(value.shape[1], 1, -1, -1)
    if mode == "down":
        return F.conv2d(value, weight, stride=2, groups=value.shape[1])
    if mode == "up":
        return F.conv_transpose2d(
            value,
            weight * 4,
            stride=2,
            groups=value.shape[1],
        )
    raise ValueError(f"Unknown resample mode: {mode}")


def mp_silu(value):
    return F.silu(value) / 0.596


def mp_sum(first, second, balance=0.5):
    denominator = math.sqrt((1 - balance) ** 2 + balance**2)
    return (first * (1 - balance) + second * balance) / denominator


def mp_cat(first, second, dim=1, balance=0.5):
    first_channels = first.shape[dim]
    second_channels = second.shape[dim]
    scale = math.sqrt(
        (first_channels + second_channels) / ((1 - balance) ** 2 + balance**2)
    )
    first_weight = scale * (1 - balance) / math.sqrt(first_channels)
    second_weight = scale * balance / math.sqrt(second_channels)
    return torch.cat([first * first_weight, second * second_weight], dim=dim)


class MPPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        if (dim // 2) % 2 != 0:
            raise ValueError("Half the embedding dimension must be even")
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        arguments = torch.outer(timesteps.float().reshape(-1), frequencies)
        return torch.cat(
            [arguments.cos() * math.sqrt(2), arguments.sin() * math.sqrt(2)],
            dim=-1,
        ).to(timesteps.dtype)


class MPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = tuple(kernel)
        shape = (out_channels, in_channels, *self.kernel)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, value, gain=1):
        weight = normalize(self.weight.float())
        weight_size = math.prod(weight.shape[1:])
        weight = weight * (gain / math.sqrt(weight_size))
        weight = weight.to(value.dtype)
        if weight.ndim == 2:
            return F.linear(value, weight)
        padding = weight.shape[-1] // 2
        return F.conv2d(value, weight, padding=padding)


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        flavor="enc",
        resample_mode="keep",
        attention=False,
        channels_per_head=64,
        dropout=0.0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.num_heads = out_channels // channels_per_head if attention else 0

        residual_input = out_channels if flavor == "enc" else in_channels
        self.emb_gain = nn.Parameter(torch.zeros(()))
        self.conv_res0 = MPConv(residual_input, out_channels, kernel=(3, 3))
        self.emb_linear = MPConv(emb_channels, out_channels)
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=(3, 3))
        self.conv_skip = None
        if in_channels != out_channels:
            self.conv_skip = MPConv(in_channels, out_channels, kernel=(1, 1))

        self.attn_qkv = None
        self.attn_proj = None
        if self.num_heads:
            self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=(1, 1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1, 1))

    def forward(self, value, embedding):
        value = resample(value, self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                value = self.conv_skip(value)
            value = normalize(value, dim=1)

        residual = self.conv_res0(mp_silu(value))
        conditioning = self.emb_linear(embedding, gain=self.emb_gain) + 1
        conditioning = conditioning[:, :, None, None]
        residual = mp_silu(residual * conditioning.to(residual.dtype))
        residual = F.dropout(residual, p=self.dropout, training=self.training)
        residual = self.conv_res1(residual)

        if self.flavor == "dec" and self.conv_skip is not None:
            value = self.conv_skip(value)
        value = mp_sum(value, residual, self.res_balance)

        if self.num_heads:
            residual = self.attn_qkv(value)
            batch, _, height, width = residual.shape
            residual = residual.reshape(batch, self.num_heads, -1, 3, height * width)
            residual = normalize(residual, dim=2)
            query = residual[:, :, :, 0, :]
            key = residual[:, :, :, 1, :]
            values = residual[:, :, :, 2, :]
            residual = F.scaled_dot_product_attention(
                query.transpose(-2, -1).contiguous(),
                key.transpose(-2, -1).contiguous(),
                values.transpose(-2, -1).contiguous(),
                dropout_p=0.0,
            ).transpose(-2, -1).reshape_as(value)
            residual = self.attn_proj(residual)
            value = mp_sum(value, residual, self.attn_balance)

        if self.clip_act is not None:
            value = value.clamp(-self.clip_act, self.clip_act)
        return value


def _flatten_tree(tree, prefix=()):
    flattened = {}
    for name, value in tree.items():
        path = prefix + (str(name),)
        if isinstance(value, dict):
            flattened.update(_flatten_tree(value, path))
        else:
            flattened["/".join(path)] = np.asarray(value)
    return flattened


class EDM2DiamondMapUNet(nn.Module):
    def __init__(
        self,
        img_resolution=64,
        img_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_blocks=3,
        attn_resolutions=(16, 8),
        concat_balance=0.5,
        block_kwargs=None,
    ):
        super().__init__()
        block_kwargs = {} if block_kwargs is None else dict(block_kwargs)
        channels = [model_channels * multiplier for multiplier in channel_mult]
        embedding_channels = max(channels)
        timestep_channels = channels[0]
        self.concat_balance = concat_balance
        self.out_gain = nn.Parameter(torch.zeros(()))

        self.emb_t_fourier = MPPositionalEmbedding(timestep_channels)
        self.emb_t_prime_fourier = MPPositionalEmbedding(timestep_channels)
        self.emb_s_fourier = MPPositionalEmbedding(timestep_channels)
        self.emb_s_prime_fourier = MPPositionalEmbedding(timestep_channels)
        self.emb_t_linear = MPConv(timestep_channels, embedding_channels)
        self.emb_t_prime_linear = MPConv(timestep_channels, embedding_channels)
        self.emb_s_linear = MPConv(timestep_channels, embedding_channels)
        self.emb_s_prime_linear = MPConv(timestep_channels, embedding_channels)

        encoder = {}
        output_channels = 2 * img_channels + 1
        for level, level_channels in enumerate(channels):
            resolution = img_resolution >> level
            if level == 0:
                encoder[f"{resolution}x{resolution}_conv"] = MPConv(
                    output_channels, level_channels, kernel=(3, 3)
                )
                output_channels = level_channels
            else:
                encoder[f"{resolution}x{resolution}_down"] = Block(
                    output_channels,
                    output_channels,
                    embedding_channels,
                    flavor="enc",
                    resample_mode="down",
                    **block_kwargs,
                )
            for index in range(num_blocks):
                input_channels = output_channels
                output_channels = level_channels
                encoder[f"{resolution}x{resolution}_block{index}"] = Block(
                    input_channels,
                    output_channels,
                    embedding_channels,
                    flavor="enc",
                    attention=resolution in attn_resolutions,
                    **block_kwargs,
                )
        self.encoder = nn.ModuleDict(encoder)

        decoder = {}
        skips = [module.out_channels for module in self.encoder.values()]
        for level, level_channels in reversed(list(enumerate(channels))):
            resolution = img_resolution >> level
            if level == len(channels) - 1:
                decoder[f"{resolution}x{resolution}_in0"] = Block(
                    output_channels,
                    output_channels,
                    embedding_channels,
                    flavor="dec",
                    attention=True,
                    **block_kwargs,
                )
                decoder[f"{resolution}x{resolution}_in1"] = Block(
                    output_channels,
                    output_channels,
                    embedding_channels,
                    flavor="dec",
                    **block_kwargs,
                )
            else:
                decoder[f"{resolution}x{resolution}_up"] = Block(
                    output_channels,
                    output_channels,
                    embedding_channels,
                    flavor="dec",
                    resample_mode="up",
                    **block_kwargs,
                )
            for index in range(num_blocks + 1):
                input_channels = output_channels + skips.pop()
                output_channels = level_channels
                decoder[f"{resolution}x{resolution}_block{index}"] = Block(
                    input_channels,
                    output_channels,
                    embedding_channels,
                    flavor="dec",
                    attention=resolution in attn_resolutions,
                    **block_kwargs,
                )
        self.decoder = nn.ModuleDict(decoder)
        self.out_conv = MPConv(output_channels, img_channels, kernel=(3, 3))

    def forward(
        self,
        value,
        outer_time,
        outer_delta,
        inner_time,
        inner_delta,
        outer_state,
    ):
        outer_embedding = mp_sum(
            self.emb_t_linear(self.emb_t_fourier(outer_time)),
            self.emb_t_prime_linear(self.emb_t_prime_fourier(outer_delta)),
            balance=0.5,
        )
        inner_embedding = mp_sum(
            self.emb_s_linear(self.emb_s_fourier(inner_time)),
            self.emb_s_prime_linear(self.emb_s_prime_fourier(inner_delta)),
            balance=0.5,
        )
        embedding = mp_silu(mp_sum(outer_embedding, inner_embedding, balance=0.5))

        value = torch.cat([value, torch.ones_like(value[:, :1]), outer_state], dim=1)
        skips = []
        for name, module in self.encoder.items():
            if "conv" in name:
                value = module(value)
            else:
                value = module(value, embedding)
            skips.append(value)

        for name, module in self.decoder.items():
            if "block" in name:
                value = mp_cat(value, skips.pop(), balance=self.concat_balance)
            value = module(value, embedding)
        return self.out_conv(value, gain=self.out_gain)

    def checkpoint_parameter_map(self):
        parameters = {
            "unet/out_gain": self.out_gain,
            "unet/emb_t_linear/mpconv_weight": self.emb_t_linear.weight,
            "unet/emb_t_prime_linear/mpconv_weight": (self.emb_t_prime_linear.weight),
            "unet/emb_s_linear/mpconv_weight": self.emb_s_linear.weight,
            "unet/emb_s_prime_linear/mpconv_weight": (self.emb_s_prime_linear.weight),
            "unet/out_conv/mpconv_weight": self.out_conv.weight,
        }
        for prefix, modules in (
            ("enc", self.encoder),
            ("dec", self.decoder),
        ):
            for name, module in modules.items():
                module_prefix = f"unet/{prefix}_{name}"
                if isinstance(module, MPConv):
                    parameters[f"{module_prefix}/mpconv_weight"] = module.weight
                    continue
                parameters[f"{module_prefix}/emb_gain"] = module.emb_gain
                for child_name in (
                    "conv_res0",
                    "emb_linear",
                    "conv_res1",
                    "conv_skip",
                    "attn_qkv",
                    "attn_proj",
                ):
                    child = getattr(module, child_name)
                    if child is not None:
                        key = f"{module_prefix}/{child_name}/mpconv_weight"
                        parameters[key] = child.weight
        return parameters


class EDM2DiamondMap(nn.Module):
    sigma_data = 1.0

    def __init__(self, use_bfloat16=True):
        super().__init__()
        self.use_bfloat16 = use_bfloat16
        self.unet = EDM2DiamondMapUNet(
            img_resolution=64,
            img_channels=3,
            model_channels=128,
            channel_mult=(1, 2, 3, 4),
            num_blocks=3,
            attn_resolutions=(16, 8),
            block_kwargs={"dropout": 0.0},
        )
        self.logvar_linear = MPConv(128, 1)

    @staticmethod
    def _time_batch(value, batch_size, device):
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
        if value.numel() == 1:
            return value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError("Time must be scalar or have one value per image")
        return value.reshape(batch_size)

    def forward(
        self,
        inner_state,
        outer_state,
        outer_time,
        outer_target,
        inner_time,
        inner_target,
    ):
        batch_size = len(inner_state)
        device = inner_state.device
        outer_time = self._time_batch(outer_time, batch_size, device)
        outer_target = self._time_batch(outer_target, batch_size, device)
        inner_time = self._time_batch(inner_time, batch_size, device)
        inner_target = self._time_batch(inner_target, batch_size, device)
        compute_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        prediction = self.unet(
            (inner_state.float() / self.sigma_data).to(compute_dtype),
            outer_time[:, None, None, None],
            (outer_target - outer_time)[:, None, None, None],
            inner_time[:, None, None, None],
            (inner_target - inner_time)[:, None, None, None],
            (outer_state.float() / self.sigma_data).to(compute_dtype),
        )
        velocity = self.sigma_data * prediction.float()
        delta = (inner_target - inner_time)[:, None, None, None]
        return inner_state + delta * velocity

    def checkpoint_parameter_map(self):
        parameters = self.unet.checkpoint_parameter_map()
        parameters["logvar_linear/mpconv_weight"] = self.logvar_linear.weight
        return parameters


def load_celeba_diamond_map(checkpoint, ema_factor=0.9999):
    with open(checkpoint, "rb") as handle:
        payload = pickle.load(handle)
    ema_key = str(ema_factor)
    ema_params = payload.get("ema_params", {})
    if ema_key not in ema_params:
        available = ", ".join(sorted(ema_params))
        raise KeyError(f"EMA factor {ema_key} is unavailable; found: {available}")
    tree = ema_params[ema_key]["params"]["diamond_map"]["net"]
    source = _flatten_tree(tree)
    del payload

    model = EDM2DiamondMap(use_bfloat16=True)
    target = model.checkpoint_parameter_map()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    if missing or unexpected:
        raise ValueError(
            "Checkpoint parameter mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    with torch.no_grad():
        for name, parameter in target.items():
            tensor = torch.from_numpy(np.array(source[name], copy=True))
            if parameter.shape != tensor.shape:
                raise ValueError(
                    f"Checkpoint shape mismatch for {name}: "
                    f"expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                )
            parameter.copy_(tensor)
    model.requires_grad_(False)
    model.eval()
    return model

