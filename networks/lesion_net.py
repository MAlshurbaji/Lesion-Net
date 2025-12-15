import torch
from torch import nn
from einops import rearrange
from torchvision.ops import StochasticDepth
from typing import List, Iterable


# ============================================================
# Normalization
# ============================================================
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


# ============================================================
# Adaptive Patch Aggregation
# ============================================================
class AdaptivePatchAggregation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )


# ============================================================
# Dynamic Self-Attention
# ============================================================
class DynamicSelfAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=reduction_ratio,
                stride=reduction_ratio,
            ),
            LayerNorm2d(channels),
        )
        self.attention = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape

        reduced_x = self.reducer(x)
        reduced_x = rearrange(reduced_x, 'b c h w -> b (h w) c')
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        out, _ = self.attention(x_flat, reduced_x, reduced_x)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        return out


# ============================================================
# Mix-FFN
# ============================================================
class MixFFN(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )


# ============================================================
# Residual wrapper
# ============================================================
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


# ============================================================
# Encoder Unit
# (Dynamic Self-Attention + Mix-FFN)
# ============================================================
class EncoderUnit(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = 0.0,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    DynamicSelfAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixFFN(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch"),
                )
            ),
        )


# ============================================================
# Encoder Stage
# (Adaptive Patch Aggregation + Encoder Units)
# ============================================================
class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[float],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()

        self.patch_aggregation = AdaptivePatchAggregation(
            in_channels, out_channels, patch_size, overlap_size
        )

        self.units = nn.Sequential(
            *[
                EncoderUnit(
                    out_channels,
                    reduction_ratio,
                    num_heads,
                    mlp_expansion,
                    drop_probs[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.patch_aggregation(x)
        x = self.units(x)
        return x


# ============================================================
# Utilities
# ============================================================
def chunks(data: Iterable, sizes: List[int]):
    curr = 0
    for size in sizes:
        yield data[curr : curr + size]
        curr += size


# ============================================================
# Encoder
# ============================================================
class LesionNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = 0.0,
    ):
        super().__init__()

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]

        self.stages = nn.ModuleList(
            [
                EncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions,
                )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# ============================================================
# Decoder
# (Channel Refinement + Upsampling)
# ============================================================
class LesionNetDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class LesionNetDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                LesionNetDecoderBlock(in_ch, out_channels, sf)
                for in_ch, sf in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        return [stage(f) for f, stage in zip(features, self.stages)]


# ============================================================
# Segmentation Head
# ============================================================
class LesionNetSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        return self.predict(x)


# ============================================================
# Full Model
# ============================================================
class LesionNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.encoder = LesionNetEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )

        self.decoder = LesionNetDecoder(
            decoder_channels, widths[::-1], scale_factors
        )

        self.head = LesionNetSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        return self.head(features)
