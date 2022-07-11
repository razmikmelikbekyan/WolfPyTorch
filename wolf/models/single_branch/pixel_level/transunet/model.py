import math
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


ACT2FN = {"gelu": nn.functional.gelu, "relu": nn.functional.relu, "swish": swish}


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, in_channels, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict([
                ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
                ('gn', nn.GroupNorm(32, width, eps=1e-6)),
                ('relu', nn.ReLU(inplace=True)),
            ])
        )

        block_1 = nn.Sequential(
            OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))
                 for i in range(2, block_units[0] + 1)],
            ))
        block_2 = nn.Sequential(
            OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2))
                 for i in range(2, block_units[1] + 1)],
            ))
        block_3 = nn.Sequential(
            OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4))
                 for i in range(2, block_units[2] + 1)],
            ))
        self.body = nn.Sequential(OrderedDict([('block1', block_1), ('block2', block_2), ('block3', block_3)]))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features = [x]
        x = self.pool(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 0 < pad < 3, f"x {x.size()} should {right_size}"
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class Attention(nn.Module):
    def __init__(self, num_heads, hidden_size, att_drop_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(att_drop_rate)
        self.proj_dropout = nn.Dropout(att_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(drop_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, img_size, in_channels, grid, patch_size, num_layers, width_factor, hidden_size, drop_rate):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if grid is not None:  # ResNet
            grid_size = grid
            patch_size = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = True
        else:
            patch_size = _pair(patch_size)
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(in_channels, num_layers, width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, drop_rate, att_drop_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, drop_rate)
        self.attn = Attention(num_heads, hidden_size, att_drop_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, drop_rate, att_drop_rate, num_layers):
        super(Encoder, self).__init__()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.layer = nn.ModuleList(
            [deepcopy(Block(hidden_size, num_heads, mlp_dim, drop_rate, att_drop_rate)) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, in_channels, img_size, grid, patch_size, num_layers, width_factor, hidden_size, drop_rate,
                 att_drop_rate, num_heads, mlp_dim, trans_num_layers):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(
            img_size, in_channels, grid, patch_size, num_layers, width_factor, hidden_size, drop_rate
        )
        self.encoder = Encoder(hidden_size, num_heads, mlp_dim, drop_rate, att_drop_rate, trans_num_layers)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels,
            img_size,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_skip = nn.UpsamplingBilinear2d(scale_factor=skip_channels * img_size // 65536)

    def forward(self, x, skip=None):
        if skip is not None:
            try:
                y = self.up_skip(x)
                y = torch.cat([y, skip], dim=1)
            except (ValueError, RuntimeError):
                y = self.up(x)
                y = torch.cat([y, skip], dim=1)
        else:
            y = self.up(x)
        x = self.conv1(y)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, hidden_size, decoder_channels, n_skip, skip_channels, img_size):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        self.n_skip = n_skip
        if n_skip != 0:
            skip_channels = skip_channels
            for i in range(4 - n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, img_size, n_skip) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class TransUNet(nn.Module):
    """TransUNet is a image segmentation model which merits both transformers and unet

    Args:
        img_size: size of the input image
        classes: number of classes in output
        in_channels: number of channels in input image
        hidden_size: size of transformer hidden layers
        trans_num_layers: number of transformer layers
        grid: grid for patches which are passed in embedding layer
        decoder_channels: number of channels in decoder layers
        patch_size: size of patch in case grid is None
        num_layers: number of layers in each resnet block
        width_factor: width factor for resnet
        drop_rate: dropout rate in most layers
        att_drop_rate: dropout rate in attention layers
        num_heads: number of heads in transformer multi-headed attention
        mlp_dim: dimension of middle layer in mlp block
        n_skip: number of skip connections
        skip_channels: number of channels of skip connections
    Returns:
        ``torch.nn.Module``: **TransUNet**

    .. TransUNet:
        https://arxiv.org/pdf/2102.04306.pdf
    """

    def __init__(self, img_size: int = 512, classes: int = 1, in_channels: int = 3, hidden_size: int = 768,
                 trans_num_layers: int = 12, grid: tuple = (16, 16), decoder_channels: tuple = (256, 128, 64, 16),
                 patch_size: int = 16, num_layers: tuple = (3, 4, 9), width_factor: int = 1, drop_rate: float = 0.1,
                 att_drop_rate: float = 0.0, num_heads: int = 12, mlp_dim: int = 3072, n_skip: int = 0,
                 skip_channels: tuple = (512, 256, 64, 16)):
        super(TransUNet, self).__init__()
        if in_channels == 1:
            in_channels = 3
        skip_channels = list(skip_channels)
        self.transformer = Transformer(in_channels, img_size, grid, patch_size, num_layers, width_factor,
                                       hidden_size, drop_rate, att_drop_rate, num_heads, mlp_dim, trans_num_layers)
        self.decoder = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels, img_size)
        if n_skip == 0:
            upsampling = img_size / 256
        else:
            upsampling = 1
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
            upsampling=upsampling
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
