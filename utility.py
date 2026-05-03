# -*- coding: utf-8 -*-
import math
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import platform
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div


# =======================
# Helpers for DataLoader (OS/GPU-safe)
# =======================
_DL_CONFIG_CACHE = None  # Cache for DataLoader config

def _dl_worker_cfg():
    """Return (num_workers, persistent_workers) based on OS - with caching."""
    global _DL_CONFIG_CACHE
    if _DL_CONFIG_CACHE is not None:
        return _DL_CONFIG_CACHE
    
    is_windows = platform.system().lower().startswith("win")
    num_workers = 2 if is_windows else 4  # 2 workers on Windows for speed
    persistent = (num_workers > 0)
    
    _DL_CONFIG_CACHE = (num_workers, persistent)
    return _DL_CONFIG_CACHE


def _pin(device: torch.device) -> bool:
    return (str(device).startswith("cuda") and torch.cuda.is_available())


# =======================
# GEM Energy Network - OPTIMIZED FOR 97-98% PERFORMANCE
# =======================
class GEMEnergyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], p_drop=0.1, use_tanh=False):
        super(GEMEnergyNetwork, self).__init__()
        print(f"Initializing GEM Energy Network with input_dim={input_dim} [Tanh: {use_tanh}]")
        self.use_tanh = use_tanh
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(spectral_norm(nn.Linear(prev_dim, hidden_dim)))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.LeakyReLU(0.1))  # Reduced slope for more stability
            self.layers.append(nn.Dropout(p_drop))
            prev_dim = hidden_dim

        # FIX: Removed Tanh to increase energy range and better ID/OOD distinction
        # Tanh limited output to [-1,1] which was problematic for VOS-EBM
        # But for Baseline (without VOS) it's necessary to prevent gate saturation
        self.out = nn.Sequential(
            spectral_norm(nn.Linear(prev_dim, 512)),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.01),
            spectral_norm(nn.Linear(512, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        for layer in self.layers:
            x = layer(x)
        energy = self.out(x)
        if self.use_tanh:
            energy = torch.tanh(energy)
        return energy


# =======================
# GEM Multi-Head Integration Gate - OPTIMIZED
# =======================
class GEMIntegrationGate(nn.Module):
    def __init__(self, num_classes, feature_dim, hidden_dim=512):
        super().__init__()
        print(f"Initializing GEM Integration Gate (num_classes={num_classes})")

        self.fuse = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim + 1, hidden_dim)),
            nn.LeakyReLU(0.1),  # Reduced slope
            nn.Dropout(0.03),  # Reduced dropout
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(hidden_dim // 2, num_classes)),
            nn.Sigmoid()
        )

    def forward(self, s, feat):
        h = torch.cat([feat, s], dim=1)
        gate = self.fuse(h)
        gate = torch.clamp(gate, min=0.1, max=0.9)
        return gate


# =======================
# GEM Base architectures - OPTIMIZED
# =======================
class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(x.shape[0], self.out_c - self.in_c, x.shape[2], x.shape[3], device=x.device)
        x = torch.cat((x, pad), dim=1)
        return x


class EnhancedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(EnhancedBasicBlock, self).__init__()
        self.conv1 = wrapped_conv(input_size, in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(math.ceil(input_size / stride), planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(
                    wrapped_conv(input_size, in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class EnhancedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, temp=1.0, spectral_normalization=True, mod=True, coeff=3,
                 n_power_iterations=1, mnist=False):
        super(EnhancedResNet, self).__init__()
        self.in_planes = 64
        self.mod = mod

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride, padding=0):
            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            if not spectral_normalization:
                return conv
            if kernel_size == 1:
                wrapped_conv = spectral_norm(conv)
            else:
                wrapped_conv = spectral_norm(conv)
            return wrapped_conv

        self.wrapped_conv = wrapped_conv
        self.bn1 = nn.BatchNorm2d(64)

        if mnist:
            self.conv1 = wrapped_conv(28, 1, 64, kernel_size=3, stride=1, padding=1)
            self.layer1 = self._make_layer(block, 28, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 28, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 14, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 7, 512, num_blocks[3], stride=2)
        else:
            # Improved architecture for CIFAR-10/CIFAR-100
            self.pre_layer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.conv1 = wrapped_conv(32, 64, 64, kernel_size=3, stride=1, padding=1)
            self.layer1 = self._make_layer(block, 32, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 16, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 8, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.activation = F.leaky_relu if self.mod else F.relu
        self.feature = None
        self.temp = temp

    def _make_layer(self, block, input_size, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(input_size, self.wrapped_conv, self.in_planes, planes, stride, self.mod))
            self.in_planes = planes * block.expansion
            input_size = math.ceil(input_size / stride)
        return nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self, 'pre_layer'):
            x = self.pre_layer(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        self.feature = out
        out = self.fc(out) / self.temp
        return out


def enhanced_resnet18(num_classes, spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    return EnhancedResNet(EnhancedBasicBlock, [2, 2, 2, 2], num_classes, spectral_normalization=spectral_normalization,
                          mod=mod, temp=temp, mnist=mnist, **kwargs)


class VGG(nn.Module):
    def __init__(self, features, output_dim, k_lipschitz=None, p_drop=None):
        super(VGG, self).__init__()
        self.features = features
        if k_lipschitz is not None:
            l_1, l_2, l_3 = spectral_norm(nn.Linear(512, 512)), spectral_norm(nn.Linear(512, 512)), spectral_norm(
                nn.Linear(512, output_dim))
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                l_1,
                nn.ReLU(True),
                nn.Dropout(p=p_drop),
                l_2,
                nn.ReLU(True),
                l_3,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=p_drop),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, output_dim),
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self.feature = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, k_lipschitz=None):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if k_lipschitz is not None:
                conv2d = spectral_norm(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn(output_dim, k_lipschitz=None, p_drop=.5):
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 16.)
    return VGG(make_layers(cfg['D'], batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def conv_net(embedding_dim=576):
    input_dims = [28, 28, 1]
    linear_hidden_dims = [64, 64, 64]
    conv_hidden_dims = [64, 64, 64]
    output_dim = 10
    kernel_dim = 5
    k_lipschitz = 1
    batch_size = 64
    return convolution_linear_sequential(
        input_dims=input_dims,
        linear_hidden_dims=linear_hidden_dims,
        conv_hidden_dims=conv_hidden_dims,
        output_dim=output_dim,
        kernel_dim=kernel_dim,
        batch_size=batch_size,
        k_lipschitz=k_lipschitz,
        p_drop=None,
        embedding_dim=embedding_dim
    )


def vgg16(p_drop):
    output_dim = 10
    k_lipschitz = 1
    return vgg16_bn(output_dim=10, k_lipschitz=1, p_drop=p_drop)


def resnet(num_classes=100):
    return enhanced_resnet18(num_classes, spectral_normalization=True, mod=True, temp=1.0, mnist=False)


class SpectralLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_linear(x)
        return y


class SpectralConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_dim, padding, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_dim, padding=padding))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_conv(x)
        return y


def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1. / num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)


def convolution_sequential(input_dims, hidden_dims, output_dim, kernel_dim, k_lipschitz=None, p_drop=None):
    channel_dim = input_dims[2]
    dims = [channel_dim] + hidden_dims
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralConv(dims[i], dims[i + 1], kernel_dim, (kernel_dim - 1) // 2, k_lipschitz ** (1. / num_layers))
            layers.append(l)
        else:
            layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_dim, padding=(kernel_dim - 1) // 2))
        layers.append(nn.ReLU())
        if p_drop is not None:
            layers.append(nn.Dropout(p=p_drop))
        layers.append(nn.MaxPool2d(2, padding=0))
    return nn.Sequential(*layers)


class ConvLinSeq(nn.Module):
    def __init__(self, input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size,
                 k_lipschitz, p_drop, embedding_dim):
        super().__init__()
        if k_lipschitz is not None:
            k_lipschitz = k_lipschitz ** (1. / 2.)
        self.convolutions = convolution_sequential(
            input_dims=input_dims,
            hidden_dims=conv_hidden_dims,
            output_dim=conv_hidden_dims[-1],
            kernel_dim=kernel_dim,
            k_lipschitz=k_lipschitz,
            p_drop=p_drop
        )
        conv_output_size = conv_hidden_dims[-1] * (input_dims[0] // 2 ** len(conv_hidden_dims)) * (
                input_dims[1] // 2 ** len(conv_hidden_dims))
        self.linear = nn.Sequential(
            nn.Linear(conv_output_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, input):
        batch_size = input.size(0)
        input = self.convolutions(input)
        self.feature = input.clone().detach().reshape(batch_size, -1)
        input = self.linear(self.feature)
        return input


def convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size,
                                  k_lipschitz, p_drop=None, embedding_dim=576):
    return ConvLinSeq(
        input_dims=input_dims,
        linear_hidden_dims=linear_hidden_dims,
        conv_hidden_dims=conv_hidden_dims,
        output_dim=output_dim,
        kernel_dim=kernel_dim,
        batch_size=batch_size,
        k_lipschitz=k_lipschitz,
        p_drop=p_drop,
        embedding_dim=embedding_dim
    )


# =======================
# GEM-CORE Model - OPTIMIZED FOR 97-98% PERFORMANCE
# =======================
class EnhancedGEMModel(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, temp=1.0, use_tanh_energy=False):
        super(EnhancedGEMModel, self).__init__()
        print(f"Initializing Enhanced GEM Model (multi-head gate + aux uncertainty head) [Tanh Energy: {use_tanh_energy}]")
        self.base_model = base_model
        self.energy_network = GEMEnergyNetwork(feature_dim, use_tanh=use_tanh_energy)
        self.integration_gate = GEMIntegrationGate(num_classes=num_classes, feature_dim=feature_dim)
        self.uncertainty_head = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, 256)),
            nn.LeakyReLU(0.1),  # Reduced slope
            nn.Dropout(0.03),  # Reduced dropout
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(128, 1))
        )
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(1) * temp)

    def get_features_consistent(self, x):
        """Get consistent features with a single forward pass (without gradient cutoff)"""
        # 1) Use pre-computed feature if valid
        if hasattr(self.base_model, 'feature') and self.base_model.feature is not None and self.base_model.feature.size(
                0) == x.size(0):
            feats = self.base_model.feature
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
            feats = torch.clamp(feats, -1e6, 1e6)
            return F.normalize(feats, p=2, dim=1)

        # 2) Hook from last block (if available)
        if hasattr(self.base_model, 'layer4'):
            buf = [None]

            def _hook(_m, _i, o):
                buf[0] = o

            h = self.base_model.layer4.register_forward_hook(_hook)
            _ = self.base_model(x)  # One full pass
            h.remove()
            if buf[0] is not None:
                raw = buf[0]
                if raw.dim() == 4:
                    raw = torch.nn.functional.adaptive_avg_pool2d(raw, (1, 1)).view(raw.size(0), -1)
                raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
                raw = torch.clamp(raw, -1e6, 1e6)
                return F.normalize(raw, p=2, dim=1)

        # 3) Manual extraction ResNet-style (if layers exist)
        if hasattr(self.base_model, 'fc') and hasattr(self.base_model, 'conv1'):
            x_temp = x
            if hasattr(self.base_model, 'pre_layer'):
                x_temp = self.base_model.pre_layer(x_temp)
            x_temp = self.base_model.conv1(x_temp)
            x_temp = self.base_model.bn1(x_temp)
            x_temp = self.base_model.activation(x_temp)
            if hasattr(self.base_model, 'maxpool'):
                x_temp = self.base_model.maxpool(x_temp)
            x_temp = self.base_model.layer1(x_temp)
            x_temp = self.base_model.layer2(x_temp)
            x_temp = self.base_model.layer3(x_temp)
            x_temp = self.base_model.layer4(x_temp)
            x_temp = self.base_model.avgpool(x_temp)
            raw = x_temp.view(x_temp.size(0), -1)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
            raw = torch.clamp(raw, -1e6, 1e6)
            return F.normalize(raw, p=2, dim=1)

        # 4) Safe fallback
        p = torch.ones(x.size(0), 512, device=x.device)
        p = torch.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
        p = torch.clamp(p, -1e6, 1e6)
        return F.normalize(p, p=2, dim=1)


    def get_energy(self, x, return_features: bool = False):
        """Return energy E(x) for a batch x.
        Designed for VOS/EBM: supports gradients w.r.t. input (do NOT wrap in no_grad).
        If return_features=True, also returns normalized features used by energy head.
        """
        feats = self.get_features_consistent(x)
        e = self.energy_network(feats)
        e = e.squeeze(-1) if e.dim() > 1 else e
        if return_features:
            return e, feats
        return e

    def forward(self, x, return_features=False):
        try:
            logits = self.base_model(x)
            features = self.get_features_consistent(x)  # Use improved function

            energy = self.energy_network(features)
            s = torch.sigmoid(energy)
            gate_weights = self.integration_gate(s, features)
            u_log_alpha0 = self.uncertainty_head(features)

            # Calculate Alpha0 from Dirichlet parameters
            alpha = torch.exp(torch.clamp(logits, min=-15, max=15)) + 1e-8
            alpha0 = alpha.sum(dim=1, keepdim=True)

            gated_logits = logits * gate_weights
            gated_logits = gated_logits / self.temperature

            if return_features:
                return gated_logits, features, energy, gate_weights, u_log_alpha0, alpha0.squeeze(1)
            return gated_logits
        except Exception as e:
            print(f"Enhanced GEM Model Warning: {e}")
            logits = self.base_model(x)
            features = self.base_model.feature if hasattr(self.base_model, 'feature') else None
            energy = torch.zeros(x.shape[0], 1, device=x.device)
            gate_weight = torch.ones(x.shape[0], self.num_classes, device=x.device)
            u_log_alpha0 = torch.zeros(x.shape[0], 1, device=x.device)
            alpha0 = torch.ones(x.shape[0], device=x.device) * 10.0
            if return_features:
                return logits, features, energy, gate_weight, u_log_alpha0, alpha0
            return logits


# =======================
# Enhanced GEM-MIX Model - OPTIMIZED FOR 97-98% PERFORMANCE
# =======================
class EnhancedGEMMixModel(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, num_components=3, temp=1.0, fi_lambda=0.5, use_tanh_energy=False, use_fi_modulation=True):
        super(EnhancedGEMMixModel, self).__init__()
        print(f"Initializing Enhanced GEM-MIX Model with {num_components} components (λ={fi_lambda}) [Tanh Energy: {use_tanh_energy}, FI Modulation: {use_fi_modulation}]")

        self.base_model = base_model
        self.num_components = num_components
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.fi_lambda = fi_lambda
        self.use_fi_modulation = use_fi_modulation  # Ablation: separate control for FI modulation
        self.temperature = nn.Parameter(torch.ones(1) * temp)
        self.gmm_model = None

        self.dirichlet_heads = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Linear(feature_dim, 512)),
                nn.LeakyReLU(0.1),  # Reduced slope
                nn.Dropout(0.03),  # Reduced dropout
                spectral_norm(nn.Linear(512, 256)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Linear(256, num_classes))
            ) for _ in range(num_components)
        ])

        self.mixture_gate = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim + 1, 256)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.03),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(128, num_components)),
            nn.Softmax(dim=1)
        )

        self.energy_network = GEMEnergyNetwork(feature_dim, use_tanh=use_tanh_energy)
        self.integration_gate = GEMIntegrationGate(num_classes=num_classes, feature_dim=feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_features_consistent(self, x):
        """Get consistent features with a single forward pass (without gradient cutoff)"""
        # 1) Use pre-computed feature if valid
        if hasattr(self.base_model, 'feature') and self.base_model.feature is not None and self.base_model.feature.size(
                0) == x.size(0):
            feats = self.base_model.feature
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
            feats = torch.clamp(feats, -1e6, 1e6)
            return F.normalize(feats, p=2, dim=1)

        # 2) Hook from last block (if available)
        if hasattr(self.base_model, 'layer4'):
            buf = [None]

            def _hook(_m, _i, o):
                buf[0] = o

            h = self.base_model.layer4.register_forward_hook(_hook)
            _ = self.base_model(x)  # One full pass
            h.remove()
            if buf[0] is not None:
                raw = buf[0]
                if raw.dim() == 4:
                    raw = torch.nn.functional.adaptive_avg_pool2d(raw, (1, 1)).view(raw.size(0), -1)
                raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
                raw = torch.clamp(raw, -1e6, 1e6)
                return F.normalize(raw, p=2, dim=1)

        # 3) Manual extraction ResNet-style (if layers exist)
        if hasattr(self.base_model, 'fc') and hasattr(self.base_model, 'conv1'):
            x_temp = x
            if hasattr(self.base_model, 'pre_layer'):
                x_temp = self.base_model.pre_layer(x_temp)
            x_temp = self.base_model.conv1(x_temp)
            x_temp = self.base_model.bn1(x_temp)
            x_temp = self.base_model.activation(x_temp)
            if hasattr(self.base_model, 'maxpool'):
                x_temp = self.base_model.maxpool(x_temp)
            x_temp = self.base_model.layer1(x_temp)
            x_temp = self.base_model.layer2(x_temp)
            x_temp = self.base_model.layer3(x_temp)
            x_temp = self.base_model.layer4(x_temp)
            x_temp = self.base_model.avgpool(x_temp)
            raw = x_temp.view(x_temp.size(0), -1)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
            raw = torch.clamp(raw, -1e6, 1e6)
            return F.normalize(raw, p=2, dim=1)

        # 4) Safe fallback
        p = torch.ones(x.size(0), 512, device=x.device)
        p = torch.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
        p = torch.clamp(p, -1e6, 1e6)
        return F.normalize(p, p=2, dim=1)

    def apply_fisher_modulation(self, raw_weights, fi_traces):
        if fi_traces is None or self.fi_lambda == 0:
            return raw_weights

        fi_normalized = fi_traces / (fi_traces.sum(dim=1, keepdim=True) + 1e-8)
        modulation = torch.exp(self.fi_lambda * (1 - fi_normalized))
        modulated_weights = raw_weights * modulation
        modulated_weights = modulated_weights / (modulated_weights.sum(dim=1, keepdim=True) + 1e-8)
        modulated_weights = F.normalize(modulated_weights + 1e-4, p=1, dim=1)
        return modulated_weights

    def compute_fisher_information(self, features, component_logits, labels):
        batch_size = features.size(0)
        fi_traces = torch.zeros(batch_size, self.num_components, device=features.device)
        for k in range(self.num_components):
            logits_k = component_logits[k]
            log_probs = F.log_softmax(logits_k, dim=1)
            with torch.enable_grad():
                sample_fi = []
                for i in range(batch_size):
                    log_prob_i = log_probs[i, labels[i]]
                    grad_first = torch.autograd.grad(log_prob_i, logits_k, retain_graph=True, create_graph=False)[0]
                    if grad_first is not None:
                        fi_value = (grad_first[i] ** 2).sum()
                        sample_fi.append(fi_value)
                    else:
                        sample_fi.append(torch.tensor(0.0, device=features.device))
                fi_traces[:, k] = torch.stack(sample_fi)
        return fi_traces

    def compute_feature_density(self, features):
        if self.gmm_model is None:
            return torch.ones(features.size(0), 1, device=features.device)
        with torch.no_grad():
            log_probs_per_class = self.gmm_model.log_prob(features[:, None, :])
            log_pz = torch.logsumexp(log_probs_per_class, dim=1) - torch.log(
                torch.tensor(self.num_classes, device=features.device))
            density = torch.pow(torch.sigmoid(log_pz), 1.2)
            return density.unsqueeze(1)


    def get_energy(self, x, return_features: bool = False):
        """Return energy E(x) for a batch x.
        Designed for VOS/EBM: supports gradients w.r.t. input (do NOT wrap in no_grad).
        If return_features=True, also returns normalized features used by energy head.
        """
        feats = self.get_features_consistent(x)
        e = self.energy_network(feats)
        e = e.squeeze(-1) if e.dim() > 1 else e
        if return_features:
            return e, feats
        return e

    def forward(self, x, labels=None, return_features=False, use_fi_regularization=True, full_output=False):
        do_fi = bool(use_fi_regularization) and bool(self.training) and bool(torch.is_grad_enabled())

        try:
            logits = self.base_model(x)
            features = self.get_features_consistent(x)

            energy = self.energy_network(features)
            s = torch.sigmoid(energy)

            component_logits = [head(features) / self.temperature for head in self.dirichlet_heads]
            component_alphas = [torch.exp(torch.clamp(logits, min=-10, max=10)) + 1e-8 for logits in component_logits]

            gate_input = torch.cat([features, s], dim=1)
            raw_mixture_weights = self.mixture_gate(gate_input)

            fi_traces = None
            if do_fi:
                labels_for_fi = labels
                if labels_for_fi is None:
                    with torch.no_grad():
                        avg_logits = sum(component_logits) / len(component_logits)
                        labels_for_fi = torch.argmax(avg_logits, dim=1)
                try:
                    fi_traces = self.compute_fisher_information(features, component_logits, labels_for_fi)
                    # Ablation: only apply modulation if use_fi_modulation is True
                    if self.use_fi_modulation:
                        mixture_weights = self.apply_fisher_modulation(raw_mixture_weights, fi_traces)
                    else:
                        mixture_weights = raw_mixture_weights  # FI computed but not used for modulation
                except Exception as e:
                    print(f"⚠️ FI computation failed: {e}")
                    mixture_weights = raw_mixture_weights
            else:
                mixture_weights = raw_mixture_weights

            density_scores = self.compute_feature_density(features)
            scaled_alphas = [a * density_scores for a in component_alphas]

            final_probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
            alpha0_effective = torch.zeros(x.size(0), device=x.device)

            for k in range(self.num_components):
                alpha_k = scaled_alphas[k]
                alpha0_k = alpha_k.sum(dim=1, keepdim=True)
                expectation_k = alpha_k / (alpha0_k + 1e-8)
                final_probs += mixture_weights[:, k].unsqueeze(1) * expectation_k
                alpha0_effective += mixture_weights[:, k] * alpha0_k.squeeze(1)

            final_probs = final_probs / (final_probs.sum(dim=1, keepdim=True) + 1e-8)

            gate_weights = self.integration_gate(s, features)
            gated_final_probs = final_probs * gate_weights
            gated_final_probs = gated_final_probs / (gated_final_probs.sum(dim=1, keepdim=True) + 1e-8)

            if return_features or full_output:
                return (gated_final_probs, features, energy, gate_weights,
                        mixture_weights, scaled_alphas, fi_traces, alpha0_effective)
            else:
                return gated_final_probs

        except Exception as e:
            print(f"❌ Error in Enhanced GEMMoBModel forward: {e}")
            base_output = self.base_model(x)
            if isinstance(base_output, torch.Tensor):
                return F.softmax(base_output, dim=1)
            return base_output


# === Temperature helpers ===
import json


def load_calibrated_temperature(output_dir: str):
    try:
        path = os.path.join(output_dir, "calibrated_temperature.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                obj = json.load(f)
            return float(obj.get("T", 1.0))
    except Exception:
        pass
    return None


def apply_temperature_to_logits(logits: torch.Tensor, T: float | None) -> torch.Tensor:
    if T is not None and T > 0:
        return logits / max(T, 1e-6)
    return logits


# =======================
# Enhanced Loaders for CIFAR-10/CIFAR-100 - OPTIMIZED
# =======================
def load_model(ID_dataset, pretrained, index, dropout_rate, device, embedding_dim, use_mob=False, num_components=3,
               backbone=None, fi_lambda=None, use_vos=False, use_spectral_norm=True, use_fi_modulation=True):
    print(f"Loading {'Enhanced GEM-MIX' if use_mob else 'Enhanced GEM'} model for {ID_dataset}:")
    print(f"  Backbone: {backbone}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Pretrained: {pretrained}")
    if use_mob:
        print(f"  Number of mixture components: {num_components}")

    # 🔥 Automatic logic for Tanh: if VOS is OFF, we MUST use Tanh to bound energy
    # because unbounded energy + sigmoid activation saturates the gate gradients.
    # If VOS is ON, the negative sampling regularizes energy, so Tanh is not needed (and possibly harmful).
    use_tanh_energy = not use_vos
    if use_tanh_energy:
        print("  ✅ [Fix] Enabled Tanh activation for Energy Network (VOS is OFF)")
    else:
        print("  ℹ️ [Info] Tanh activation Disabled for Energy Network (VOS is ON)")

    base_path = os.environ.get("GEM_BASE_PATH", "./saved_models")

    if ID_dataset == "MNIST":
        base_model = conv_net(embedding_dim=embedding_dim)
        if pretrained:
            model_path = os.path.join(base_path, f"mnist_conv_gem_{index + 1}.pt")
            if os.path.exists(model_path):
                base_model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Enhanced GEM: Loaded pretrained MNIST model from {model_path}")
            else:
                print(f"Enhanced GEM Warning: Pretrained model not found at {model_path}")

    elif ID_dataset == "CIFAR-10":
        if backbone == "ResNet18":
            base_model = enhanced_resnet18(num_classes=10, spectral_normalization=use_spectral_norm, mod=True, temp=1.0, mnist=False)
            print("  Using Enhanced ResNet-18 backbone for CIFAR-10")
        elif backbone == "VGG16":
            base_model = vgg16(dropout_rate)
            print("  Using VGG16 backbone for CIFAR-10")
        elif backbone == "ConvNet3C3F":
            base_model = convolution_linear_sequential(
                input_dims=[32, 32, 3],
                linear_hidden_dims=[64, 64],
                conv_hidden_dims=[64, 128, 256],
                output_dim=10,
                kernel_dim=3,
                batch_size=64,
                k_lipschitz=1,
                p_drop=dropout_rate,
                embedding_dim=embedding_dim
            )
            print("  Using ConvNet3C3F backbone for CIFAR-10")
        else:
            base_model = enhanced_resnet18(num_classes=10, spectral_normalization=use_spectral_norm, mod=True, temp=1.0, mnist=False)
            print(f"  Using default Enhanced ResNet-18 backbone for CIFAR-10 (unsupported backbone: {backbone})")

        if pretrained:
            model_path = os.path.join(base_path, f"cifar10_cifar10_{(backbone or 'resnet18').lower()}_gem__gem_{index + 1}.pt")
            if os.path.exists(model_path):
                base_model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Enhanced GEM: Loaded pretrained CIFAR-10 {backbone} from {model_path}")
            else:
                print(f"Enhanced GEM Warning: Pretrained model not found at {model_path}")

    elif ID_dataset == "CIFAR-100":
        if backbone == "ResNet18":
            base_model = enhanced_resnet18(num_classes=100, spectral_normalization=use_spectral_norm, mod=True, temp=1.0,
                                           mnist=False)
            print("  Using Enhanced ResNet-18 backbone for CIFAR-100")
        elif backbone == "VGG16":
            base_model = vgg16(dropout_rate)
            print("  Using VGG16 backbone for CIFAR-100")
        else:
            base_model = enhanced_resnet18(num_classes=100, spectral_normalization=use_spectral_norm, mod=True, temp=1.0,
                                           mnist=False)
            print(f"  Using default Enhanced ResNet-18 backbone for CIFAR-100 (unsupported backbone: {backbone})")

        if pretrained:
            model_path = os.path.join(base_path, f"cifar100_cifar100_{(backbone or 'resnet18').lower()}_gem__gem_{index + 1}.pt")
            if os.path.exists(model_path):
                base_model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Enhanced GEM: Loaded pretrained CIFAR-100 {backbone} from {model_path}")
            else:
                print(f"Enhanced GEM Warning: Pretrained model not found at {model_path}")
    else:
        raise ValueError(f"Enhanced GEM: Unsupported dataset {ID_dataset}")

    num_classes = 10 if ID_dataset == "CIFAR-10" else (100 if ID_dataset == "CIFAR-100" else 10)

    if use_mob:
        gem_model = EnhancedGEMMixModel(base_model, embedding_dim, num_classes, num_components=num_components,
                                          fi_lambda=fi_lambda if fi_lambda is not None else 0.5,
                                          use_tanh_energy=use_tanh_energy, use_fi_modulation=use_fi_modulation)
    else:
        gem_model = EnhancedGEMModel(base_model, embedding_dim, num_classes, use_tanh_energy=use_tanh_energy)

    gem_model.to(device)

    print(
        f"{'Enhanced GEM-MIX' if use_mob else 'Enhanced GEM'} model loaded successfully with {sum(p.numel() for p in gem_model.parameters())} parameters")
    return gem_model


def load_datasets(ID_dataset, batch_size, val_size, data_dir=None):
    print(f"Loading Enhanced GEM datasets for {ID_dataset}:")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation size: {val_size}")

    if data_dir is None:
        data_dir = os.environ.get("GEM_DATA_DIR", "./data")

    ood_loader3 = None  # Third OOD loader (TinyImageNet for CIFAR-10)
    
    if ID_dataset == "MNIST":
        trainloader, validloader, testloader, ood_loader1, ood_loader2 = dataloaders_mnist(
            batch_size, val_size, data_dir)
    elif ID_dataset == "CIFAR-10":
        trainloader, validloader, testloader, ood_loader1, ood_loader2, ood_loader3 = dataloaders_cifar10(
            batch_size, val_size, data_dir)
    elif ID_dataset == "CIFAR-100":
        trainloader, validloader, testloader, ood_loader1, ood_loader2 = dataloaders_cifar100(
            batch_size, val_size, data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {ID_dataset}")

    print("Enhanced GEM datasets loaded successfully")
    return trainloader, validloader, testloader, ood_loader1, ood_loader2, ood_loader3


def dataloaders_mnist(batch_size, val_size, data_dir):
    print("Creating Enhanced GEM MNIST data loaders...")
    split_seed = int(os.environ.get("GEM_VAL_SEED", "42"))
    g = torch.Generator().manual_seed(split_seed)

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_trainval_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    mnist_test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    n_train = len(mnist_trainval_dataset)
    n_val = int(0.2 * n_train) if val_size is None else int(val_size * n_train)
    n_train_eff = n_train - n_val
    mnist_train_dataset, mnist_val_dataset = torch.utils.data.random_split(
        mnist_trainval_dataset, [n_train_eff, n_val], generator=g
    )

    fmnist_dataset = datasets.FashionMNIST(root=data_dir, download=True, train=False, transform=test_transform)
    kmnist_dataset = datasets.KMNIST(root=data_dir, download=True, train=False, transform=test_transform)

    num_workers, allow_persist = _dl_worker_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = _pin(device)

    mnist_trainloader = DataLoader(mnist_train_dataset, shuffle=True, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=pin, persistent_workers=allow_persist)
    mnist_validloader = DataLoader(mnist_val_dataset, shuffle=False, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=pin, persistent_workers=allow_persist)
    mnist_testloader = DataLoader(mnist_test_dataset, shuffle=False, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin, persistent_workers=allow_persist)
    fmnist_loader = DataLoader(fmnist_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin, persistent_workers=allow_persist)
    kmnist_loader = DataLoader(kmnist_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin, persistent_workers=allow_persist)

    print("Enhanced GEM MNIST data loaders created successfully")
    return mnist_trainloader, mnist_validloader, mnist_testloader, fmnist_loader, kmnist_loader


def dataloaders_cifar10(batch_size, val_size, data_dir):
    print("Creating Enhanced GEM CIFAR-10 data loaders with ADVANCED augmentations...")
    split_seed = int(os.environ.get("GEM_VAL_SEED", "42"))
    rng = np.random.RandomState(split_seed)

    num_workers, allow_persist = _dl_worker_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = _pin(device)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # Advanced Data Augmentation for CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    cifar10_train_dataset_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    cifar10_valid_dataset_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=valid_transform)

    num_train = len(cifar10_train_dataset_full)
    indices = np.arange(num_train)
    rng.shuffle(indices)

    split = int(np.floor(val_size * num_train))
    cifar10_train_dataset = Subset(cifar10_train_dataset_full, indices[split:])
    cifar10_valid_dataset = Subset(cifar10_valid_dataset_full, indices[:split])

    cifar10_test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    svhn_dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=test_transform)
    cifar100_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    
    # TinyImageNet as 3rd OOD dataset - resize from 64x64 to 32x32 to match CIFAR-10
    tinyimagenet_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize from 64x64 to 32x32
        transforms.ToTensor(),
        normalize
    ])
    tinyimagenet_path = os.path.join(data_dir, "tiny-imagenet-200", "val")
    tinyimagenet_dataset = None
    if os.path.exists(tinyimagenet_path):
        from torchvision.datasets import ImageFolder
        tinyimagenet_dataset = ImageFolder(root=tinyimagenet_path, transform=tinyimagenet_transform)
        print(f"  ✅ TinyImageNet loaded from {tinyimagenet_path} ({len(tinyimagenet_dataset)} samples)")
    else:
        print(f"  ⚠️ TinyImageNet not found at {tinyimagenet_path} - will return None for ood_loader3")

    cifar10_trainloader = DataLoader(cifar10_train_dataset, batch_size=batch_size,
                                     num_workers=num_workers, shuffle=True, pin_memory=pin,
                                     persistent_workers=allow_persist)
    cifar10_validloader = DataLoader(cifar10_valid_dataset, batch_size=batch_size,
                                     num_workers=num_workers, shuffle=False, pin_memory=pin,
                                     persistent_workers=allow_persist)
    cifar10_testloader = DataLoader(cifar10_test_dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False, pin_memory=pin,
                                    persistent_workers=allow_persist)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, pin_memory=pin, persistent_workers=allow_persist)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=batch_size, num_workers=num_workers,
                                 shuffle=False, pin_memory=pin, persistent_workers=allow_persist)
    
    # TinyImageNet loader (may be None if dataset not found)
    tinyimagenet_loader = None
    if tinyimagenet_dataset is not None:
        tinyimagenet_loader = DataLoader(tinyimagenet_dataset, batch_size=batch_size, 
                                         num_workers=num_workers, shuffle=False, 
                                         pin_memory=pin, persistent_workers=allow_persist)

    print("Enhanced GEM CIFAR-10 data loaders created successfully")
    return cifar10_trainloader, cifar10_validloader, cifar10_testloader, svhn_loader, cifar100_loader, tinyimagenet_loader


def dataloaders_cifar100(batch_size, val_size, data_dir):
    print("Creating Enhanced GEM CIFAR-100 data loaders...")
    split_seed = int(os.environ.get("GEM_VAL_SEED", "42"))
    rng = np.random.RandomState(split_seed)

    num_workers, allow_persist = _dl_worker_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = _pin(device)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    resize = transforms.Resize((32, 32))
    tensorize = transforms.ToTensor()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.RandomRotation(degrees=5),
        tensorize, normalize
    ])

    valid_transform = transforms.Compose([tensorize, normalize])
    test_transform = transforms.Compose([tensorize, normalize])
    mnist2cifar = transforms.Compose([resize, transforms.Grayscale(3)])
    tin2cifar = transforms.Compose([resize, tensorize])

    cifar100_train_dataset_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    cifar100_valid_dataset_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=valid_transform)

    num_train = len(cifar100_train_dataset_full)
    indices = np.arange(num_train)
    rng.shuffle(indices)
    split = int(np.floor(val_size * num_train))

    cifar100_train_dataset = Subset(cifar100_train_dataset_full, indices[split:])
    cifar100_valid_dataset = Subset(cifar100_valid_dataset_full, indices[:split])

    cifar100_test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    svhn_dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=test_transform)
    fmnist_test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=mnist2cifar)
    tin_test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "tiny-imagenet-200/test"), transform=tin2cifar)

    cifar100_trainloader = DataLoader(cifar100_train_dataset, batch_size=batch_size,
                                      num_workers=num_workers, shuffle=True, pin_memory=pin,
                                      persistent_workers=allow_persist)
    cifar100_validloader = DataLoader(cifar100_valid_dataset, batch_size=batch_size,
                                      num_workers=num_workers, shuffle=False, pin_memory=pin,
                                      persistent_workers=allow_persist)
    cifar100_testloader = DataLoader(cifar100_test_dataset, batch_size=batch_size,
                                     num_workers=num_workers, shuffle=False, pin_memory=pin,
                                     persistent_workers=allow_persist)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin, persistent_workers=allow_persist)
    fmnist_loader = DataLoader(fmnist_test_dataset, batch_size=batch_size, num_workers=num_workers,
                               pin_memory=pin, persistent_workers=allow_persist)
    tin_loader = DataLoader(tin_test_dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin, persistent_workers=allow_persist)

    return cifar100_trainloader, cifar100_validloader, cifar100_testloader, svhn_loader, fmnist_loader


# === Added helpers for Dirichlet & MIX predictive means ===
import torch
import torch.nn.functional as F


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    alpha0 = alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
    p = alpha / alpha0
    p = p.clamp_min(1e-8)
    return p / p.sum(dim=1, keepdim=True)


def mob_predictive_probs(alpha_list, mixture_weights):
    if torch.is_tensor(alpha_list):
        if alpha_list.dim() == 3:
            alpha_list = [alpha_list[k] for k in range(alpha_list.size(0))]
        else:
            raise ValueError("alpha_list tensor must be of shape (K,B,C) if provided as a tensor")

    means = []
    for a in alpha_list:
        a0 = a.sum(dim=1, keepdim=True).clamp_min(1e-8)
        means.append(a / a0)

    means = torch.stack(means, dim=1)

    pi = mixture_weights
    if pi.dim() != 2 or pi.size(1) != means.size(1):
        raise ValueError(f"mixture_weights must be (B,K); got {tuple(pi.shape)} with K={means.size(1)} expected")

    pi = pi / (pi.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pi = pi.unsqueeze(-1)

    p = (pi * means).sum(dim=1)

    p = p.clamp_min(1e-8)
    p = p / p.sum(dim=1, keepdim=True)
    return p

# ===== Robust energy -> confidence mapping (with three safeguards) =====
import torch
import numpy as _torch_local
import numpy as _np_local

def energy_to_confidence_robust(E, Emin, Emax, logits=None, eps: float = 1e-6):
    """
    s = Clip(1 - (E - Emin) / (Emax - Emin), 0, 1)
    with robust safeguards:
      1) percentile-based range if needed (handled by caller when passing Emin/Emax)
      2) guard for tight range (fallback to -logsumexp(logits) if available)
      3) clamp to [Emin, Emax] before ratio
    Inputs can be tensors or scalars. Returns a torch.Tensor on same device as E.
    """
    if not _torch_local.is_tensor(E):
        E = _torch_local.tensor(E, dtype=_torch_local.float32)
    if not _torch_local.is_tensor(Emin):
        Emin = _torch_local.tensor(Emin, dtype=_torch_local.float32, device=E.device)
    if not _torch_local.is_tensor(Emax):
        Emax = _torch_local.tensor(Emax, dtype=_torch_local.float32, device=E.device)

    rng = (Emax - Emin).abs()
    tight = (rng < eps) | _torch_local.isnan(rng) | _torch_local.isinf(rng)

    if tight.any():
        if logits is not None:
            if not _torch_local.is_tensor(logits):
                logits = _torch_local.tensor(logits, dtype=_torch_local.float32, device=E.device)
            E_log = -_torch_local.logsumexp(logits, dim=1)
            qmin = _torch_local.quantile(E_log, 0.01)
            qmax = _torch_local.quantile(E_log, 0.99)
            rng_f = (qmax - qmin).abs().clamp_min(eps)
            E_clamped = E_log.clamp(qmin, qmax)
            s = 1.0 - (E_clamped - qmin) / rng_f
            return s.clamp(0.0, 1.0)

        # Final fallback: widen the range slightly to avoid zero division
        Emin_s = (Emin - 0.5).to(E.dtype).to(E.device)
        Emax_s = (Emax + 0.5).to(E.dtype).to(E.device)
        rng_s = (Emax_s - Emin_s).clamp_min(eps)
        E_clamped = E.clamp(Emin_s, Emax_s)
        s = 1.0 - (E_clamped - Emin_s) / rng_s
        return s.clamp(0.0, 1.0)

    E_clamped = E.clamp(Emin.to(E.device), Emax.to(E.device))
    s = 1.0 - (E_clamped - Emin.to(E.device)) / rng
    return s.clamp(0.0, 1.0)
