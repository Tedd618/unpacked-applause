"""Inference-only copy of the PANNs ``Cnn14_DecisionLevelMax`` sound-event-detection model.

Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT License,
Copyright (c) 2019 Qiuqiang Kong). Kong et al., "PANNs: Large-Scale Pretrained Audio
Neural Networks for Audio Pattern Recognition", IEEE/ACM TASLP 2020.

Training-only pieces (SpecAugment, mixup, dropout) are removed; the parameter names are
unchanged so the official checkpoint loads with ``strict=True``. ``forward`` returns the
segment-wise probabilities (one row per 32 STFT hops = 0.32 s at 32 kHz) instead of the
interpolated frame-wise output, which is all the downstream pipeline needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

SAMPLE_RATE = 32_000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN, FMAX = 50, 14_000
CLASSES_NUM = 527
DOWNSAMPLE = 32  # five 2x2 average pools in the time axis
SEGMENT_SAMPLES = HOP_SIZE * DOWNSAMPLE  # 10 240 samples = 0.32 s


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int]) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14DecisionLevelMax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE, n_fft=WINDOW_SIZE, n_mels=MEL_BINS, fmin=FMIN, fmax=FMAX,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(MEL_BINS)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, CLASSES_NUM, bias=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(batch, samples) float waveform in [-1, 1] -> (batch, segments, 527) probabilities."""
        x = self.spectrogram_extractor(waveform)  # (B, 1, frames, freq)
        x = self.logmel_extractor(x)  # (B, 1, frames, mel)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3,
                      self.conv_block4, self.conv_block5):
            x = block(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)  # (B, 2048, segments)
        x = F.max_pool1d(x, 3, 1, 1) + F.avg_pool1d(x, 3, 1, 1)
        x = F.relu_(self.fc1(x.transpose(1, 2)))
        return torch.sigmoid(self.fc_audioset(x))


def load_model(checkpoint_path, device: str) -> Cnn14DecisionLevelMax:
    model = Cnn14DecisionLevelMax()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:  # older pickles carry non-tensor training state
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(device)
