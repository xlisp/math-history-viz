"""Spectrogram — short-time Fourier transform.

The same Fourier idea (1807) applied to a sliding window gives a spectrogram:
which frequencies are present *when*. This is how Shazam recognizes songs
and how speech recognizers see phonemes.

This script synthesizes a signal (chirp + tone + whistle) so it runs without
needing a microphone or audio file.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sample_rate = 8000
duration = 4.0
t = torch.linspace(0, duration, int(sample_rate * duration))

chirp = torch.sin(2 * math.pi * (200 + 600 * t / duration) * t)
tone = 0.5 * torch.sin(2 * math.pi * 1000 * t) * ((t > 1.0) & (t < 2.5)).float()
whistle = 0.7 * torch.sin(2 * math.pi * 2200 * t) * (t > 3.0).float()
signal = chirp + tone + whistle

n_fft = 512
hop = 128
spec = torch.stft(
    signal,
    n_fft=n_fft,
    hop_length=hop,
    return_complex=True,
    window=torch.hann_window(n_fft),
)
mag = spec.abs()
log_mag = torch.log1p(mag)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(t.numpy(), signal.numpy(), lw=0.3)
axes[0].set_xlabel("t (s)")
axes[0].set_ylabel("amplitude")
axes[0].set_title("Synthesized signal: chirp + tone (1–2.5 s) + whistle (3+ s)")

freqs = torch.linspace(0, sample_rate / 2, mag.shape[0])
times = torch.arange(mag.shape[1]) * hop / sample_rate
axes[1].imshow(
    log_mag.numpy(),
    origin="lower",
    aspect="auto",
    extent=[times[0].item(), times[-1].item(), freqs[0].item(), freqs[-1].item()],
    cmap="magma",
)
axes[1].set_xlabel("t (s)")
axes[1].set_ylabel("frequency (Hz)")
axes[1].set_title("Spectrogram (log magnitude STFT)")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
