"""1D ResNet for 12-lead ECG (12 x 2500 @ 250 Hz) with 3 heads: K+ value, hyperkalemia, hypokalemia."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.down = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.down is None else self.down(x)))


class ResNet1D(nn.Module):
    def __init__(self, width=64, kernel=7, dropout=0.3, n_leads=12):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(n_leads, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.ReLU(),
                                  nn.MaxPool1d(3, 2, 1))
        chans = [w, 2 * w, 4 * w, 8 * w]
        layers, cin = [], w
        for i, c in enumerate(chans):
            layers += [Block(cin, c, 1 if i == 0 else 2, kernel), Block(c, c, 1, kernel)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * cin, 3)          # [k_z, logit_hyper, logit_hypo]

    def forward(self, x, return_maps=False):
        a = self.features(self.stem(x))
        z = torch.cat([a.mean(-1), a.amax(-1)], 1)
        out = self.head(self.drop(z))
        return (out, a) if return_maps else out


@torch.enable_grad()
def gradcam(model, x, target=1):
    """Grad-CAM over the last residual stage -> (B, T) saliency upsampled to input length, each row sums to 1."""
    model.eval()
    x = x.clone().requires_grad_(False)
    out, a = model(x, return_maps=True)
    a.retain_grad()
    out[:, target].sum().backward()
    w = a.grad.mean(-1, keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[:, 0]
    tot = cam.sum(-1, keepdim=True)
    cam = torch.where(tot > 0, cam / tot.clamp_min(1e-12), torch.full_like(cam, 1.0 / cam.shape[-1]))  # all-zero -> uniform
    model.zero_grad(set_to_none=True)
    return cam.detach()
