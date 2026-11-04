"""
Test domain for CatGAN-26 (128x128 AFHQ-aligned cat faces), with the same interface as the 2018
anime tests (exp3: ATTRS, attributes; exp4: run, eye_mask, score), so wpgit and the exp3/exp4 tools
can test cat repositories unchanged. Regions come from the mean AFHQ cat (results/exp11_out/mean_cat.png).
"""
import numpy as np

import exp4_anime_eye_circuit as e4

def lum(x): return x @ np.array([0.299, 0.587, 0.114], np.float32)

EYES = np.zeros((128, 128), bool); EYES[56:74, 36:58] = EYES[56:74, 70:92] = True
FACE = np.zeros((128, 128), bool); FACE[40:112, 24:104] = True; FACE &= ~EYES
BACK = np.zeros((128, 128), bool); BACK[:12, 50:78] = BACK[-12:, :12] = BACK[-12:, -12:] = True
# Ice-blue irises that keep the original shading: luminance times a blue tint (a linear map, like SWAP_RG)
TARGET = np.outer([0.55, 0.85, 1.55], [0.299, 0.587, 0.114]).astype(np.float32)

ATTRS = {
    "fur_warmth":    lambda x: (x[:, FACE, 0] - x[:, FACE, 2]).mean(1),
    "fur_lightness": lambda x: lum(x[:, FACE]).mean(1),
    "background":    lambda x: lum(x[:, BACK]).mean(1),
    "eye_green":     lambda x: (x[:, EYES, 1] - (x[:, EYES, 0] + x[:, EYES, 2]) / 2).mean(1),
    "saturation":    lambda x: (x.max(-1) - x.min(-1)).mean((1, 2)),
}

def attributes(img):
    x = np.clip(img, 0, 1)
    return np.stack([f(x) for f in ATTRS.values()], 1)

def eye_mask(x):
    """Iris pixels: coloured (amber / yellow / green), brighter than the pupil, inside the eye boxes.
    Defined on base images only; edited images are scored at the same pixels."""
    mx, mn = x.max(-1), x.min(-1); h = e4.hue(x)
    return EYES[None] & (mx - mn > 0.10) & (mx > 0.25) & (h > 25) & (h < 170)

def score(x1, x0, m0, target=TARGET): return e4.score(x1, x0, m0, target)

def run(z, W, ops, chunk=100, keep=False):
    """CatGAN-26 forward pass: on the GPU (full fp32) when PyTorch + CUDA are available, else the
    exp4 NumPy engine. keep=True (activations of the last convs) always uses the NumPy engine."""
    if not keep:
        try:
            import torch, modern_np as mnp
            if torch.cuda.is_available():
                torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
                T = mnp.to_torch(W); out = []
                with torch.no_grad():
                    for s in range(0, len(z), chunk):
                        x = mnp.catgan_torch(torch.from_numpy(np.asarray(z[s:s + chunk], np.float32)).cuda(), T)
                        out.append(x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
                return np.concatenate(out)
        except ImportError:
            pass
    return e4.run(z, W, ops, chunk, keep)
