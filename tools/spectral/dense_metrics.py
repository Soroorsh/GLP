"""
Label-free dense-feature quality metrics for the "zero experiment".

The question the zero experiment answers is:
    (Q1) can dense-feature collapse be reproduced at small scale at all?
    (Q2) does a spectral statistic track that collapse?

Q1 needs a measure of dense quality. Q2 needs the spectral statistic. Everything
here is label-free so no segmentation dataset is required.

Metrics
-------
gram_drift            ‖XXᵀ − X_ref X_refᵀ‖_F on L2-normalised patch tokens. This
                      is literally DINOv3's Gram-anchoring objective, used as a
                      *measurement* against an early-checkpoint reference rather
                      than as a loss. Rises as dense structure drifts.
patch_sim_entropy     Mean normalised entropy of each patch's similarity
                      distribution over the other patches. Over-smoothed /
                      collapsed maps are flatter, so this rises toward 1.
hf_energy             Fraction of 2-D spectral energy of the patch-feature map
                      above `cutoff` x Nyquist. This is the quantity a
                      "spectral anchoring" loss would regularise. Expected to
                      FALL if collapse is loss of high-frequency detail.
correspondence_acc    Cross-view patch matching accuracy under a known crop
                      geometry. The "does it actually still work" number: real
                      degradation of dense features shows up here.

Collapse signature to look for, jointly:
    correspondence_acc down, gram_drift up, patch_sim_entropy up, hf_energy down.
If hf_energy does not move with correspondence_acc, the spectral-anchoring
hypothesis is not supported and the project should stop at this stage.
"""

import math

import torch
import torch.nn.functional as F


def _grid_side(num_tokens):
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"{num_tokens} patch tokens is not a square grid")
    return side


@torch.no_grad()
def gram_drift(tokens, ref_tokens):
    """Frobenius distance between patch Gram matrices, normalised by token count."""
    x = F.normalize(tokens.float(), dim=-1)
    r = F.normalize(ref_tokens.float(), dim=-1)
    diff = (x @ x.transpose(1, 2)) - (r @ r.transpose(1, 2))
    return (diff.pow(2).sum(dim=(1, 2)).sqrt() / x.shape[1]).mean().item()


@torch.no_grad()
def patch_sim_entropy(tokens, temperature=0.1):
    """Normalised entropy of per-patch similarity distributions. Higher = flatter."""
    x = F.normalize(tokens.float(), dim=-1)
    sim = x @ x.transpose(1, 2)
    n = sim.shape[1]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    p = (sim / temperature).softmax(dim=-1)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    return (ent / math.log(n - 1)).mean().item()


@torch.no_grad()
def hf_energy(tokens, cutoff=0.5):
    """Fraction of 2-D FFT energy of the feature map above `cutoff` x Nyquist."""
    b, n, d = tokens.shape
    g = _grid_side(n)
    fmap = tokens.float().transpose(1, 2).reshape(b, d, g, g)
    fmap = fmap - fmap.mean(dim=(2, 3), keepdim=True)          # drop DC
    spec = torch.fft.fftshift(
        torch.fft.fft2(fmap, norm="ortho"), dim=(-2, -1)).abs().pow(2)
    freq = torch.fft.fftshift(torch.fft.fftfreq(g, device=tokens.device))
    radius = (freq[:, None].pow(2) + freq[None, :].pow(2)).sqrt() / 0.5
    mask = radius >= cutoff
    high = spec[..., mask].sum(dim=(1, 2))
    total = spec.sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return (high / total).mean().item()


def sample_crop_pairs(batch_size, height, width, generator=None):
    """Two overlapping square boxes per image, as (y, x, side) float tensors.

    The scale ratio and translation are deliberately large. If the two views were
    near-identical, a model could score well by matching positional embeddings
    alone; `correspondence_acc` reports the position-only baseline so that this
    stays visible rather than silently inflating the metric.
    """
    base = float(min(height, width))
    rand = lambda: torch.rand(batch_size, generator=generator)

    side_a = (0.45 + 0.25 * rand()) * base
    y_a = rand() * (height - side_a).clamp_min(0)
    x_a = rand() * (width - side_a).clamp_min(0)

    side_b = (side_a * (0.60 + 1.00 * rand())).clamp(max=base)
    y_b = (y_a + (rand() - 0.5) * 1.0 * side_a).clamp_min(0)
    x_b = (x_a + (rand() - 0.5) * 1.0 * side_a).clamp_min(0)
    y_b = torch.minimum(y_b, (height - side_b).clamp_min(0))
    x_b = torch.minimum(x_b, (width - side_b).clamp_min(0))

    return (y_a, x_a, side_a), (y_b, x_b, side_b)


def _crop_resize(images, box, out_size):
    """Crop each image by its box and resize to out_size, via grid_sample."""
    y, x, side = box
    b, _, h, w = images.shape
    dev = images.device
    y, x, side = y.to(dev), x.to(dev), side.to(dev)

    lin = (torch.arange(out_size, device=dev) + 0.5) / out_size        # [0,1)
    gy = y[:, None] + lin[None, :] * side[:, None]                     # [B,out]
    gx = x[:, None] + lin[None, :] * side[:, None]
    ny = (gy / h) * 2 - 1
    nx = (gx / w) * 2 - 1
    grid = torch.stack([
        nx[:, None, :].expand(b, out_size, out_size),
        ny[:, :, None].expand(b, out_size, out_size),
    ], dim=-1)
    return F.grid_sample(images, grid, mode="bilinear", align_corners=False)


@torch.no_grad()
def correspondence_acc(encode, images, out_size, tolerance=1, seed=0):
    """Cross-view patch-matching accuracy under known crop geometry.

    `encode(view) -> [B, N, D]` patch tokens for a batch of cropped views.
    A patch in view A is correct if its nearest neighbour in view B lands within
    `tolerance` grid cells of the geometrically implied location.
    """
    b, _, h, w = images.shape
    gen = torch.Generator().manual_seed(seed)
    box_a, box_b = sample_crop_pairs(b, h, w, generator=gen)

    tok_a = encode(_crop_resize(images, box_a, out_size))
    tok_b = encode(_crop_resize(images, box_b, out_size))
    g = _grid_side(tok_a.shape[1])
    dev = tok_a.device

    # patch centres of view A, in [0,1) view-A coordinates
    centres = (torch.arange(g, device=dev) + 0.5) / g
    ca_y = centres[:, None].expand(g, g).reshape(-1)                   # [N]
    ca_x = centres[None, :].expand(g, g).reshape(-1)

    y_a, x_a, s_a = (t.to(dev) for t in box_a)
    y_b, x_b, s_b = (t.to(dev) for t in box_b)

    # view A -> original image -> view B
    img_y = y_a[:, None] + ca_y[None, :] * s_a[:, None]
    img_x = x_a[:, None] + ca_x[None, :] * s_a[:, None]
    vb_y = (img_y - y_b[:, None]) / s_b[:, None]
    vb_x = (img_x - x_b[:, None]) / s_b[:, None]

    valid = (vb_y >= 0) & (vb_y < 1) & (vb_x >= 0) & (vb_x < 1)
    if valid.sum() == 0:
        return {"correspondence_acc": float("nan"),
                "correspondence_pos_baseline": float("nan")}

    true_row = (vb_y * g).floor().clamp(0, g - 1)
    true_col = (vb_x * g).floor().clamp(0, g - 1)

    sim = F.normalize(tok_a.float(), dim=-1) @ F.normalize(tok_b.float(), dim=-1).transpose(1, 2)
    pred = sim.argmax(dim=-1)                                          # [B, N]

    def score(pred_row, pred_col):
        err = torch.maximum((pred_row - true_row).abs(), (pred_col - true_col).abs())
        return ((err <= tolerance) & valid).sum().div(valid.sum()).item()

    # position-only control: answer "same grid cell as in view A", ignoring content
    same_row = torch.arange(g, device=dev)[:, None].expand(g, g).reshape(-1)
    same_col = torch.arange(g, device=dev)[None, :].expand(g, g).reshape(-1)

    return {
        "correspondence_acc": score(pred // g, pred % g),
        "correspondence_pos_baseline": score(same_row[None, :].expand_as(true_row),
                                             same_col[None, :].expand_as(true_col)),
    }


@torch.no_grad()
def evaluate(encode, images, out_size, ref_tokens=None, seed=0):
    """Run every metric on one fixed batch. Returns a dict (+ tokens for reuse)."""
    box, _ = sample_crop_pairs(images.shape[0], images.shape[2], images.shape[3],
                               generator=torch.Generator().manual_seed(seed))
    tokens = encode(_crop_resize(images, box, out_size))
    stats = {
        **correspondence_acc(encode, images, out_size, seed=seed),
        "patch_sim_entropy": patch_sim_entropy(tokens),
        "hf_energy": hf_energy(tokens),
    }
    stats["gram_drift"] = (gram_drift(tokens, ref_tokens)
                           if ref_tokens is not None else 0.0)
    return stats, tokens
