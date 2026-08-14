"""
Zero experiment: does dense-feature collapse reproduce at small scale?

A compact DINO + iBOT trainer (CLS-level self-distillation + masked patch-level
self-distillation), instrumented so that dense-feature quality is probed on a
fixed held-out batch every N epochs and written to a CSV.

This script does NOT implement spectral anchoring. It exists only to answer the
go/no-go question: if `correspondence_acc` never degrades over a long schedule,
there is no collapse to repair at this scale and the spectral-anchoring project
should stop here. If it does degrade, check whether `hf_energy` falls with it.

Smoke test (no data needed, runs on CPU/MPS in ~a minute):
    python3 tools/spectral/train_ssl.py --smoke

Real run (ImageFolder layout, any class structure — labels are unused):
    python3 tools/spectral/train_ssl.py \
        --data /path/to/imagenet100/train \
        --epochs 400 --batch-size 128 --out runs/pilot
"""

import os
import csv
import json
import math
import time
import copy
import argparse

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from dense_metrics import evaluate


def _patch_timm_for_mps():
    """MPS has no backward for antialiased bicubic upsampling, which timm uses to
    resample positional embeddings for the smaller local crops. Antialiasing a
    14x14 position grid changes nothing that matters, so drop it on MPS."""
    import timm.models.vision_transformer as vit_mod

    original = vit_mod.resample_abs_pos_embed

    def no_antialias(*a, **kw):
        kw["antialias"] = False
        return original(*a, **kw)

    vit_mod.resample_abs_pos_embed = no_antialias


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class MultiCrop:
    """DINO multi-crop: 2 global views + `n_local` local views."""

    def __init__(self, global_size, local_size, n_local):
        self.n_local = n_local
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.GaussianBlur(5, (0.1, 2.0)),
            *tail])
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(128, p=0.2),
            *tail])
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.5),
            *tail])

    def __call__(self, img):
        return ([self.global1(img), self.global2(img)]
                + [self.local(img) for _ in range(self.n_local)])


class SyntheticImages(Dataset):
    """Structured noise for the smoke test. Not a substitute for real data."""

    def __init__(self, n, size, transform):
        self.n, self.size, self.transform = n, size, transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        small = torch.rand(3, 8, 8, generator=g)
        img = F.interpolate(small[None], size=(self.size, self.size),
                            mode="bicubic", align_corners=False)[0].clamp(0, 1)
        img = img + 0.15 * torch.rand(3, self.size, self.size, generator=g)
        arr = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        return self.transform(Image.fromarray(arr)), 0


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class Backbone(nn.Module):
    """timm ViT that exposes CLS + patch tokens and supports iBOT mask tokens."""

    def __init__(self, arch, img_size, patch_size):
        super().__init__()
        self.net = timm.create_model(
            arch, pretrained=False, num_classes=0,
            img_size=img_size, patch_size=patch_size, dynamic_img_size=True)
        self.embed_dim = self.net.embed_dim
        self.num_prefix = self.net.num_prefix_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        net = self.net
        x = net.patch_embed(x)
        shape = x.shape
        flat = x.reshape(shape[0], -1, shape[-1])
        if mask is not None:
            flat = torch.where(mask.unsqueeze(-1),
                               self.mask_token.to(flat.dtype), flat)
        x = flat.reshape(shape)
        x = net._pos_embed(x)
        x = net.patch_drop(x)
        x = net.norm_pre(x)
        x = net.blocks(x)
        x = net.norm(x)
        return x[:, 0], x[:, self.num_prefix:]


class Head(nn.Module):
    """DINO projection head."""

    def __init__(self, in_dim, out_dim, hidden=512, bottleneck=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck))
        last = nn.Linear(bottleneck, out_dim, bias=False)
        self.last = nn.utils.parametrizations.weight_norm(last)
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class Model(nn.Module):
    def __init__(self, arch, img_size, patch_size, out_dim):
        super().__init__()
        self.backbone = Backbone(arch, img_size, patch_size)
        d = self.backbone.embed_dim
        self.cls_head = Head(d, out_dim)
        self.patch_head = Head(d, out_dim)


class DistillLoss(nn.Module):
    """Centred + sharpened cross-entropy, shared by the CLS and patch objectives."""

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits, teacher_logits, teacher_temp):
        s = (student_logits / self.student_temp).log_softmax(dim=-1)
        t = ((teacher_logits - self.center) / teacher_temp).softmax(dim=-1).detach()
        return -(t * s).sum(dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_logits):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1 - self.center_momentum)


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #

def cosine_schedule(base, final, total, warmup=0, warmup_start=0.0):
    """Linear warmup then cosine decay. Warmup is clamped so short runs still work."""
    total = max(int(total), 1)
    warmup = max(0, min(int(warmup), total - 1))
    warm = ([] if warmup == 0
            else list(torch.linspace(warmup_start, base, warmup).numpy()))
    steps = torch.arange(total - warmup)
    cos = final + 0.5 * (base - final) * (1 + torch.cos(math.pi * steps / len(steps)))
    return list(warm) + list(cos.numpy())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="ImageFolder root (labels unused)")
    p.add_argument("--probe-data", default=None,
                   help="ImageFolder root for the probe batch; defaults to --data. "
                        "Point it at a held-out split for a clean measurement.")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run")
    p.add_argument("--arch", default="vit_tiny_patch16_224")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--local-size", type=int, default=64)
    p.add_argument("--n-local", type=int, default=6)
    p.add_argument("--out-dim", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=None,
                   help="default 400, or 3 under --smoke")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4, help="per 256 samples")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--teacher-temp", type=float, default=0.07)
    p.add_argument("--teacher-temp-warmup", type=int, default=30)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--probe-batch", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", default="auto", choices=["auto", "on", "off"],
                   help="bfloat16 autocast; auto = on for CUDA only")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/ckpt.pt if present (spot instances)")
    p.add_argument("--out", default="runs/pilot")
    args = p.parse_args()

    if args.smoke:
        args.batch_size, args.n_local = 8, 2
        args.probe_every, args.probe_batch, args.workers = 1, 8, 0
        args.out_dim, args.warmup_epochs = 256, 1
        args.epochs = args.epochs or 3
    args.epochs = args.epochs or 400

    # MPS intermittently produces non-finite activations in this ViT forward
    # (~50% of smoke runs, vs. 0% on CPU). The smoke test is tiny, so pin it to
    # CPU unless a device was named explicitly; real runs go to CUDA anyway.
    if args.smoke and args.device == "auto":
        args.device = "cpu"

    device = pick_device(args.device)
    if device.type == "mps":
        _patch_timm_for_mps()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2)
    print(f"device={device}  out={args.out}")

    # ---- data -------------------------------------------------------------- #
    crop = MultiCrop(args.img_size, args.local_size, args.n_local)
    if args.smoke:
        train_set = SyntheticImages(64, 160, crop)
    elif args.data:
        train_set = ImageFolder(args.data, transform=crop)
    else:
        raise SystemExit("pass --data <imagefolder root> or --smoke")
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))
    print(f"train images: {len(train_set)}  steps/epoch: {len(loader)}")

    # fixed held-out batch for probing, kept at native resolution
    probe_tf = transforms.Compose([
        transforms.Resize(args.img_size * 2),
        transforms.CenterCrop(args.img_size * 2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    probe_set = (SyntheticImages(args.probe_batch, 160, probe_tf) if args.smoke
                 else ImageFolder(args.probe_data or args.data, transform=probe_tf))
    probe_idx = torch.linspace(0, len(probe_set) - 1, args.probe_batch).long()
    probe_images = torch.stack([probe_set[int(i)][0] for i in probe_idx]).to(device)

    # ---- model ------------------------------------------------------------- #
    student = Model(args.arch, args.img_size, args.patch_size, args.out_dim).to(device)
    teacher = copy.deepcopy(student).to(device)
    for q in teacher.parameters():
        q.requires_grad = False
    n_params = sum(q.numel() for q in student.backbone.parameters())
    print(f"backbone params: {n_params / 1e6:.1f}M  "
          f"tokens/global crop: {(args.img_size // args.patch_size) ** 2}")

    cls_loss = DistillLoss(args.out_dim).to(device)
    patch_loss = DistillLoss(args.out_dim).to(device)

    scaled_lr = args.lr * args.batch_size / 256
    total_steps = args.epochs * len(loader)
    lr_sched = cosine_schedule(scaled_lr, 1e-6, total_steps,
                               warmup=args.warmup_epochs * len(loader))
    wd_sched = cosine_schedule(0.04, 0.4, total_steps)
    mom_sched = cosine_schedule(0.996, 1.0, total_steps)
    temp_sched = (list(torch.linspace(0.04, args.teacher_temp,
                                      args.teacher_temp_warmup * len(loader)).numpy())
                  + [args.teacher_temp] * total_steps)

    opt = torch.optim.AdamW(student.parameters())

    use_amp = (args.amp == "on") or (args.amp == "auto" and device.type == "cuda")
    autocast = ((lambda: torch.autocast(device.type, dtype=torch.bfloat16))
                if use_amp else nullcontext)
    print(f"bfloat16 autocast: {use_amp}")

    def encode(views):
        return teacher.backbone(views)[1]

    csv_path = os.path.join(args.out, "probe.csv")
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    fields = ["epoch", "step", "loss", "correspondence_acc",
              "correspondence_pos_baseline", "patch_sim_entropy",
              "hf_energy", "gram_drift"]

    ref_tokens = None
    start_epoch, step = 0, 0
    n_global = 2

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        opt.load_state_dict(ck["opt"])
        cls_loss.load_state_dict(ck["cls_loss"])
        patch_loss.load_state_dict(ck["patch_loss"])
        ref_tokens = ck["ref_tokens"]
        if ref_tokens is not None:
            ref_tokens = ref_tokens.to(device)
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"resumed from {ckpt_path} at epoch {start_epoch}")
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        student.train()
        t0, running = time.time(), 0.0

        for views, _ in loader:
            for group in opt.param_groups:
                group["lr"] = lr_sched[step]
                group["weight_decay"] = wd_sched[step]
            t_temp = temp_sched[step]

            views = [v.to(device, non_blocking=True) for v in views]
            globals_ = torch.cat(views[:n_global])

            with torch.no_grad(), autocast():
                t_cls, t_patch = teacher.backbone(globals_)
                t_cls_logits = teacher.cls_head(t_cls).float()
                t_patch_logits = teacher.patch_head(t_patch).float()

            n_tokens = t_patch.shape[1]
            mask = (torch.rand(globals_.shape[0], n_tokens, device=device)
                    < args.mask_ratio)
            mask[:, 0] = True                     # guarantee a target per sample

            # `globals_` is cat(view0, view1), so mask rows split the same way
            mask_per_view = mask.view(n_global, -1, n_tokens)

            with autocast():
                s_cls_logits, s_patch_logits = [], []
                for i, view in enumerate(views):
                    if i < n_global:
                        c, pt = student.backbone(view, mask=mask_per_view[i])
                        s_patch_logits.append(student.patch_head(pt).float())
                    else:
                        c, _ = student.backbone(view)
                    s_cls_logits.append(student.cls_head(c).float())

            # CLS objective: every student view against each *other* teacher view
            t_cls_split = t_cls_logits.chunk(n_global)
            terms = []
            for ti, t_logit in enumerate(t_cls_split):
                for si, s_logit in enumerate(s_cls_logits):
                    if si == ti:
                        continue
                    terms.append(cls_loss(s_logit, t_logit, t_temp).mean())
            loss_cls = torch.stack(terms).mean()

            # iBOT objective: masked patches, same view
            s_patch_all = torch.cat(s_patch_logits)
            m = mask.reshape(-1)
            loss_patch = patch_loss(
                s_patch_all.reshape(-1, args.out_dim)[m],
                t_patch_logits.reshape(-1, args.out_dim)[m], t_temp).mean()

            loss = loss_cls + loss_patch
            if not torch.isfinite(loss):
                parts = {
                    "loss_cls": loss_cls, "loss_patch": loss_patch,
                    "teacher_cls_logits": t_cls_logits,
                    "teacher_patch_logits": t_patch_logits,
                    "student_cls_logits": torch.cat(s_cls_logits),
                    "student_patch_logits": s_patch_all,
                    "cls_center": cls_loss.center,
                    "patch_center": patch_loss.center,
                }
                bad = [k for k, v in parts.items() if not torch.isfinite(v).all()]
                raise SystemExit(
                    f"non-finite loss at step {step} (epoch {epoch}); "
                    f"first non-finite tensors: {bad or 'none — loss only'}\n"
                    f"  device={device}. On MPS this is a known numerical "
                    f"instability; rerun with --device cpu to confirm, and "
                    f"prefer CUDA for real runs.")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if epoch < 1:                          # DINO's freeze-last-layer trick
                for head in (student.cls_head, student.patch_head):
                    for q in head.last.parameters():
                        q.grad = None
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()

            cls_loss.update_center(t_cls_logits)
            patch_loss.update_center(t_patch_logits.reshape(-1, args.out_dim))

            with torch.no_grad():
                m_ema = mom_sched[step]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m_ema).add_(ps.detach(), alpha=1 - m_ema)

            running += loss.item()
            step += 1

        avg = running / len(loader)
        print(f"epoch {epoch:4d}  loss {avg:.4f}  {time.time() - t0:.1f}s", flush=True)

        if epoch % args.probe_every == 0 or epoch == args.epochs - 1:
            teacher.eval()
            stats, tokens = evaluate(encode, probe_images, args.img_size,
                                     ref_tokens=ref_tokens)
            if ref_tokens is None:
                ref_tokens = tokens          # provisional reference, see below

            # Keep every probe's tokens. The live `gram_drift` column is measured
            # against the FIRST probe, which is a near-random model — so it mostly
            # tracks "how far from init", which rises during healthy training and
            # says little about degradation. The reference that actually matters is
            # the checkpoint where dense quality peaks, and that is only knowable
            # afterwards. Saving the tokens (~12 MB per probe in fp16) lets
            # regram.py recompute the column against any reference post hoc.
            tok_dir = os.path.join(args.out, "probe_tokens")
            os.makedirs(tok_dir, exist_ok=True)
            torch.save(tokens.half().cpu(),
                       os.path.join(tok_dir, f"epoch_{epoch:04d}.pt"))
            row = {"epoch": epoch, "step": step, "loss": round(avg, 4),
                   **{k: round(v, 5) for k, v in stats.items()}}
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print("  probe " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()),
                  flush=True)
            torch.save({"epoch": epoch, "step": step,
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "opt": opt.state_dict(),
                        "cls_loss": cls_loss.state_dict(),
                        "patch_loss": patch_loss.state_dict(),
                        "ref_tokens": ref_tokens.cpu() if ref_tokens is not None else None},
                       ckpt_path)

    print(f"\ndone. probe curve -> {csv_path}")
    print("read it as: correspondence_acc DOWN = collapse reproduced;")
    print("            hf_energy falling with it = spectral hypothesis supported.")


if __name__ == "__main__":
    main()
