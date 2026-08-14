# Zero experiment — can dense-feature collapse be reproduced at small scale?

## Why this exists

DINOv3 introduces **Gram anchoring** to fix dense feature maps degrading over long
training schedules. Its loss is

    L_Gram = ‖ X_S · X_Sᵀ − X_G · X_Gᵀ ‖²_F

on L2-normalised patch features, where `X_G` comes from a **Gram teacher** — an
earlier EMA snapshot (~100k–200k iterations) with better dense properties,
refreshed every 10k iterations. It needs an extra teacher forward pass at **2×
resolution**, and it only switches on after 1M iterations. Reported gain:
ADE20k 53.6 → 55.7 mIoU, NYU depth 0.285 → 0.281.

The proposal this experiment gates: **dense collapse is loss of high-frequency
structure in the patch-feature map, so a spectral regulariser could do Gram
anchoring's job with no second teacher and at O(N log N) instead of O(N²·d).**

That proposal is worth nothing until two things are established:

- **Q1** Can collapse be reproduced at a scale you can afford?
- **Q2** Does a spectral statistic track it?

If Q1 fails there is nothing to repair. If Q2 fails the regulariser targets the
wrong quantity. **Answer both before spending budget on the real comparison.**

## Files

| file | role |
|---|---|
| `dense_metrics.py` | label-free dense-quality metrics (no segmentation dataset needed) |
| `train_ssl.py` | compact DINO + iBOT trainer, probes dense quality every N epochs |
| `plot_probe.py` | plots the curves and prints the go/no-go verdict |
| `prepare_imagenet100.py` | pulls ImageNet-100 off the Hub into an ImageFolder tree |
| `runpod_bootstrap.sh` | one paste-and-go setup for a RunPod pod |

The four metrics, and the collapse signature to look for:

| metric | meaning | on collapse |
|---|---|---|
| `correspondence_acc` | cross-view patch matching under known crop geometry | **down** |
| `gram_drift` | DINOv3's Gram objective used as a *measurement* vs. an early checkpoint | up |
| `patch_sim_entropy` | flatness of per-patch similarity distributions | up |
| `hf_energy` | fraction of 2-D FFT energy above ½ Nyquist — **the spectral hypothesis** | down |

`correspondence_acc` ships with a `correspondence_pos_baseline` control: the score
obtainable by matching grid position alone, ignoring content. If the two ever
converge, the probe has stopped measuring feature quality and the run is
uninformative. Always read them together.

## Quick start

Validate the pipeline locally first — no data, about a minute:

```bash
python3 tools/spectral/train_ssl.py --smoke --out runs/smoke
```

Real run — on a GPU box, everything in one paste:

```bash
bash tools/spectral/runpod_bootstrap.sh
```

That clones the repo, installs dependencies, fetches the data, and launches
training in the background with `--resume`. Re-run it verbatim after a spot
preemption; both data prep and training continue from where they stopped.

The manual equivalent:

```bash
python3 tools/spectral/prepare_imagenet100.py --out /workspace/data/imagenet100
python3 tools/spectral/train_ssl.py \
    --data /workspace/data/imagenet100/train \
    --probe-data /workspace/data/imagenet100/val \
    --epochs 400 --batch-size 128 --workers 12 \
    --resume --out runs/pilot
python3 tools/spectral/plot_probe.py runs/pilot/probe.csv
```

`--resume` picks up from `<out>/ckpt.pt`, written at every probe. Use it — the
run is long enough that interruption is likely, especially on spot instances.

### About the dataset

`clane9/imagenet-100` on the Hub: 126,689 train / 5,000 val images over 100
classes, public, no auth. It ships as parquet, so `prepare_imagenet100.py`
converts it shard by shard and deletes each parquet right after, keeping peak
disk near the size of the final tree.

Two things worth knowing: the images are **pre-resized to 160 px on the short
side**, which is why the default `--img-size 128` is a good match and why the
JPEG tree is only **~2 GB**, not the ~16 GB the raw parquet size suggests. Class
folders are slugified (`African_hunting_dog`), because the original synset labels
contain commas and spaces that make `tar`/`rsync`/`find` awkward on a remote box.
Pass `--raw-names` to keep them verbatim. Labels are unused by the SSL objective
either way.

## Where to run this

**Not on the M3.** Measured here: the target config (batch 128, 2×128px global +
6×64px local crops, ViT-Tiny/8) did not finish **4 steps in 600 s** on MPS. A
reduced config (batch 16, 2+2 crops) manages ~0.65 s/step, which is fine for
debugging and nothing else. The full run needs ~400,000 steps. The Mac is for the
smoke test only.

Prepare the dataset **on the rented box**, not locally — a 30 GB volume is
plenty (the JPEG tree is ~2 GB).

| option | cost | verdict |
|---|---|---|
| **RunPod / Vast.ai RTX 4090** | ~$0.35–0.70/hr → **~$10–25 total** | **recommended.** Best price/performance for a small ViT. Use spot + `--resume`. |
| Lambda Labs A100 40GB | ~$1.29/hr → ~$25–40 | More reliable than Vast, no spot preemption. Overkill for ViT-Tiny. |
| Colab Pro+ (A100) | ~$50/month | Convenient if you already pay. Session limits make `--resume` mandatory. |
| Kaggle (2×T4, free) | free, 30 h/week | Zero cost but ~4× slower than a 4090, so ~5 weeks of quota. Viable only if budget is zero. |
| Your RTX 2060 (6 GB) | free | Needs batch ≤48 and gradient accumulation. Roughly 10× slower than a 4090 — weeks of wall-clock. Not worth it. |

Estimate for the 4090 path: ~990 steps/epoch on ImageNet-100 at batch 128,
roughly 2 min/epoch with bf16, so **~15–25 hours for 400 epochs**.

Two practical warnings:

1. **Data loading will probably be your bottleneck**, not the GPU. Each image
   produces 8 crops, so an epoch is ~1M PIL transform operations. Rent a box with
   ≥8 vCPU, set `--workers` to match, and check GPU utilisation early — if it sits
   under 60%, add workers before adding GPU.
2. **bf16 autocast turns on automatically for CUDA** (`--amp auto`). It is off on
   MPS/CPU because support is partial.
3. **Do not train on MPS.** Measured here: MPS produced non-finite activations
   inside the ViT forward in 3 of 6 short runs, against 0 of 6 on CPU — a backend
   numerical bug, not an instability in the objective. `--smoke` therefore pins
   itself to CPU unless you pass `--device` explicitly. If a real run ever dies
   with a non-finite loss, the error message names the first tensor that went
   bad; on CUDA that would point at a genuine problem worth investigating.

If you need to cut cost, in this order: `--epochs 300`, `--n-local 4`,
`--img-size 112`. Together those roughly halve the bill. But do not cut too far —
collapse is a *long-schedule* phenomenon, and shortening the schedule is exactly
the thing most likely to make Q1 answer "no" for the wrong reason.

## Reading the result

`plot_probe.py` prints one of three verdicts:

- **No collapse** — dense quality never degraded. Nothing to repair at this
  scale. Train longer or larger, or stop the project here. This is a real and
  acceptable outcome; it is the whole point of running the gate first.
- **Collapse + `hf_energy` falls with it** — hypothesis survives. Proceed to the
  four-arm comparison: none / Gram anchoring / spectral anchoring / both.
- **Collapse but `hf_energy` flat or rising** — collapse is real but is not a loss
  of high-frequency detail. The spectral regulariser targets the wrong quantity.
  Reconsider before spending more.

## Caveats

- This is a scale model of the phenomenon, not a replication. DINOv3 sees collapse
  at 7B parameters after 1M iterations; a ViT-Tiny on ImageNet-100 may simply not
  exhibit it. That is what Q1 tests.
- Metrics are label-free by design, to avoid needing a segmentation dataset. If Q1
  and Q2 both pass, add a real dense benchmark (ADE20k or VOC linear segmentation)
  before making any claim — the label-free proxies are for triage, not for a paper.
- `--probe-data` should point at a held-out split. It defaults to `--data` for
  convenience, which leaks slightly.
