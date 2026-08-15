# Global/Local Processing: what we ran, what we found

*14–15 August 2026. All code in `tools/`, all data in `results/`.*

---

## Bottom line

We asked whether the GLP paper's machinery could be repurposed to replace DINOv3's
**Gram anchoring** — an expensive fix for dense features degrading during long
training runs. We built a cheap experiment to test the idea before building it.

**The idea failed, and we know why.** The specific statistic we hoped would work —
high-frequency energy in the feature map — does not track dense-feature quality.
It falls steadily throughout training whether quality is improving or degrading.
Total cost of finding this out: about **$16 of GPU time**, instead of weeks spent
building a loss function on a false premise.

**A second, unrelated line of work succeeded.** The GLP paper's Navon experiment,
reanalysed properly and pointed at modern foundation models, produced a working
benchmark with large, clean effects. That is the publishable result.

---

## Background: the question

DINOv3's authors found that when you train a large vision transformer for a long
time, its *dense* features degrade. Dense features are the per-patch vectors that
downstream tasks like segmentation and depth estimation rely on. Over a long
schedule, patches gradually stop describing their own region and start looking
alike.

Their fix, Gram anchoring, adds a loss that pins the patch-to-patch similarity
structure to an earlier, healthier checkpoint. It works — ADE20k segmentation
improves from 53.6 to 55.7 mIoU — but it is expensive: a second teacher network,
an extra forward pass at double resolution, and an N×N matrix per image.

The question was whether GLP could replace it. Directly, no: GLP injects *global*
information into a model that is too local, whereas Gram anchoring rescues *local*
information in a model drifting global. Opposite directions.

But there was a reformulation worth testing. If dense collapse is fundamentally a
loss of fine detail, then a regulariser on the **spatial frequency spectrum** of
the feature map might do the same job with no second teacher and at O(N log N)
instead of O(N²·d). That became the hypothesis.

---

## Part 1: A finding hidden in the dataset

Before any training, we found something in the existing Navon data.

Navon figures are hierarchical: a large shape built out of small shapes. A circle
made of squares. The *global* level and the *local* level can agree (congruent) or
conflict (incongruent).

The GLP paper's test set is stored twice — once foldered by global shape, once by
local shape — over the same images. Pairing the two folders by content hash
recovers the congruency label for every image. It was already there, unused.

The set turns out to be **244 unique images**, duplicated to 576 files on disk to
balance the design, and **exactly 50% congruent**.

This matters because congruent trials cannot measure bias. If the big shape and the
little shapes are both circles, any correct answer is correct at both levels. Only
**incongruent** trials force the model to choose, and only those reveal which level
it reads. The paper's Table 1 pools both kinds together, which mixes a difficulty
measure into a bias measure and roughly halves the apparent effect.

Tool: `tools/make_navon_congruency.py` → `data/navon_test_manifest.csv`.

---

## Part 2: Testing the spectral idea

### The design

Rather than assume dense collapse would appear, we built a gate to answer two
questions before committing to anything:

- **Q1.** Can dense collapse be reproduced at a scale we can afford?
- **Q2.** If it can, does a spectral statistic track it?

If Q1 fails, there is nothing to repair. If Q2 fails, a spectral regulariser is
aiming at the wrong quantity. Either way, stop.

We trained a ViT-Small (21.5M parameters) with a DINO + iBOT objective on
ImageNet-100 (126,689 images), and every 10 epochs measured dense-feature quality
on a fixed held-out batch. Four measurements, all label-free — no segmentation
dataset required:

| metric | what it measures | direction under collapse |
|---|---|---|
| `correspondence_acc` | can a patch in one crop find its true match in an overlapping crop? | down |
| `patch_sim_entropy` | how flat is each patch's similarity profile? | up |
| `gram_drift` | distance from the model's own best patch-similarity structure | up |
| `hf_energy` | fraction of spectral energy above half Nyquist — **the hypothesis** | down |

`correspondence_acc` ships with a control: the score obtainable by matching grid
position alone, ignoring image content. It sat at 0.076 throughout, while the model
scored 0.40–0.63. The probe was measuring features, not position.

The run reached epoch 268 of a planned 300 before credit ran out — far enough.

### What happened

Dense quality rose to a peak at **epoch 140** (`correspondence_acc` = 0.6279), then
declined and settled around 0.60 — a **4% drop**, small but consistent across
eight consecutive measurements. `patch_sim_entropy` bottomed out at epoch 130, one
measurement before the peak, then rose steadily. `gram_drift` measured against the
peak checkpoint rose monotonically from zero, eight times in a row.

**Q1: yes.** Collapse reproduces at ViT-Small scale. It is mild and appears to be
self-limiting rather than a runaway, but it is real and three independent
measurements agree on it.

**Q2: no.** And the failure is decisive:

| phase | dense quality | `hf_energy` | correlation |
|---|---|---|---|
| epochs 30→140 | **up** +0.134 | **down** −0.044 | −0.70 |
| epochs 140→220 | **down** −0.025 | **down** −0.032 | +0.92 |

`hf_energy` falls throughout, regardless of what quality is doing. The correlation
between them flips sign halfway through the run. It behaves like a clock — a
function of training time — not a measure of quality.

The early co-movement that looked promising at epoch 20 turned out to be the
transition from random initialisation to learned features, not a stable
relationship.

### Verdict

**Do not build spectral anchoring.** This is exactly the decision the gate existed
to produce, and it cost one GPU run.

One consolation prize: `patch_sim_entropy` did catch the turning point, with its
minimum one measurement from the quality peak. It is a better candidate than the
spectrum. But it kept climbing after quality flattened, so it is not a tight proxy
either. The honest summary is that **neither cheap statistic is a reliable stand-in
for dense-feature quality**, and whether entropy could work *as a loss* — as opposed
to as a measurement — remains untested.

Tools: `tools/spectral/{train_ssl,dense_metrics,regram,plot_probe}.py`.
Data: `results/ssl_probe.csv`, `results/probe_tokens.tgz`.

---

## Part 3: The Navon benchmark

This is the part that worked.

### Method

Freeze a pretrained backbone. Train a two-class logistic regression on plain
circles and squares — nothing else is trained. Then run the frozen features over
the Navon figures and apply that probe. It outputs one label per image. Score it
against both the global and the local label, split by congruency.

On incongruent trials, answering "circle" for a circle-made-of-squares means the
model read the global level; "square" means it read the local level. The fraction
answered globally is the **global precedence index (GPI)** — the headline number.

The training shapes are generated with size, fill, stroke width, position and
rotation all decorrelated from the class label. This matters: if training circles
were always large, the probe would learn "read the big thing" and every model would
score as global-biased regardless of what it actually does.

Every model passed a sanity check — perfect accuracy separating plain circles from
plain squares — so no result below is noise from a probe that could not do the task.

### Results

All ten models at fixed 224px input, no edge cropping, and identical average
pooling over patch tokens:

| model | architecture | GPI |
|---|---|---|
| CLIP ViT-B/16 | ViT-B/16 | 0.123 |
| ResNet18 | CNN | 0.131 |
| DINOv3 ConvNeXt-B | ConvNeXt | 0.205 |
| ConvNeXt-T | ConvNeXt | 0.320 |
| ConvNeXt-B | ConvNeXt | 0.336 |
| ViT-B supervised | ViT-B/16 | 0.385 |
| ResNet50 | CNN | 0.418 |
| DINOv2 + registers | ViT-B/14 | 0.492 |
| DINOv3 ViT-B/16 | ViT-B/16 | 0.762 |
| DINOv2 ViT-B/14 | ViT-B/14 | 0.795 |

Standard error is about 4.5 points on 122 incongruent trials.

### Two controlled comparisons

These are the load-bearing results, because neither can be explained away as
"the newer model is simply better."

**Architecture gates the effect.** DINOv3 ViT-B scores 0.762; DINOv3 ConvNeXt-B
scores 0.205. Same data, same ViT-7B teacher, same distillation, same pooling. Only
the architecture differs — a 3.7× gap, roughly 12 standard errors. And the ConvNeXt
student lands *below* a plain supervised ConvNeXt (0.336), meaning DINOv3's training
recipe moved the transformer enormously and did nothing for the convnet.

**Within a permitting architecture, training decides everything.** CLIP (0.123),
supervised (0.385) and DINOv3 (0.762) are all ViT-B/16 with 196 tokens. Identical
architecture, patch size, resolution and pooling. A 6.2× range, about 14 standard
errors, from training alone. This also rules out patch size as an explanation, since
CLIP and DINOv3 are identical there.

GPI is not a quality ranking. A 2016 ResNet50 (0.418) beats DINOv2, CLIP,
ConvNeXt-B, and DINOv3's own ConvNeXt student.

### The surprise

Reading the same frozen weights two different ways gives contradictory answers, but
only for one family of models:

| model | GPI from patch mean | GPI from CLS token | gap |
|---|---|---|---|
| CLIP ViT-B/16 | 0.123 | 0.221 | +0.098 |
| ViT-B supervised | 0.385 | 0.369 | −0.016 |
| DINOv2 + registers | 0.492 | 0.500 | +0.008 |
| DINOv3 ViT-B/16 | 0.762 | 0.221 | **−0.541** |
| DINOv2 ViT-B/14 | 0.795 | 0.246 | **−0.549** |

In DINOv2 and DINOv3, the patch tokens read *global* and the CLS token reads
*local*. This is backwards from intuition — the CLS token is supposed to be the
model's whole-image summary. Supervised and CLIP models show no such split.

We do not have a mechanism for this. It appears to be a signature of DINO-style
self-supervised training, and it is not simply about register tokens: DINOv3 has
registers and shows the large gap, while DINOv2-with-registers does not.

Tools: `tools/navon/{make_shapes,probe,merge_pools}.py`.
Data: `results/navon_avg.csv`, `results/navon_cls.csv`, `results/navon_both.csv`.

---

## What this supports, and what it does not

**Supported.** Global/local balance is not a fixed property of a model and not a
by-product of model quality. Architecture sets a ceiling; training determines where
you land beneath it. The measurement instrument works, discriminates over a
six-fold range, and produces effects far above noise.

**Not yet supported.** The central claim a paper would need — that this index is a
*new axis*, not a restatement of shape-bias — is still a hypothesis. Two hints point
that way: CLIP and DINOv2 sit near the bottom here despite ranking high in the
shape-bias literature. But nobody has plotted GPI against published shape-bias
scores, and until that plot exists the claim is a guess.

**Known limitations.** One random seed, one generated shape set, one stimulus pair
(circle vs square, where the original Navon paradigm used letters), 244 images, no
human baseline. Setup choices moved numbers by up to 0.05, so differences smaller
than that should not be interpreted.

---

## What to do next, in order

1. **Plot GPI against published shape-bias scores.** One afternoon, no GPU. If the
   points fall on a line, the metric is redundant and this project ends. If they
   scatter, there is a paper. Do this before anything else — it can kill the whole
   direction cheaply, which is exactly why it goes first.
2. **Error bars.** Multiple seeds and shape sets, plus the DINOv2 and DINOv3 size
   ladders. One overnight GPU run.
3. **A second stimulus pair**, ideally letters, to show the result is not an artefact
   of circles and squares.
4. **Submit to a workshop.** Venues on human and machine visual representation are
   the natural audience, and reviewer feedback there will tell you what a full
   conference version needs.

The entropy-as-a-loss experiment — two arms, control versus an entropy penalty,
about $20 — is worth running eventually. It should not come before the shape-bias
plot.

---

## Corrections made along the way

Recorded because each one would have silently corrupted the results, and anyone
repeating this work will hit them:

- **Input resolution.** Model default configs range from 224 to 518. Navon figures
  are natively 128×128, so a model upsampling further resolves the small local
  elements better. Resolution correlated with model identity until we pinned it.
- **Edge cropping.** The default preprocessing resizes then centre-crops, cutting
  off the image border — which damages the global shape specifically. Fixed by
  disabling the crop.
- **Pooling.** timm's default pooling is *not* consistent across models: DINOv3
  averages patch tokens while CLIP, DINOv2 and supervised ViTs return CLS. Our first
  benchmark unknowingly compared a mean-pooled model against CLS-pooled ones, and it
  landed on the model with the most extreme result. Fixing this changed DINOv2 from
  0.246 to 0.795 and reversed the apparent effect of register tokens.
- **The Gram reference point.** Measuring drift from the *first* checkpoint measures
  distance from random initialisation, which grows during healthy training. The
  reference has to be the epoch where quality peaks, which is only identifiable in
  hindsight — hence saving per-probe tokens and recomputing afterwards.
- **A silent CPU fallback.** The training script did not stop when PyTorch could not
  initialise CUDA. It would have trained on CPU at a few hundred times the intended
  runtime while the GPU sat idle. It now fails immediately.

---

## Cost

About **$20** total: one RTX 4090 for roughly 24 hours, plus a broken host that had
to be discarded early. The Navon benchmark ran on the same machine's idle CPU cores
and cost nothing extra.
