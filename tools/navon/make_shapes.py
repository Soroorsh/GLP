"""
Generate the single-shape training set for the Navon probe.

The probe is trained on plain circles and squares, then evaluated on Navon
hierarchical figures. Whatever cue separates the two classes in this training
set is the cue the probe will look for in Navon — so every property that
correlates with LEVEL rather than SHAPE has to be decorrelated from the label,
or the resulting "global precedence" number measures our own generator instead
of the model.

Confounds this generator controls:

  size       Navon global shapes are large and local elements are small. A probe
             trained only on large shapes learns "read the large thing" and
             scores as global-biased on any model. Radii are therefore sampled
             log-uniformly across the full range, from local-element scale up to
             near the full canvas, INDEPENDENTLY of class.
  fill       If circles were filled and squares hollow, the probe would learn
             fill, not shape. Fill is a coin flip, independent of class.
  stroke     Same reasoning; line width is sampled independently of class.
  position   Centre is jittered so absolute location carries no class signal.
  rotation   Squares get a small orientation jitter only. Navon squares are
             axis-aligned at BOTH levels, so rotating training squares through
             the full quarter-turn would make most of them diamonds and add a
             domain gap for no benefit. Straight edges vs curvature is a real
             circle/square difference, not a shortcut, so it is fine to leave it
             as the discriminating cue.

Canvas matches the Navon test images: 128x128, white shape on black.

    python3 tools/navon/make_shapes.py --out data/navon_shapes --n 3000
"""

import os
import math
import argparse

from PIL import Image, ImageDraw

SUPERSAMPLE = 4          # draw large, downsample: cheap anti-aliasing
CANVAS = 128


def _draw_circle(draw, cx, cy, r, width, filled):
    box = [cx - r, cy - r, cx + r, cy + r]
    if filled:
        draw.ellipse(box, fill=255)
    else:
        draw.ellipse(box, outline=255, width=width)


def _draw_square(draw, cx, cy, r, width, filled, angle):
    pts = []
    for k in range(4):
        theta = angle + math.pi / 4 + k * math.pi / 2
        pts.append((cx + r * math.sqrt(2) * math.cos(theta),
                    cy + r * math.sqrt(2) * math.sin(theta)))
    if filled:
        draw.polygon(pts, fill=255)
    else:
        draw.polygon(pts, outline=255, width=width)


def render(shape, rng, canvas=CANVAS):
    s = SUPERSAMPLE
    img = Image.new("L", (canvas * s, canvas * s), 0)
    draw = ImageDraw.Draw(img)

    # log-uniform radius: small local-element scale .. near-full-canvas scale,
    # sampled the same way for both classes
    lo, hi = math.log(5.0), math.log(canvas * 0.42)
    r = math.exp(rng.uniform(lo, hi))

    filled = rng.random() < 0.5
    width = max(1, int(round(rng.uniform(1.0, max(1.5, r * 0.25)))))
    margin = r + width
    cx = rng.uniform(margin, canvas - margin) if canvas - 2 * margin > 1 else canvas / 2
    cy = rng.uniform(margin, canvas - margin) if canvas - 2 * margin > 1 else canvas / 2

    if shape == "circle":
        _draw_circle(draw, cx * s, cy * s, r * s, width * s, filled)
    else:
        jitter = math.radians(12)
        _draw_square(draw, cx * s, cy * s, r * s, width * s, filled,
                     rng.uniform(-jitter, jitter))

    return img.resize((canvas, canvas), Image.LANCZOS).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/navon_shapes")
    p.add_argument("--n", type=int, default=3000, help="total images, split evenly")
    p.add_argument("--canvas", type=int, default=CANVAS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import random
    rng = random.Random(args.seed)

    per_class = args.n // 2
    for shape in ("circle", "square"):
        d = os.path.join(args.out, shape)
        os.makedirs(d, exist_ok=True)
        for i in range(per_class):
            render(shape, rng, args.canvas).save(
                os.path.join(d, f"{shape}_{i:05d}.png"))
        print(f"{shape}: {per_class} images -> {d}")

    print(f"\ncanvas {args.canvas}x{args.canvas}, white on black, "
          f"radius log-uniform over [5, {args.canvas * 0.42:.0f}] px")
    print("fill / stroke / position / rotation all independent of class")


if __name__ == "__main__":
    main()
