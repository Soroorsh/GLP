"""
Navon global-precedence benchmark for frozen vision backbones.

Method
------
1. Freeze the backbone. Extract pooled features for plain circles and squares.
2. Fit a 2-class logistic regression on those features. Nothing else is trained.
3. Run the same frozen features over the Navon hierarchical figures and apply
   the probe. The probe emits one label per image: "circle" or "square".
4. Score that single prediction against BOTH the global and the local label,
   split by congruency.

Why the congruency split is the whole point
-------------------------------------------
On a congruent figure (circle made of circles) either level yields the same
answer, so the trial says nothing about bias — only about whether the model can
see shape at all. It is a difficulty check.

Only incongruent figures force a choice. A circle made of squares scores as
"global" if the probe says circle and "local" if it says square. So:

    global_precedence_index = fraction of incongruent trials answered globally

That index is the headline number. A pooled accuracy over congruent and
incongruent together — which is what the GLP paper's Table 1 reports — mixes a
difficulty measure into a bias measure and dilutes the effect roughly twofold.

Sanity gate
-----------
`probe_heldout_acc` is accuracy on held-out plain shapes. If a backbone cannot
separate a plain circle from a plain square, its Navon number is noise, not
bias. Rows below --min-probe-acc are reported but flagged.

    python3 tools/navon/probe.py \
        --shapes data/navon_shapes \
        --navon-root data \
        --manifest data/navon_test_manifest.csv \
        --models resnet18 resnet50 convnext_tiny vit_base_patch16_224 \
        --out results/navon_benchmark.csv
"""

import os
import csv
import json
import argparse

import numpy as np
import torch
from PIL import Image


def build(name, device):
    import timm
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return model, tf, cfg


@torch.no_grad()
def features(model, tf, paths, device, batch=64):
    out = []
    for i in range(0, len(paths), batch):
        chunk = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        x = torch.stack(chunk).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def load_shapes(root):
    paths, labels = [], []
    for label, cls in enumerate(("circle", "square")):
        d = os.path.join(root, cls)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(d, f))
                labels.append(label)
    return paths, np.array(labels)


def load_navon(manifest, navon_root):
    rows = list(csv.DictReader(open(manifest)))
    paths = [os.path.join(navon_root, r["path_global"]) for r in rows]
    g = np.array([0 if r["global_shape"] == "circle" else 1 for r in rows])
    l = np.array([0 if r["local_shape"] == "circle" else 1 for r in rows])
    con = np.array([int(r["congruent"]) for r in rows]).astype(bool)
    return paths, g, l, con


def evaluate_model(name, shape_paths, shape_y, navon_paths, g, l, con, device, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split

    model, tf, cfg = build(name, device)
    fs = features(model, tf, shape_paths, device)
    fn = features(model, tf, navon_paths, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tr_x, te_x, tr_y, te_y = train_test_split(
        fs, shape_y, test_size=0.2, random_state=seed, stratify=shape_y)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(tr_x, tr_y)
    heldout = float(clf.score(te_x, te_y))

    pred = clf.predict(fn)

    # congruent trials: difficulty check (both levels agree)
    cong_acc = float((pred[con] == g[con]).mean())

    # incongruent trials: the actual bias measurement
    inc = ~con
    inc_global = float((pred[inc] == g[inc]).mean())
    inc_local = float((pred[inc] == l[inc]).mean())

    # pooled numbers, i.e. what a table that ignores congruency would report
    pooled_global = float((pred == g).mean())
    pooled_local = float((pred == l).mean())

    return {
        "model": name,
        "input_size": cfg["input_size"][-1],
        "probe_heldout_acc": round(heldout, 4),
        "congruent_acc": round(cong_acc, 4),
        "global_precedence_index": round(inc_global, 4),
        "incongruent_local": round(inc_local, 4),
        "pooled_global_acc": round(pooled_global, 4),
        "pooled_local_acc": round(pooled_local, 4),
        "n_congruent": int(con.sum()),
        "n_incongruent": int(inc.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", required=True)
    p.add_argument("--navon-root", required=True,
                   help="dir containing Navon-New/ (manifest paths are relative to it)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", default="results/navon_benchmark.csv")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-probe-acc", type=float, default=0.90)
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    shape_paths, shape_y = load_shapes(args.shapes)
    navon_paths, g, l, con = load_navon(args.manifest, args.navon_root)
    missing = [p_ for p_ in navon_paths[:5] if not os.path.exists(p_)]
    if missing:
        raise SystemExit(f"navon images not found, e.g. {missing[0]}\n"
                         f"  --navon-root should contain Navon-New/")
    print(f"device={device}  shapes={len(shape_paths)}  navon={len(navon_paths)} "
          f"({con.sum()} congruent / {(~con).sum()} incongruent)\n")

    fields = ["model", "input_size", "probe_heldout_acc", "congruent_acc",
              "global_precedence_index", "incongruent_local",
              "pooled_global_acc", "pooled_local_acc",
              "n_congruent", "n_incongruent"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"{'model':34s} {'probe':>6s} {'cong':>6s} {'GPI':>6s} {'inc_loc':>8s}")
    for name in args.models:
        try:
            row = evaluate_model(name, shape_paths, shape_y, navon_paths,
                                 g, l, con, device, args.seed)
        except Exception as exc:                      # noqa: BLE001
            print(f"{name:34s} FAILED: {str(exc)[:60]}")
            continue
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        flag = "" if row["probe_heldout_acc"] >= args.min_probe_acc else "  <- probe too weak"
        print(f"{name:34s} {row['probe_heldout_acc']:6.3f} {row['congruent_acc']:6.3f} "
              f"{row['global_precedence_index']:6.3f} {row['incongruent_local']:8.3f}{flag}")

    print(f"\nwrote {args.out}")
    print("GPI = global precedence index = fraction of INCONGRUENT trials "
          "answered with the global shape.")
    print("Compare GPI against published shape-bias scores: if the two are "
          "collinear the metric is redundant; if they scatter, it is a new axis.")


if __name__ == "__main__":
    main()
