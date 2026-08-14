"""
Recompute gram_drift against a reference chosen after the fact.

The live column in probe.csv is measured against the first probe, i.e. a model
that has barely trained. Distance from a near-random init grows during perfectly
healthy training, so that column conflates "learning" with "degrading".

DINOv3 anchors to a snapshot taken while dense features were still good — around
100k-200k iterations, well after init. The analogous reference here is the epoch
where dense quality peaks, which can only be identified once the run is far
enough along. This script re-derives the column against that epoch (or any epoch
you name), using the per-probe tokens train_ssl.py saved.

    python3 tools/spectral/regram.py runs/small
    python3 tools/spectral/regram.py runs/small --ref-epoch 120
"""

import os
import csv
import sys
import glob
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_metrics import gram_drift


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--ref-epoch", type=int, default=None,
                   help="reference epoch; default = argmax correspondence_acc")
    p.add_argument("--out", default=None, help="default <run_dir>/probe_regram.csv")
    args = p.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.run_dir, "probe.csv"))))
    if not rows:
        raise SystemExit("probe.csv is empty")

    tok_dir = os.path.join(args.run_dir, "probe_tokens")
    files = {int(os.path.basename(f)[6:10]): f
             for f in glob.glob(os.path.join(tok_dir, "epoch_*.pt"))}
    if not files:
        raise SystemExit(f"no saved tokens in {tok_dir}\n"
                         f"  (only runs started after the token-saving change have them)")

    have = [r for r in rows if int(r["epoch"]) in files]
    if len(have) < 2:
        raise SystemExit(f"need >=2 probes with saved tokens, found {len(have)}")

    if args.ref_epoch is None:
        best = max(have, key=lambda r: float(r["correspondence_acc"]))
        ref_epoch = int(best["epoch"])
        why = f"peak correspondence_acc = {float(best['correspondence_acc']):.4f}"
    else:
        ref_epoch = args.ref_epoch
        why = "specified on the command line"
        if ref_epoch not in files:
            raise SystemExit(f"no tokens saved for epoch {ref_epoch}; "
                             f"have {sorted(files)}")

    ref = torch.load(files[ref_epoch], map_location="cpu").float()
    print(f"reference epoch {ref_epoch}  ({why})")
    print(f"tokens available for {len(files)} probes\n")

    out = args.out or os.path.join(args.run_dir, "probe_regram.csv")
    fields = ["epoch", "correspondence_acc", "hf_energy",
              "gram_drift_vs_init", "gram_drift_vs_peak"]
    print(f"{'epoch':>6s} {'corr_acc':>9s} {'hf_energy':>10s} "
          f"{'vs_init':>9s} {'vs_peak':>9s}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in have:
            ep = int(r["epoch"])
            tok = torch.load(files[ep], map_location="cpu").float()
            d = gram_drift(tok, ref)
            row = {
                "epoch": ep,
                "correspondence_acc": float(r["correspondence_acc"]),
                "hf_energy": float(r["hf_energy"]),
                "gram_drift_vs_init": float(r["gram_drift"]),
                "gram_drift_vs_peak": round(d, 6),
            }
            w.writerow(row)
            mark = "  <- reference" if ep == ref_epoch else ""
            print(f"{ep:6d} {row['correspondence_acc']:9.4f} {row['hf_energy']:10.4f} "
                  f"{row['gram_drift_vs_init']:9.4f} {d:9.4f}{mark}")

    print(f"\nwrote {out}")
    print("gram_drift_vs_peak is the column to read: distance from the model's")
    print("own best dense features, so it rises only as they degrade.")


if __name__ == "__main__":
    main()
