"""
Join the avg-pooled and CLS-pooled benchmark runs into one table.

A single GPI per model hides half the story. DINOv2 reads 0.795 from the patch
mean and 0.246 from CLS — the same frozen weights, three-fold apart. So the
global/local balance is a property of the READOUT as much as of the model, and a
one-column table silently picks one and calls it the answer.

The gap between the two columns is itself informative: it measures how much a
model separates global from local information across its representations. That is
exactly what register tokens are supposed to do, and it shows up here as the two
columns moving in opposite directions when registers are added.

CNNs appear with an avg value only — they have no CLS token, so there is no
second reading to give.

    python3 tools/navon/merge_pools.py navon_avg.csv navon_cls.csv
"""

import csv
import sys
import argparse


def load(path):
    try:
        with open(path) as f:
            return {r["model"]: r for r in csv.DictReader(f)}
    except FileNotFoundError:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("avg_csv")
    p.add_argument("cls_csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    avg, cls = load(args.avg_csv), load(args.cls_csv)
    if not avg:
        raise SystemExit(f"no rows in {args.avg_csv}")

    rows = []
    for name, a in avg.items():
        c = cls.get(name)
        ga = float(a["global_precedence_index"])
        gc = float(c["global_precedence_index"]) if c else None
        rows.append({
            "model": name,
            "gpi_avg": round(ga, 4),
            "gpi_cls": round(gc, 4) if gc is not None else "",
            "gap_cls_minus_avg": round(gc - ga, 4) if gc is not None else "",
            "congruent_avg": float(a["congruent_acc"]),
            "congruent_cls": float(c["congruent_acc"]) if c else "",
        })

    rows.sort(key=lambda r: r["gpi_avg"])

    w = max(len(r["model"]) for r in rows)
    print(f"{'model':{w}s} {'GPI avg':>8s} {'GPI cls':>8s} {'gap':>7s} "
          f"{'cong avg':>9s}")
    for r in rows:
        gc = f"{r['gpi_cls']:8.3f}" if r["gpi_cls"] != "" else f"{'—':>8s}"
        gp = f"{r['gap_cls_minus_avg']:+7.3f}" if r["gap_cls_minus_avg"] != "" else f"{'—':>7s}"
        print(f"{r['model']:{w}s} {r['gpi_avg']:8.3f} {gc} {gp} "
              f"{r['congruent_avg']:9.3f}")

    out = args.out or "navon_both_pools.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {out}")
    print("gap = GPI(cls) - GPI(avg). Large positive means global information is")
    print("concentrated in CLS and kept out of the patch tokens.")


if __name__ == "__main__":
    main()
