"""
Plot the zero-experiment probe curves and print the go/no-go verdict.

    python3 tools/spectral/plot_probe.py runs/pilot/probe.csv
"""

import csv
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = [
    ("correspondence_acc", "cross-view patch matching\n(dense quality: DOWN = collapse)"),
    ("gram_drift", "Gram drift vs. early checkpoint\n(UP = dense structure drifting)"),
    ("patch_sim_entropy", "patch similarity entropy\n(UP = flatter, over-smoothed)"),
    ("hf_energy", "high-frequency energy\n(DOWN = spectral hypothesis supported)"),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/pilot/probe.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise SystemExit(f"{path}: need at least 2 probe points, found {len(rows)}")

    epochs = [int(r["epoch"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (key, title) in zip(axes, PANELS):
        ax.plot(epochs, col(key), marker="o", ms=3)
        if key == "correspondence_acc" and "correspondence_pos_baseline" in rows[0]:
            ax.plot(epochs, col("correspondence_pos_baseline"), "--",
                    color="gray", label="position-only baseline")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(path) or ".", "probe_curves.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # ---- verdict --------------------------------------------------------- #
    acc, hf = col("correspondence_acc"), col("hf_energy")
    peak = max(range(len(acc)), key=lambda i: acc[i])
    drop = acc[peak] - acc[-1]

    print(f"\ncorrespondence_acc: peak {acc[peak]:.4f} @ epoch {epochs[peak]}"
          f" -> final {acc[-1]:.4f}   (drop {drop:+.4f})")
    print(f"hf_energy:          {hf[peak]:.4f} @ peak -> {hf[-1]:.4f}"
          f"   (change {hf[-1] - hf[peak]:+.4f})")

    if "correspondence_pos_baseline" in rows[0]:
        base = col("correspondence_pos_baseline")[-1]
        if acc[-1] <= base:
            print("\nWARNING: final accuracy is at or below the position-only "
                  "baseline. The probe is not measuring feature quality.")

    print()
    if drop < 0.01:
        print("VERDICT: no collapse reproduced at this scale.")
        print("  Dense quality never degraded, so there is nothing for spectral")
        print("  anchoring to repair here. Either train longer / larger, or stop.")
    elif hf[-1] < hf[peak]:
        print("VERDICT: collapse reproduced AND high-frequency energy fell with it.")
        print("  The spectral-anchoring hypothesis survives its first test.")
        print("  Next: implement the loss and run the 4-arm comparison.")
    else:
        print("VERDICT: collapse reproduced, but high-frequency energy did NOT fall.")
        print("  Collapse is real but is not a loss of high-frequency detail, so a")
        print("  spectral regulariser targets the wrong quantity. Reconsider before")
        print("  spending GPU budget on the 4-arm comparison.")


if __name__ == "__main__":
    main()
