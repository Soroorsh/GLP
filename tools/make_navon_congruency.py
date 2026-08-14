"""
Build a congruency manifest for the Navon test set.

The test set is stored twice under ./data:
    Navon-New/{circle,square}/        -> folder name is the GLOBAL shape
    Navon-New-Local/{circle,square}/  -> folder name is the LOCAL shape

Both trees hold the same images (byte-identical), so pairing them by content
hash recovers (global_shape, local_shape) per image, and congruency is just
whether the two agree.

Usage:
    python tools/make_navon_congruency.py --data-root ./data \
        --out ./data/navon_test_manifest.csv
"""

import os
import csv
import hashlib
import argparse
import collections


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def scan(root):
    """Map content hash -> list of (class_folder, filename)."""
    table = collections.defaultdict(list)
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                table[md5(os.path.join(cdir, fname))].append((cls, fname))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--global-dir', default='Navon-New')
    parser.add_argument('--local-dir', default='Navon-New-Local')
    parser.add_argument('--out', default='./data/navon_test_manifest.csv')
    opt = parser.parse_args()

    g = scan(os.path.join(opt.data_root, opt.global_dir))
    l = scan(os.path.join(opt.data_root, opt.local_dir))

    only_g, only_l = set(g) - set(l), set(l) - set(g)
    if only_g or only_l:
        raise RuntimeError(
            f"trees do not match: {len(only_g)} images only in {opt.global_dir}, "
            f"{len(only_l)} only in {opt.local_dir}")

    for name, table in [(opt.global_dir, g), (opt.local_dir, l)]:
        bad = [k for k, v in table.items() if len({c for c, _ in v}) > 1]
        if bad:
            raise RuntimeError(f"{name}: {len(bad)} images filed under >1 class")

    rows = []
    for k in sorted(g):
        gc, lc = g[k][0][0], l[k][0][0]
        rows.append(dict(
            md5=k,
            global_shape=gc,
            local_shape=lc,
            congruent=int(gc == lc),
            n_copies_global=len(g[k]),
            n_copies_local=len(l[k]),
            path_global=os.path.join(opt.global_dir, gc, g[k][0][1]),
            path_local=os.path.join(opt.local_dir, lc, l[k][0][1]),
        ))

    os.makedirs(os.path.dirname(opt.out) or '.', exist_ok=True)
    with open(opt.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_files = sum(r['n_copies_global'] for r in rows)
    n_cong = sum(r['congruent'] for r in rows)
    cells = collections.Counter((r['global_shape'], r['local_shape']) for r in rows)
    weighted = collections.Counter()
    for r in rows:
        weighted[(r['global_shape'], r['local_shape'])] += r['n_copies_global']

    print(f"wrote {opt.out}")
    print(f"  unique images : {len(rows)}  (files on disk: {n_files})")
    print(f"  congruent     : {n_cong} / {len(rows)} ({100 * n_cong / len(rows):.1f}%)")
    print("  cell counts (unique | files):")
    for cell in sorted(cells):
        print(f"    global={cell[0]:7s} local={cell[1]:7s} : "
              f"{cells[cell]:4d} | {weighted[cell]:4d}")


if __name__ == '__main__':
    main()
