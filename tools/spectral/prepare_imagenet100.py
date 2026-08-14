"""
Fetch ImageNet-100 and write it out as an ImageFolder tree.

`clane9/imagenet-100` on the Hub ships as parquet (126,689 train / 5,000 val,
~8.4 GB), but train_ssl.py wants plain folders. This converts shard by shard and
deletes each parquet immediately after, so peak disk stays near the size of the
final JPEG tree rather than parquet + JPEGs at once.

Resumable: a .done marker per shard, so re-running after an interruption picks up
where it stopped. Safe to re-run.

    python3 tools/spectral/prepare_imagenet100.py --out /workspace/data/imagenet100

Produces:
    <out>/train/<class>/*.jpg
    <out>/val/<class>/*.jpg
"""

import io
import os
import re
import json
import shutil
import argparse

REPO = "clane9/imagenet-100"


def slugify(name):
    """'African hunting dog, hyena dog, ...' -> 'African_hunting_dog'.

    ImageNet synset labels carry commas and spaces, which make the resulting
    folders painful to handle with tar/rsync/find on a remote box. Labels are
    unused by the SSL objective anyway, so a clean slug loses nothing.
    """
    head = name.split(",")[0].strip()
    slug = re.sub(r"[^0-9A-Za-z]+", "_", head).strip("_")
    return slug or "unknown"


def class_names(schema):
    """Recover label names from the HF features blob in the parquet schema."""
    meta = schema.metadata or {}
    blob = meta.get(b"huggingface")
    if not blob:
        return None
    try:
        feats = json.loads(blob.decode())["info"]["features"]
        return feats["label"]["names"]
    except (KeyError, ValueError, TypeError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="destination root")
    p.add_argument("--workdir", default=None,
                   help="scratch dir for parquet shards (default <out>/_shards)")
    p.add_argument("--splits", default="train,validation")
    p.add_argument("--raw-names", action="store_true",
                   help="keep original synset labels verbatim, spaces and all")
    args = p.parse_args()

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import list_repo_files, hf_hub_download
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency: {exc.name}\n"
            f"  pip install huggingface_hub pyarrow pillow")

    workdir = args.workdir or os.path.join(args.out, "_shards")
    os.makedirs(workdir, exist_ok=True)
    markers = os.path.join(workdir, "markers")
    os.makedirs(markers, exist_ok=True)

    wanted = tuple(s.strip() for s in args.splits.split(","))
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.endswith(".parquet")
                   and os.path.basename(f).startswith(wanted))
    if not files:
        raise SystemExit(f"no parquet shards matched splits={wanted}")
    print(f"{len(files)} shards to process")

    total = 0
    for i, remote in enumerate(files, 1):
        base = os.path.basename(remote)
        split = "val" if base.startswith("validation") else "train"
        marker = os.path.join(markers, base + ".done")
        if os.path.exists(marker):
            print(f"[{i}/{len(files)}] {base}: already done, skipping")
            continue

        print(f"[{i}/{len(files)}] {base}: downloading", flush=True)
        local = hf_hub_download(REPO, remote, repo_type="dataset",
                                local_dir=workdir)
        table = pq.read_table(local)
        names = class_names(table.schema)

        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        written = 0
        for j, (img, label) in enumerate(zip(images, labels)):
            if names:
                cls = names[label] if args.raw_names else slugify(names[label])
            else:
                cls = f"class_{label:03d}"
            cdir = os.path.join(args.out, split, cls)
            os.makedirs(cdir, exist_ok=True)
            path = os.path.join(cdir, f"{base[:-8]}_{j:06d}.jpg")
            if os.path.exists(path):
                continue
            data = img["bytes"] if isinstance(img, dict) else img
            with Image.open(io.BytesIO(data)) as im:
                im.convert("RGB").save(path, "JPEG", quality=92)
            written += 1

        total += written
        os.remove(local)
        open(marker, "w").close()
        print(f"    wrote {written} images (running total {total})", flush=True)

    # keep markers, drop any leftover blobs
    for entry in os.listdir(workdir):
        full = os.path.join(workdir, entry)
        if entry != "markers":
            shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) \
                else os.remove(full)

    for split in ("train", "val"):
        root = os.path.join(args.out, split)
        if os.path.isdir(root):
            n_cls = len(os.listdir(root))
            n_img = sum(len(fs) for _, _, fs in os.walk(root))
            print(f"{split}: {n_img} images across {n_cls} classes -> {root}")


if __name__ == "__main__":
    main()
