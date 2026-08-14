#!/usr/bin/env bash
# One-shot setup for the zero experiment on a RunPod pod.
#
#   1. Push `tools/` to GitHub first — this script clones the repo.
#   2. Start a pod: RTX 4090, Community Cloud, "RunPod PyTorch" template,
#      >= 8 vCPU, >= 30 GB volume mounted at /workspace.
#   3. Paste this whole file into the pod terminal, or:
#        curl -sL <raw-url>/runpod_bootstrap.sh | bash
#
# Safe to re-run. Data preparation and training both resume, so after a spot
# preemption just run it again.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Soroorsh/GLP.git}"
ROOT="${ROOT:-/workspace}"
DATA="$ROOT/data/imagenet100"
RUN="$ROOT/runs/pilot"
EPOCHS="${EPOCHS:-400}"
BATCH="${BATCH:-128}"
# Local-crop count is the cheapest knob: 6 -> 4 cuts roughly 30% of per-step
# compute while leaving the schedule length untouched. Prefer this over cutting
# EPOCHS, since collapse is a long-schedule phenomenon and a shorter run is the
# most likely way to get a false "no collapse" answer.
N_LOCAL="${N_LOCAL:-6}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "no nvidia-smi — this is not a GPU pod"; exit 1; }

# nvidia-smi succeeding is NOT enough: the driver can be visible while torch
# still fails to initialise CUDA (notably a cu12.x build on a CUDA 13.x driver).
# Without this gate train_ssl.py silently falls back to CPU and burns rental
# hours at a few hundred times the intended speed.
python3 - <<'PY' || exit 1
import sys, torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", ok)
if not ok:
    print()
    print("ERROR: torch cannot initialise CUDA, so training would run on CPU.")
    print("  Driver is visible to nvidia-smi, so this is a container/driver")
    print("  mismatch, not a missing GPU. Usual fixes, in order:")
    print("    1. restart the pod (clears a transient init failure)")
    print("    2. redeploy on a host whose CUDA major version matches the")
    print("       image — this image is cu128, so pick a CUDA 12.8 host")
    sys.exit(1)
PY

echo
echo "=== repo ==="
mkdir -p "$ROOT"
cd "$ROOT"
if [ -d GLP/.git ]; then
    cd GLP && git pull --ff-only || echo "(pull skipped)"
else
    git clone "$REPO_URL" GLP && cd GLP
fi
if [ ! -f tools/spectral/train_ssl.py ]; then
    echo "ERROR: tools/spectral/ is missing from the clone."
    echo "Commit and push it from your laptop first, then re-run."
    exit 1
fi

echo
echo "=== dependencies ==="
pip install --quiet --upgrade timm huggingface_hub pyarrow pillow matplotlib

echo
echo "=== data (~2 GB, resumable) ==="
python3 tools/spectral/prepare_imagenet100.py --out "$DATA"

# 8 crops per image makes loading the likely bottleneck, so use plenty of
# workers — but cap by default, since more workers than the machine has cores
# just adds contention. Override with WORKERS=... on a big host.
WORKERS="${WORKERS:-$(python3 -c "import os; print(min(os.cpu_count() or 4, 16))")}"
mkdir -p "$RUN"

echo
echo "=== launch ==="
echo "epochs=$EPOCHS batch=$BATCH n_local=$N_LOCAL workers=$WORKERS"
nohup python3 tools/spectral/train_ssl.py \
    --data "$DATA/train" \
    --probe-data "$DATA/val" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --n-local "$N_LOCAL" \
    --workers "$WORKERS" \
    --resume \
    --out "$RUN" \
    > "$RUN/train.log" 2>&1 &

sleep 20
tail -n 15 "$RUN/train.log" || true

cat <<EOF

------------------------------------------------------------------
started in the background.

  follow:      tail -f $RUN/train.log
  probe curve: cat $RUN/probe.csv
  plot:        python3 tools/spectral/plot_probe.py $RUN/probe.csv

CHECK GPU UTILISATION IN THE FIRST FEW MINUTES:

  nvidia-smi --query-gpu=utilization.gpu --format=csv -l 5

If it sits below ~60%, the dataloader is the bottleneck, not the GPU
(each image produces 8 crops). Kill the run, restart with more workers
or a pod with more vCPU. Paying for an idle 4090 is the main way this
experiment wastes money.

If the pod is preempted, just run this script again — data prep and
training both resume from where they stopped.
------------------------------------------------------------------
EOF
