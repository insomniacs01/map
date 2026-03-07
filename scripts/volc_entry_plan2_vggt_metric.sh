#!/bin/bash
set -euo pipefail

# Volcano/MLP-friendly entrypoint for Plan2 (VGGT metric pose+depth, no explicit scale head).
#
# Design goals:
# - Works when the repo is mounted under different base paths across machines.
# - Does NOT rely on SSH between nodes.
# - Can use 2 nodes in a "split" mode (node0 runs coop, node1 runs single) to maximize throughput
#   OR a true multi-node torchrun ("ddp") mode.
#
# Usage (typical):
#   # 2-node task (2x8 A800). Default: split mode
#   bash map-anything/scripts/volc_entry_plan2_vggt_metric.sh
#
#   # Force a single job that uses all nodes (ddp across 2 nodes)
#   PLAN2_TOPO=ddp PLAN2_JOB=coop bash map-anything/scripts/volc_entry_plan2_vggt_metric.sh
#
# Environment variables (optional):
#   PLAN2_TOPO=split|ddp            (default: split)
#   PLAN2_JOB=auto|single|coop      (default: auto; in split: node0->coop, node1->single)
#   PLAN2_EPOCHS=<int>              (default: 1)
#   PLAN2_WARM_CKPT=<path>          (default: a known local warm checkpoint if present)
#   PLAN2_LOSS=<loss_cfg>           (default: opv2v_vggt_pose_metric_loss)
#   PLAN2_EXTRA_ARGS="<hydra overrides...>"
#   PLAN2_MASTER_PORT=<port>        (ddp only; default derived from task id)

_now_utc() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

echo "[plan2] start ts=$(_now_utc)"
echo "[plan2] host=$(hostname 2>/dev/null || echo unknown) user=$(whoami 2>/dev/null || echo unknown) pwd=$(pwd)"

# ---- Locate repo root (robust across mount points) ----
REPO_ROOT=""
for base in \
  /J6P-perception/yijinxiong_workspace \
  /mlp-vepfs/J6P-perception/yijinxiong_workspace \
  /workspace/J6P-perception/yijinxiong_workspace \
  /data/J6P-perception/yijinxiong_workspace \
  /J6P-perception \
  /mlp-vepfs \
  /workspace \
  /data; do
  if [ -d "${base}/vggt_series_4_coop/map-anything" ]; then
    REPO_ROOT="${base}/vggt_series_4_coop/map-anything"
    break
  fi
done
if [ -z "${REPO_ROOT}" ]; then
  echo "[plan2][ERR] repo not found under common mount points." >&2
  exit 3
fi
echo "[plan2] REPO_ROOT=${REPO_ROOT}"

# ---- Python / torchrun ----
PYTHON_CANDIDATES=(
  "/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"
  "${REPO_ROOT}/../venvs/mapanything-cu121/bin/python"
  "${REPO_ROOT}/../../venvs/mapanything-cu121/bin/python"
)
PYTHON=""
for p in "${PYTHON_CANDIDATES[@]}"; do
  if [ -x "${p}" ]; then
    PYTHON="${p}"
    break
  fi
done
if [ -z "${PYTHON}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"
  fi
fi
echo "[plan2] PYTHON=${PYTHON}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

# Basic dependency sanity check for "no luca / clean env" starts.
if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1; then
import torch  # noqa: F401
import hydra  # noqa: F401
import omegaconf  # noqa: F401
PY
  echo "[plan2][ERR] Python environment is missing required deps (torch/hydra/omegaconf)." >&2
  echo "[plan2][ERR] Suggested bootstrap (run once):" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate" >&2
  echo "  pip install -U pip" >&2
  echo "  pip install -r ${REPO_ROOT}/condaenv.xl9zch7u.requirements.txt" >&2
  echo "  pip install -e ${REPO_ROOT}" >&2
  exit 11
fi

# ---- Cache + tmp (avoid AF_UNIX path too long) ----
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/J6P-perception/yijinxiong_workspace}"
if [ ! -d "${WORKSPACE_ROOT}" ]; then
  WORKSPACE_ROOT="$(dirname "${REPO_ROOT}")"
fi
export TORCH_HOME="${TORCH_HOME:-${WORKSPACE_ROOT}/.cache/torch}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" || true

WORK_TMP_ROOT="${WORK_TMP_ROOT:-/tmp/ma_tmp}"
mkdir -p "${WORK_TMP_ROOT}" || true
export TMPDIR="${TMPDIR:-${WORK_TMP_ROOT}}"
export TORCHELASTIC_TMPDIR="${TORCHELASTIC_TMPDIR:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Prefer the primary NIC in MLP environments; do not override if user already set it.
if [ -z "${NCCL_SOCKET_IFNAME:-}" ] && [ -n "${MLP_PRIMARY_NIC_NAME:-}" ]; then
  export NCCL_SOCKET_IFNAME="${MLP_PRIMARY_NIC_NAME}"
fi

# ---- Topology selection ----
PLAN2_TOPO="${PLAN2_TOPO:-split}"   # split | ddp
PLAN2_JOB="${PLAN2_JOB:-auto}"      # auto | single | coop
PLAN2_EPOCHS="${PLAN2_EPOCHS:-1}"
PLAN2_LOSS="${PLAN2_LOSS:-opv2v_vggt_pose_metric_loss}"
PLAN2_EXTRA_ARGS="${PLAN2_EXTRA_ARGS:-}"

NNODES="${MLP_WORKER_NUM:-${NNODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
MASTER_ADDR="${MLP_WORKER_0_PRIMARY_HOST:-${MASTER_ADDR:-${MLP_PRIMARY_HOST:-127.0.0.1}}}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_GPU:-8}}"
if command -v nvidia-smi >/dev/null 2>&1; then
  detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')"
  if [ "${detected_gpus}" -gt 0 ]; then
    GPUS_PER_NODE="${GPUS_PER_NODE:-${detected_gpus}}"
  fi
fi

echo "[plan2] topo=${PLAN2_TOPO} nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR} gpus_per_node=${GPUS_PER_NODE} epochs=${PLAN2_EPOCHS} loss=${PLAN2_LOSS}"

resolve_port_from_task_id() {
  local base=29500
  local span=1000
  local seed="${1:-}"
  if [ -z "${seed}" ]; then
    echo "${base}"
    return 0
  fi
  # cksum is portable and gives a stable int; mod to keep it in a safe range.
  local sum
  sum="$(echo -n "${seed}" | cksum | awk '{print $1}')"
  echo $((base + (sum % span)))
}

if [ "${PLAN2_TOPO}" = "ddp" ]; then
  if [ "${NNODES}" -lt 2 ]; then
    echo "[plan2][ERR] PLAN2_TOPO=ddp requires >=2 nodes (MLP_WORKER_NUM/NNODES)." >&2
    exit 4
  fi
fi

# ---- Decide which job to run on this node ----
job="${PLAN2_JOB}"
if [ "${job}" = "auto" ]; then
  if [ "${PLAN2_TOPO}" = "split" ] && [ "${NNODES}" -ge 2 ]; then
    if [ "${NODE_RANK}" -eq 0 ]; then
      job="coop"
    else
      job="single"
    fi
  else
    job="single"
  fi
fi
echo "[plan2] selected job=${job}"

dataset_cfg=""
train_params_cfg=""
run_prefix=""
case "${job}" in
  single)
    dataset_cfg="${PLAN2_DATASET_SINGLE:-opv2v_ft}"
    train_params_cfg="${PLAN2_TRAIN_PARAMS_SINGLE:-opv2v_vggt_single_4v_long}"
    run_prefix="plan2_single_metric"
    ;;
  coop)
    dataset_cfg="${PLAN2_DATASET_COOP:-opv2v_coop_ft_2a8v}"
    train_params_cfg="${PLAN2_TRAIN_PARAMS_COOP:-opv2v_vggt_coop_8v_long}"
    run_prefix="plan2_coop_metric"
    ;;
  *)
    echo "[plan2][ERR] unknown PLAN2_JOB=${PLAN2_JOB} (resolved job=${job})" >&2
    exit 5
    ;;
esac

# ---- Warm start checkpoint (avoid HF download whenever possible) ----
DEFAULT_WARM="${REPO_ROOT}/experiments/vggt/training/opv2v_single_vggt_pose_metric_vggt1k/20260114_220502/checkpoint-best.pth"
WARM_CKPT="${PLAN2_WARM_CKPT:-${DEFAULT_WARM}}"
model_preload_args=()
if [ -f "${WARM_CKPT}" ]; then
  echo "[plan2] warm_ckpt=${WARM_CKPT}"
  model_preload_args+=("model.model_config.load_pretrained_weights=false")
  model_preload_args+=("model.model_config.pretrained_checkpoint_path=${WARM_CKPT}")
else
  echo "[plan2][WARN] warm_ckpt not found (${WARM_CKPT}); will try HF download (facebook/VGGT-1B) if allowed."
  model_preload_args+=("model.model_config.load_pretrained_weights=true")
fi

RUN_ID="${PLAN2_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_TAG="${PLAN2_RUN_TAG:-${RUN_ID}_${run_prefix}_n${NNODES}r${NODE_RANK}}"
OUT_DIR="${REPO_ROOT}/experiments/vggt/training/${RUN_TAG}"

mkdir -p "${OUT_DIR}"
echo "[plan2] out_dir=${OUT_DIR}"

# Machine paths: override to be robust across mounts.
machine_args=(
  "machine=local_a800"
  "machine.root_data_dir=${REPO_ROOT}/data"
  "machine.mapanything_dataset_metadata_dir=${REPO_ROOT}/metadata"
  "machine.root_pretrained_checkpoints_dir=${REPO_ROOT}/checkpoints"
  "machine.root_experiments_dir=${REPO_ROOT}/experiments"
  "machine.root_uniception_pretrained_checkpoints_dir=${REPO_ROOT}/checkpoints/uniception"
  "machine.external_benchmark_data_root_data_dir=${REPO_ROOT}/external_benchmark_data"
  "machine.opv2v_images_root=${REPO_ROOT}/data/opv2v_images"
  "machine.opv2v_depth_root=${REPO_ROOT}/data/opv2v_depth"
)

hydra_args=(
  "${machine_args[@]}"
  "model=vggt"
  "model.model_config.enable_metric_scale_head=false"
  "${model_preload_args[@]}"
  "dataset=${dataset_cfg}"
  "dataset.num_workers=${PLAN2_NUM_WORKERS:-12}"
  "dataset.principal_point_centered=true"
  "loss=${PLAN2_LOSS}"
  "train_params=${train_params_cfg}"
  "train_params.epochs=${PLAN2_EPOCHS}"
  "train_params.eval_freq=1"
  "train_params.save_freq=1"
  "train_params.full_eval_freq=${PLAN2_FULL_EVAL_FREQ:-0}"
  "hydra.run.dir=${OUT_DIR}"
)

if [ -n "${PLAN2_EXTRA_ARGS}" ]; then
  # Split PLAN2_EXTRA_ARGS on whitespace into an array (shellcheck: intentional).
  read -r -a _extra <<< "${PLAN2_EXTRA_ARGS}"
  hydra_args+=("${_extra[@]}")
fi

echo "[plan2] hydra_args:"
printf '  %q\n' "${hydra_args[@]}"

if [ "${PLAN2_TOPO}" = "ddp" ]; then
  MASTER_PORT="${PLAN2_MASTER_PORT:-$(resolve_port_from_task_id "${MLP_TASK_ID:-}")}"
  echo "[plan2] ddp master_port=${MASTER_PORT}"
  set -x
  "${PYTHON}" -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/train.py \
    "${hydra_args[@]}" \
    2>&1 | tee "${OUT_DIR}/launch.log"
else
  # split mode: each node runs an independent 8-GPU job; no inter-node comm needed.
  # Use a local random master port to avoid collisions on the same node.
  MASTER_PORT="${PLAN2_MASTER_PORT:-$((29500 + RANDOM % 1000))}"
  echo "[plan2] split master_port(local)=${MASTER_PORT}"
  set -x
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    scripts/train.py \
    "${hydra_args[@]}" \
    2>&1 | tee "${OUT_DIR}/launch.log"
fi

echo "[plan2] done ts=$(_now_utc)"
