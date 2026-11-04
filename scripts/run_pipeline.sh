#!/usr/bin/env bash
# End-to-end P2N pipeline: generate -> audit -> train -> evaluate.
#
# Run directly on a GPU node, or submit with SLURM, e.g.
#   sbatch --partition=<gpu-partition> --account=<account> --gpus=4 --cpus-per-task=32 \
#          --mem=128G --time=24:00:00 scripts/run_pipeline.sh
#
# All knobs are environment variables (defaults in brackets):
#   DATA_DIR [data]  OUT_DIR [p2n-model]  SAMPLES [50000]  SEED [1]
#   NGPU [auto]  BATCH [4]  ACCUM [4]  EPOCHS [10]  IMAGE_SIZE ["1024 1024"]  MAX_LEN [1024]
#   STAGE ["all" | "data" | "train" | "eval"]
#SBATCH --job-name=p2n
#SBATCH --output=p2n-%j.log
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-p2n-model}
SAMPLES=${SAMPLES:-50000}
SEED=${SEED:-1}
NGPU=${NGPU:-$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
EPOCHS=${EPOCHS:-10}
IMAGE_SIZE=${IMAGE_SIZE:-"1024 1024"}
MAX_LEN=${MAX_LEN:-1024}
STAGE=${STAGE:-all}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}

if [[ $STAGE == all || $STAGE == data ]]; then
  if [[ ! -f $DATA_DIR/metadata.jsonl ]]; then
    python generate_data.py --samples "$SAMPLES" --output_dir "$DATA_DIR" --aux_tasks \
      --degrade_fraction 0.3 --seed "$SEED" --cpus "$CPUS"
  fi
  python tools/audit_generators.py --runs 3 --cpus "$CPUS" --out "$DATA_DIR/generator_audit.json" | tail -5
  python tools/target_stats.py --dataset_path "$DATA_DIR" --max_length "$MAX_LEN" | head -40
fi

if [[ $STAGE == all || $STAGE == train ]]; then
  # shellcheck disable=SC2086
  torchrun --nproc_per_node="$NGPU" train.py \
    --dataset_path "$DATA_DIR" --output_dir "$OUT_DIR" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
    --image_size $IMAGE_SIZE --max_length "$MAX_LEN" \
    --num_workers $(( CPUS / NGPU > 8 ? 8 : CPUS / NGPU )) \
    --eval_steps 1000 --save_steps 2000 --gen_eval_samples 64
fi

if [[ $STAGE == all || $STAGE == eval ]]; then
  python evaluate.py --model_dir "$OUT_DIR" --dataset_path "$DATA_DIR" --num_samples 300 --batch_size 16
fi
