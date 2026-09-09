DATASET=${DATASET:-psma}        # psma or fdg
MODALITY=${MODALITY:-petct}     # pet, ct, or petct ([CT, PET, PET])
SAVE_HEATMAPS=${SAVE_HEATMAPS:-0}  # 0 none, N first N, -1 all
PYTHON=${PYTHON:-python}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON"
  echo "Activate your conda environment first, or run with PYTHON=/path/to/python."
  exit 1
fi

SAVE_DIR_ARG=""
if [ -n "$SAVE_DIR" ]; then
  SAVE_DIR_ARG="--save_dir $SAVE_DIR"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"

"$PYTHON" "$REPO_DIR/main.py" \
    --gpu 0 \
    --seed 0 \
    --test test \
    --save_heatmaps "$SAVE_HEATMAPS" \
    $SAVE_DIR_ARG \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 30 \
    --eval_epochs 30 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 1840 \
  dataset \
    --petct_dataset "$DATASET" \
    --distribution 2 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 256 \
    --imagesize 256 \
    -d "$MODALITY" \
    petct
