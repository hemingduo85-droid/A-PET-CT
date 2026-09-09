#!/bin/bash
set -e


# cd /data/cyf/codes/A-PET-CT/VisualAD-main

# # 双模态 PET+CT（推荐）
# bash scripts/run_petct.sh cuda:2 psma petct 30 8 256

# # 只跑 CT
#nohup bash scripts/run_petct.sh cuda:2 ct 30 8 256 > ct_psma.log 2>&1 &

# # 只跑 PET
# nohup bash scripts/run_petct.sh cuda:0 pet 30 8 256 > pet.log 2>&1 &

# ========== 参数（均可命令行覆盖） ==========
GPU="${1:-cuda:0}"
DATASET="${2:-psma}"     # psma / fdg
MODALITY="${3:-petct}"   # pet / ct / petct
EPOCHS="${4:-30}"
BATCH_SIZE="${5:-8}"
IMAGE_SIZE="${6:-256}"

SAVE_ROOT="./experiments"
RESULT_DIR="./test_results_petct/${DATASET}_${MODALITY}"

echo "================================================================"
echo "VisualAD PET-CT  (train 30 epochs + final eval)"
echo "GPU:        ${GPU}"
echo "Dataset:    ${DATASET}    (psma / fdg)"
echo "Modality:   ${MODALITY}   (pet / ct / petct)"
echo "Epochs:     ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMAGE_SIZE}"
echo "Save root:  ${SAVE_ROOT}"
echo "================================================================"

python train_petct.py \
    --dataset         "${DATASET}"    \
    --save_path       "${SAVE_ROOT}"  \
    --modality        "${MODALITY}"   \
    --epoch           "${EPOCHS}"     \
    --batch_size      "${BATCH_SIZE}" \
    --image_size      "${IMAGE_SIZE}" \
    --device          "${GPU}"

python test_petct.py \
    --dataset         "${DATASET}"    \
    --checkpoint_root "${SAVE_ROOT}"  \
    --save_path       "${RESULT_DIR}" \
    --modality        "${MODALITY}"   \
    --epoch           "${EPOCHS}"     \
    --image_size      "${IMAGE_SIZE}" \
    --device          "${GPU}"

echo ""
echo "================================================================"
echo "完成！模型保存在: ${SAVE_ROOT}/${DATASET}_${MODALITY}/epoch_${EPOCHS}.pth"
echo "测试结果保存在: ${RESULT_DIR}"
echo "================================================================"
