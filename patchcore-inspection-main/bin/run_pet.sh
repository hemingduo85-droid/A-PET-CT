#!/usr/bin/env bash
# Run PatchCore with single-modal PET input on the AutoPET/PSMA dataset.
#
# Usage:
#   bash bin/run_pet.sh [gpu_id] [seed]
#
# Examples:
#   bash bin/run_pet.sh 0 0   # GPU 0, seed 0
#   bash bin/run_pet.sh 1 42  # GPU 1, seed 42
#
# The script must be run from the repository root, e.g.:
#   cd /data/cyf/codes/A-PET-CT/patchcore-inspection-main
#   bash bin/run_pet.sh 0 0

# cd /data/cyf/codes/A-PET-CT/patchcore-inspection-main

# # 单模态 PET
# bash bin/run_pet.sh 3 42    # GPU 0, seed 0

# # 单模态 CT
# bash bin/run_ct.sh 3 42     # GPU 1, seed 42

set -euo pipefail

GPU=${1:-0}
SEED=${2:-0}

DATAPATH=/data/cyf/shared_data/PET-CT/AutoPET/2d_equal/PSMA
RESULTS_ROOT=/data/cyf/codes/A-PET-CT/patchcore-inspection-main/results
LOG_GROUP=PET_PSMA_IM256_WR50_L2-3_P01_D1024-1024_PS3_AN1_SingleModal_S${SEED}
LOG_PROJECT=PetCT_Results

echo "=========================================="
echo " PatchCore  — Single-Modal PET"
echo " GPU: ${GPU}  |  Seed: ${SEED}"
echo " Data: ${DATAPATH}"
echo " Results: ${RESULTS_ROOT}/${LOG_PROJECT}/${LOG_GROUP}"
echo "=========================================="

PYTHONPATH=src python bin/run_patchcore.py \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --save_patchcore_model \
  --save_segmentation_images \
  --log_group "${LOG_GROUP}" \
  --log_project "${LOG_PROJECT}" \
  "${RESULTS_ROOT}" \
  \
  patch_core \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --faiss_on_gpu \
    --pretrain_embed_dimension 1024 \
    --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 \
    --patchsize 3 \
  \
  sampler \
    -p 0.1 approx_greedy_coreset \
  \
  dataset \
    --resize 256 \
    --imagesize 256 \
    --batch_size 8 \
    --num_workers 4 \
    --modality pet \
    -d psma \
    petct "${DATAPATH}"
