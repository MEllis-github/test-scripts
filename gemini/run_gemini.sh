set -x
# distplan in ["colossalai", "zero1", "zero2", "torch_ddp", "torch_zero"]
export DISTPLAN=${DISTPLAN:-"colossalai"}

export NTRAIN_STEPS=${NTRAIN_STEPS:-10}

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-1}
export TPDEGREE=${TPDEGREE:-1}
export PLACEMENT=${PLACEMENT:-"cpu"}
export USE_SHARD_INIT=${USE_SHARD_INIT:-False}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_medium"}
export NVME_OFFLOAD_FRACTION=${NVME_OFFLOAD_FRACTION:-"0.0"}
export NVME_OFFLOAD_DIR=${NVME_OFFLOAD_DIR:-"./"}

# export PYTHONPATH=$PWD:$PYTHONPATH

PYTORCH_PROFILE=${PYTORCH_PROFILE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"gemini_logs"}
mkdir -p ${OUTPUT_DIR}

torchrun --standalone --nproc_per_node=${GPUNUM} ./train_gpt_demo.py \
--num_train_steps=${NTRAIN_STEPS} \
--tp_degree=${TPDEGREE} \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--placement=${PLACEMENT} \
--shardinit=${USE_SHARD_INIT} \
--distplan=${DISTPLAN} \
--pytorch_profile=${PYTORCH_PROFILE} \
--output_dir=${OUTPUT_DIR} \
--nvme_offload_fraction=${NVME_OFFLOAD_FRAC} \
--nvme_offload_dir=${NVME_OFFLOAD_DIR} \
2>&1 | tee ${OUTPUT_DIR}/${MODEL_TYPE}_${DISTPLAN}_nsteps${NTRAIN_STEPS}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_tp_${TPDEGREE}_nvme_${NVME_OFFLOAD_FRAC}_${PLACEMENT}${USE_SHARD_INIT}.log
