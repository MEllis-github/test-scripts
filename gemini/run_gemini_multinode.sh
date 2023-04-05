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

# export PYTHONPATH=$PWD:$PYTHONPATH

PYTORCH_PROFILE=${PYTORCH_PROFILE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"gemini_logs"}
mkdir -p ${OUTPUT_DIR}

torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=${GPUNUM} --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
./train_gpt_demo.py \
--num_train_steps=${NTRAIN_STEPS} \
--tp_degree=${TPDEGREE} \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--placement=${PLACEMENT} \
--shardinit=${USE_SHARD_INIT} \
--distplan=${DISTPLAN} \
--pytorch_profile=${PYTORCH_PROFILE} \
--output_dir=${OUTPUT_DIR} \
2>&1 | tee ${OUTPUT_DIR}/${MODEL_TYPE}_${DISTPLAN}_nsteps${NTRAIN_STEPS}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_tp_${TPDEGREE}_${PLACEMENT}${USE_SHARD_INIT}.log
