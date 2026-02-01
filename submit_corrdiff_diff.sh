#!/bin/bash

#SBATCH --account nesccmgmt
#SBATCH --qos=admin
#SBATCH --partition=o-h100

#SBATCH -J corrdiff-train
#SBATCH -o xxrrdiff_stdout_%J.txt
#SBATCH -e yyrrdiff_stderr_%J.txt

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one launcher task per node
#SBATCH --cpus-per-task=8            # increased for 2 GPUs
#SBATCH --gres=gpu:2                 # 2 GPUs per node
#SBATCH --mem=256G                   # Request 256GB RAM per node

#SBATCH -t 8:00:00
#SBATCH --export=ALL

# ===================================================================
# Weights & Biases Configuration
# ===================================================================
export WANDB_MODE=offline

# ===================================================================
# Threading and Distributed Setup
# ===================================================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# --- NCCL / rendezvous ---
#export NCCL_DEBUG=INFO  # Uncomment for debugging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Rendezvous for torchrun
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "==================================================================="
echo "SLURM Job Information"
echo "==================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_NNODES"
echo "Node List: $SLURM_NODELIST"
echo ""
echo "Distributed Training Configuration:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "Total GPUs: 4 (2 nodes × 2 GPUs)"
echo "==================================================================="

echo "Starting at $(date)"
startTime=$(date +%s)

# --- GPU sanity check on all nodes ---
echo "Checking GPU availability on all nodes..."
srun --ntasks-per-node=1 \
     --gres=gpu:2 \
     bash -c 'echo "Host: $(hostname)"; nvidia-smi -L'

echo ""
echo "==================================================================="
echo "Starting CorrDiff Distributed Training (4 GPUs total)"
echo "==================================================================="

# Launch torchrun - one per node, each spawns 2 processes (1 per GPU)
srun --ntasks-per-node=1 \
     --gres=gpu:2 \
  bash -c "
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun \
      --nnodes=${SLURM_NNODES} \
      --nproc_per_node=2 \
      --node_rank=\${SLURM_NODEID} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      --rdzv_id=${SLURM_JOB_ID} \
      train.py --config-name=config_training_hrrr_mini_diffusion.yaml   ++training.io.regression_checkpoint_path=/tds_scratch2/SYSADMIN/nesccmgmt/Ron.Millikan/devl/corrdiff/corrdiff2/checkpoints_regression/UNet.0.2000128.mdlus
    "

training_exit_code=$?

stopTime=$(date +%s)
echo ""
echo "==================================================================="
echo "Job Summary"
echo "==================================================================="
echo "Job completed at: $(date)"
echo "Total runtime: $((stopTime-startTime)) seconds"
echo "Training exit code: $training_exit_code"

if [ $training_exit_code -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Generated files:"

    if [ -d "wandb" ]; then
        echo "  - W&B runs: wandb/"
    fi
    if [ -d "checkpoints_regression" ]; then
        echo "  - Regression checkpoints: checkpoints_regression/"
    fi
    if [ -d "checkpoints_diffusion" ]; then
        echo "  - Diffusion checkpoints: checkpoints_diffusion/"
    fi
else
    echo "✗ Training failed with exit code: $training_exit_code"
    echo "Check logs:"
    echo "  - corrdiff_stdout_${SLURM_JOB_ID}.txt"
    echo "  - corrdiff_stderr_${SLURM_JOB_ID}.txt"
fi

echo "==================================================================="
