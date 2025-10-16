#!/bin/bash -l
#SBATCH --account=EUHPC_D18_005
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=48:00:00
#SBATCH --chdir=/leonardo_scratch/fast/EUHPC_D18_005/david/relora-3000
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/relora_from_scratch.out
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/relora_from_scratch.err

# See running jobs
# watch -n 1 squeue --me

# Interactive mode
# salloc --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --account=EUHPC_D18_005 --partition=boost_usr_prod
# srun --pty /bin/bash

# Set HF home directory for offline datasets
export HF_HOME="/leonardo_work/EUHPC_D18_005/david/hf-datasets-cache"

# Change directory to where the script is located
cd /leonardo_scratch/fast/EUHPC_D18_005/david/relora-3000

# Activate the virtual environment
source venv/bin/activate

# Load required modules
module load python/3.11.7

# Set data path
export DATA_PATH=preprocessed_data/c4_en_t5-base_512

#WANDB offfline
export WANDB_MODE=offline

# HF offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run the torchrun script
torchrun --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --batch_size 24 \
    --total_batch_size 1152 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft True \
    --relora 5000 \
    --cycle_length 5000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_relora True \
    --num_training_steps 20000 \
    --save_every 5000 \
    --eval_every 5000 \
    --tags relora_from_scratch_60M \
    --dataset_path preprocessed_data/c4_en_t5-base_512
