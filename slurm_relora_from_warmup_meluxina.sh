#!/bin/bash -l
#SBATCH --account=p200848
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=48:00:00
#SBATCH --chdir=/project/home/p200848/david/relora-3000
#SBATCH --output=/project/home/p200848/david/outputs/torchrun.out
#SBATCH --error=/project/home/p200848/david/outputs/torchrun.err

# Interactive mode
# salloc --time=00:20:00 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=32 --account=p200848 --partition=gpu --qos=test

# Set HF home directory for offline datasets
export HF_HOME="/project/home/p200848/david/huggingface"

# Change directory to where the script is located
cd /project/home/p200848/david/relora-3000

# Activate the virtual environment
source venv/bin/activate

# Load required modules
module load Python/3.11.10-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0

# Set data path
export DATA_PATH=preprocessed_data/c4_en_t5-base_512

# Run the torchrun script
torchrun --nproc-per-node 1 torchrun_main.py \
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
    --warmed_up_model checkpoints/warm_start_60M_20251015_211428/model_5000 \
    --tags relora_60M \
    --dataset_path preprocessed_data/c4_en_t5-base_512
