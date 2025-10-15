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
# salloc --time=00:20:00 --nodes=1 --gres=gpu:4 --ntasks-per-node=1 --cpus-per-task=32 --account=p200848 --partition=gpu --qos=test

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
torchrun --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --dataset_path $DATA_PATH \
    --batch_size 24 \
    --total_batch_size 1152 \
    --lr 5e-4 \
    --max_length 512 \
    --save_every 1000 \
    --eval_every 1000 \
    --num_training_steps 20000 \
    --tags warm_start_60M
