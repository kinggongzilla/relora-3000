#!/bin/bash -l
#SBATCH --account=p200848
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --time=48:00:00
#SBATCH --chdir=/project/home/p200848/david/relora-3000
#SBATCH --output=/project/home/p200848/david/outputs/pretokenize.out
#SBATCH --error=/project/home/p200848/david/outputs/pretokenize.err

# See running jobs
# watch -n 1 squeue --me

# Interactive mode
# salloc --time=00:20:00 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --account=p200848 --partition=gpu --qos=test

# Set HF home directory for offline datasets
export HF_HOME="/project/home/p200848/david/huggingface"

# Change directory to where the script is located
cd /project/home/p200848/david/relora-3000

# Activate the virtual environment
source venv/bin/activate

# Load required modules
module load Python/3.11.10-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0

# Run the pretokenize script
python pretokenize.py \
    --save_dir preprocessed_data \
    --tokenizer t5-base \
    --dataset c4 \
    --dataset_config en \
    --text_field text \
    --sequence_length 512 \
    --take 40000000
