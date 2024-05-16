#!/bin/bash 

#SBATCH -q ebrainccs 
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3  # Request only one GPU as we are running in single-process mode
#SBATCH --time=50:59:59
#SBATCH --mem=150GB
#SBATCH -C 80g
#SBATCH -o job_tiny_llava_1.5B_KD_finetune_SLAKE.%J.out
#SBATCH -e job_tiny_llava_1.5B_KD_finetune_SLAKE.%J.err

#SBATCH --job-name=FSSTuneVQA

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyllava

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"

export LD_LIBRARY_PATH=/home/ltl2113/miniconda3/envs/tinyllava/lib:$LD_LIBRARY_PATH

LLM_VERSION=/scratch/ltl2113/LLaVA-Med/TinyLLaVABench_1/checkpoints/tiny-llava-KD-pretrain/checkpoint-6594
VT_VERSION=google/siglip-so400m-patch14-384

DATA_PATH=/scratch/ltl2113/LLaVA-Med/data/Vqa/Slake/train.json
IMAGE_PATH=/scratch/ltl2113/LLaVA-Med/data/Vqa/Slake/imgs
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

#TRAIN STAGE 1
deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model False \
    --fp16 True \
    --output_dir /scratch/ltl2113/LLaVA-Med/TinyLLaVABench/checkpoints/Tinny_llava_KD_SLAKE \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name tiny-llava-base-pretrain-"${LLM_VARIANT}"-"${VT_VARIANT}"