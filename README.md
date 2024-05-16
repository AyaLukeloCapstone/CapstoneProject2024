# Engineering Capstone 2024: Aya El Mir and Lukelo Luoga

## Table of Contents

- [Capstone Project Overview](#capstone-project-overview)
- [Data Download](#data-download)
- [LLaVA-Med](#llava-med)
  - [Model Description](#model-description)
  - [Install dependencies for LLaVA-Med](#install-dependencies-for-llava-med)
  - [Training](#training)
  - [LLaVA-Med Evaluation](#llava-med-evaluation)
- [TinyLLaVA-Med](#tinyllava-med)
  - [Requirements and Installation](#requirements-and-installation)
  - [TinyLLaVA Models](#tinyllava-models)
  - [Demo](#demo)
  - [Train](#train)
  - [TinyLLaVA-Med Evaluation](#tinyllava-med-evaluation)

## Capstone Project Overview

Deploying advanced AI technologies like Multi-Modal Large Language Models (MLLMs) in healthcare poses challenges due to high computational demands and significant memory requirements, hindering their use on resource-constrained devices such as Nvidia Jetson Xavier. These limitations are critical in medical settings, particularly in remote areas where computational resources are scarce yet the demand for sophisticated medical diagnostics is high. To address these issues, our project explored two optimization methods: direct fine-tuning of a smaller model on a medical dataset and the development of a knowledge distillation approach. While the latter showed promise as an effective method for reducing model size and computational needs, it also presented significant challenges due to architectural mismatches between the 'teacher' model, LLaVA-Med, and the 'student' model, TinyLLaVA. Despite these challenges, the fine-tuning approach allowed us to maintain the critical functionalities of the original TinyLLAVA model, now optimized and renamed TinyLLAVA-Med. This model meets all specified design criteria, including efficient use of computational resources and significant reductions in computational complexity and power consumption. The optimized TinyLLAVA-Med makes advanced medical AI accessible on low-power devices, enhancing healthcare delivery and diagnostics in underserved regions.

## Data Download

<p align="center">
    <img src="images/llava_med_dataset.png" width="90%"> <br>
 
  *The data statistics of biomedical multimodal instruction-following data: (a,b) The root verb-noun pairs of instruction and responses, where the inner circle of the plot represents the root verb of the output response, and the outer circle represents the direct nouns. (c) The distribution of images and QA pairs on the five domains, one image is shown per domain.*
</p>

### Data Download

| Alignment data files                                                                                                            |       Size |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_alignment_500k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/alignment/llava_med_alignment_500k.json) | 341.52 MiB |

| Instruction-Tuning data files                                                                                                                            |       Size |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_instruct_10k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_10k.json)                               |  19.24 MiB |
| [llava_med_instruct_60k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k.json)                               |  84.65 MiB |
| [llava_med_instruct_60k_inline_mention.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json) |  83.61 MiB |
| [llava_med_instruct_fig_captions.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_fig_captions.json)             | 161.39 MiB |

| Evaluation files                                                                                                                                                                               |       Size |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_eval_qa50_qa.jsonl](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_qa.jsonl)                                                                       | 256.18 KiB |
| [llava_med_eval_qa50_fig_captions.json](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_fig_captions.json)                                                     |  51.82 KiB |
| [llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json) | 100.97 KiB |

| Image URLS                                                                                                      |       Size |
| --------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_image_urls.jsonl](https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl) | 122.82 MiB |

[download_images.py](llava/data/download_images.py) is used to download the PMC articles using the above image_urls file and extract the images

To download our langauge-image multimodal instruction-folllowing dataset, please run the following script:

```bash
sh download_data.sh
```

# LLaVA-Med

<p align="center">
    <img src="images/llava_med_logo.png" width="50%"> <br>
 
  *Generated by  <a href="https://gligen.github.io/">GLIGEN</a>  using the grounded inpainting mode, with three boxes: ``white doctor coat``, ``stethoscope``, ``white doctor hat with a red cross sign``.*
 
</p>

## Model Description

Large Language and Vision Assistant for bioMedicine (i.e., “LLaVA-Med”) is a large language and vision model trained using a curriculum learning method for adapting LLaVA to the biomedical domain. It is an open-source release intended for research use only to facilitate reproducibility of the corresponding paper which claims improved performance for open-ended biomedical questions answering tasks, including common visual question answering (VQA) benchmark datasets such as PathVQA and VQA-RAD.

## Install dependencies for LLaVA-Med

1. Clone this repository and navigate to LLaVA-Med folder

```bash
https://github.com/AyaLukeloCapstone/CapstoneProject2024.git
cd CapstoneProject2024
```

2. Install Package: Create conda environment

```Shell
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip  # enable PEP 660 support
```

3. Install additional packages for training cases

```Shell
pip uninstall torch torchvision -y
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install openai==0.27.8
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers@cae78c46
pip install -e .
```

```
pip install einops ninja open-clip-torch
pip install flash-attn --no-build-isolation
```

## Training

<p align="center">
    <img src="images/llava_med_pipeline.png" width="90%"> <br>
 
  *LLaVA-Med was initialized with the general-domain LLaVA and then continuously trained in a curriculum learning fashion (first biomedical concept alignment then full-blown instruction-tuning). We evaluated LLaVA-Med on standard visual conversation and question answering tasks.*
</p>

### Initialization from LLaVA-7B Weights

To ensure the smooth adaptation in terms of the multimodal chat capability, we initialize model weights from the general-domain [LLaVA](https://llava-vl.github.io/). The delta weights of LLaVA comply with the LLaMA model license. You can add the delta to the original LLaMA weights to obtain the LLaVA weights.

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get LLaVA weights ``LLaVA-7b-v0'' by applying our delta [LLaVA-7b-delta-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0)). It will automatically download delta weights from our Hugging Face account.

This conversion command needs around 30 GB of CPU RAM.

```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/LLaVA-7b-v0 \
    --delta /huggingface.co/liuhaotian/LLaVA-7b-delta-v0
```

### LLaVA-Med Training

LLaVA-Med is trained on 8 A100 GPUs with 40GB memory with the following code. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly to keep the global batch size the same.

#### - Stage 1 (Optional): Medical Concept Alignment

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| -------------- | ----------------: | ------------: | -----: | ---------: | -----------: |
| LLaVA-Med-7B   |               128 |          2e-3 |      1 |       2048 |            0 |

<details>
<summary>Pretrain: LLaVA-Med-7B, 8x A100 (40G).  Time: ~7 hours.</summary>

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llava-7b-v0 \
    --data_path /path/to/pubmed_600k.json \
    --image_folder /path/to/pubmed_600k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
```

</details>

You may run this with a single A100 GPU for the debugging purpose. Please note that the `per_device_train_batch_size` \* `gradient_accumulation_steps` can be reduced to load model checkpoint into GPU memory. But the decreased global batch size increase the total training.

#### - Stage 2: Medical Visual Instruct Tuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| -------------- | ----------------: | ------------: | -----: | ---------: | -----------: |
| LLaVA-Med-7B   |               128 |          2e-5 |      3 |       2048 |            0 |

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/llama-med-vicuna-7b \
    --data_path /path/to/llava_med_instruct_60k_inline_mention_post.jsonl \
    --image_folder /data/to/llava_med_instruct_images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /path/to/checkpoint_llava_med_instruct_60k_inline_mention \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

You may directly perform medical instruction tuning on [`medical instruct data`](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json), by skipping Stage 1, and replacing Stage-1 checkpoint with the pretrained LLaVA checkpoint (LLaVA-7b-v0). Please see an example running script at [`run_training_llava_med.sh`](scripts/chunyl/run_training_llava_med.sh)

## LLaVA-Med Evaluation

### Medical Visual Chat (GPT-assisted Evaluation)

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models. Please see our paper for more details.

1. Generate LLaVA-Med responses

```Shell
python model_vqa.py \
    --model-name ./checkpoints/LLaVA-7B-v0 \
    --question-file data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder data/images/ \
    --answers-file /path/to/answer-file.jsonl
```

2. Evaluate the generated responses. In our case, [`llava_med_eval_qa50_qa.jsonl`](/data/eval/llava_med_eval_qa50_qa.jsonl) contains the questions, context (captions and inline-mentions) and responses generated by text-only GPT-4 (0314), which we treat as ground truth.

```Shell
python llava/eval/eval_multimodal_chat_gpt_score.py \
    --question_input_path data/eval/llava_med_eval_qa50_qa.jsonl \
    --input_path /path/to/answer-file.jsonl \
    --output_path /path/to/save/gpt4-eval-for-individual-answers.jsonl
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```

### Medical VQA

Three Medical VQA datasets are considered in our experiments, including VQA-Rad, SLAKE, Pathology-VQA. We use VQA-Rad as the running example to illustrate how LLaVA-Med is applied to a downstream scenario.

#### - Prepare Data

1. Please see VQA-Rad [repo](https://paperswithcode.com/dataset/vqa-rad) for setting up the dataset.
2. Generate VQA-Rad dataset for LLaVA-Med conversation-style format (the same format with instruct tuning). For each dataset, we process it into three components: `train.json`, `test.json`, `images`.

#### - Fine-tuning

To achieve the higher performance for given a downstream dataset, the same full-model tuning script with instruct tuning is used to continue train LLaVA-Med.

<details>
<summary> Detailed script to fine-tune to downstream datasets: LLaVA-Med-7B, 8x A100 (40G).  Time: ~1 hour.</summary>

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/checkpoint_llava_med_instruct_60k_inline_mention \
    --data_path /path/to/eval/vqa_rad/train.json \
    --image_folder /path/to/eval/vqa_rad/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

</details>

#### - Evaluation

Depending on which checkpoint is employed in evaluation, zero-shot performance is reported on medical instruct tuned checkpoint (eg, [LLaVA-Med-7B](/path/to/checkpoint_llava_med_instruct_60k_inline_mention)), and fine-tuned performance is reported on checkpoint that has been further tuned on training set of the downstream datasets (eg, [LLaVA-Med-7B-VQA-Rad](/path/to/checkpoint_llava_med_instruct_60k_inline_mention/fine_tuned/vqa_rad) ).

(a) Generate LLaVA responses on ScienceQA dataset

(a.1). [Option 1] Multiple-GPU inference
You may evaluate this with multiple GPUs, and concatenate the generated jsonl files. Please refer to our script for [batch evaluation](scripts/chunyl/finetune_on_benchmarks/eval_med_dataset_batch.sh).

```Shell
python llava/eval/run_med_datasets_eval_batch.py --num-chunks 8  --model-name /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --question-file path/to/eval/vqa_rad/test.json \
    --image-folder path/to/eval/vqa_rad/images \
    --answers-file /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

(a.2). [Option 2] Single-GPU inference

```Shell
python llava/eval/model_vqa_med.py --model-name /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --question-file path/to/eval/vqa_rad/test.json \
    --image-folder path/to/eval/vqa_rad/images \
    --answers-file /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

(b) Evaluate the generated responses

(b.1). [Option 1] Evaluation for all three VQA datasets

```Shell

python llava/eval/run_eval_batch.py \
    --pred_file_parent_path /path/to/llava-med \
    --target_test_type test-answer-file
```

It collects the decoding results of all predictions files under the project path, computes the corresponding evaluation metrics, and outputs the results in "`eval_results_med_datasets.jsonl`". To analyze the score, we provdie ipython notebook [run_eval_metrics.ipynb](llava/notebook/run_eval_metrics.ipynb).

(b.2). [Option 2] Evaluation for on one specific VQA dataset

```Shell
python llava/eval/run_eval.py \
    --gt /path/to/eval/vqa_rad/test.json \
    --pred /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

Please find the LLaVA-Med performance in [llava_med_performance.md](docs/llava_med_performance.md) or in the paper.

# TinyLLaVA-Med

## Requirements and Installation

We recommend the requirements as follows.

1. Clone this repository and navigate to LLaVA folder

```bash
cd TinyLLaVA-Med
```

2. Install Package

```Shell
conda create -n tinyllava python=3.10 -y
conda activate tinyllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases

```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to the latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

## TinyLLaVA Models

### Legacy Model

- [tiny-llava-hf](https://huggingface.co/bczhou/tiny-llava-v1-hf)

### Pretrained Models

- [TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B)
- [TinyLLaVA-2.0B](https://huggingface.co/bczhou/TinyLLaVA-2.0B)
- [TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B)

### Model Details

| Name           | LLM             | Checkpoint                                                     | LLaVA-Bench-Wild | MME    | MMBench | MM-Vet | SQA-image | VQA-v2 | GQA  | TextVQA |
| -------------- | --------------- | -------------------------------------------------------------- | ---------------- | ------ | ------- | ------ | --------- | ------ | ---- | ------- |
| TinyLLaVA-3.1B | Phi-2           | [TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B) | 75.8             | 1464.9 | 66.9    | 32.0   | 69.1      | 79.9   | 62.0 | 59.1    |
| TinyLLaVA-2.0B | StableLM-2-1.6B | [TinyLLaVA-2.0B](https://huggingface.co/bczhou/TinyLLaVA-2.0B) | 66.4             | 1433.8 | 63.3    | 32.6   | 64.7      | 78.9   | 61.9 | 56.4    |
| TinyLLaVA-1.5B | TinyLlama       | [TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B) | 60.8             | 1276.5 | 55.2    | 25.8   | 60.3      | 76.9   | 60.3 | 51.7    |

## Demo

### Gradio Web Demo

Launch a local web demo by running:

```shell
python tinyllava/serve/app.py --model-path bczhou/TinyLLaVA-3.1B --model-name TinyLLaVA-3.1B
```

## Train

### Instruction Tuning

```Shell
DATA_PATH= /path/to/llava_med_instruct_60k_inline_mention_post.jsonl \
IMAGE_PATH= /path/to/your-image-folder

LLM_VERSION=bczhou/TinyLLaVA-1.5B
VT_VERSION=bczhou/TinyLLaVA-1.5B-SigLIP

output_directory= path to the checkpoints output folder
wandb_path= Tinyllava_SIGLIP_SG3_EX3_VQARAD

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model True \
    --bf16 True \
    --output_dir $output_directory \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $wandb_path
```

## TinyLLaVA-Med Evaluation

### Medical VQA

Three Medical VQA datasets are considered in our experiments, including VQA-Rad, SLAKE, Pathology-VQA. We use VQA-Rad as the running example to illustrate how LLaVA-Med is applied to a downstream scenario.

#### - Prepare Data

1. Please see VQA-Rad [repo](https://paperswithcode.com/dataset/vqa-rad) for setting up the dataset.
2. Generate VQA-Rad dataset for TinyLLaVA-Med conversation-style format (the same format with instruct tuning). For each dataset, we process it into three components: `train.json`, `test.json`, `images`.

#### - Fine-tuning

<details>
<summary> Detailed script to fine-tune to downstream datasets: TinyLLaVA-Med-1.5B. </summary>

```Shell
DATA_PATH= /path/to/your-VQA-train-json-file \
IMAGE_PATH= /path/to/your-VQA-image-folder

output_directory= path to the checkpoints output folder
wandb_path= Tinyllava_SIGLIP_SG3_EX3_VQARAD

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model True \
    --bf16 True \
    --output_dir $output_directory \
    --num_train_epochs 18 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $wandb_path
```

</details>

#### - Evaluation

(a) Generate TinyLLaVA-Med responses on ScienceQA dataset

```Shell
python tinyllava/eval/model_vqa_med.py --model-name /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --question-file path/to/eval/vqa_rad/test.json \
    --image-folder path/to/eval/vqa_rad/images \
    --answers-file /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

(b) Evaluate the generated responses

```Shell
python ../llava/eval/run_eval.py \
    --gt /path/to/eval/vqa_rad/test.json \
    --pred /path/to/checkpoint_tinyllava/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

## Acknowledgement

- Our project is built upon [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) and [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory). They provided us the code, base models, and dataset with the amazing multimodal and langauge capabilities!
