#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="1"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="adapter"
dataset_name="cubicasa5k"  # Assuming you set this if it's dynamic
targets='multi_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
# Construct train and validation image list paths
img_folder="/home/admin/doku/datasets"  # Assuming this is the folder where images are stored
train_img_list="${img_folder}/${dataset_name}/train.csv"
val_img_list="${img_folder}/${dataset_name}/val.csv"

# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_encoderdecoder_${finetune_type}_${dataset_name}_noprompt"

# Run the Python script
python SingleGPU_train_finetune_noprompt.py \
    -num_cls 12 \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_mask_decoder_adapter True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "ckpts/sam_hq_vit_b.pth" \
    -targets "$targets" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list"
    # -if_encoder_adapter True \
    # -if_update_encoder True \