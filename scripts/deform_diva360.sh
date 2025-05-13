gpu_idx=$1
object_name=$2
idx_from=$3
idx_to=$4
cam_idx=$5
# wandb_group_name=$6

CUDA_VISIBLE_DEVICES=$gpu_idx python train.py -s data/Diva360/$object_name \
    --expname "Diva360/$object_name" \
    --configs arguments/Diva360/$object_name.py \
    --idx_from $idx_from \
    --idx_to $idx_to \
    --cam_idx $cam_idx

##                                       GPU object             idx_from idx_to cam_idx wandb_group
# bash scripts/finetuning_drag_diva360.sh 5   penguin            0217     0239   00      tmp

# bash scripts/deform_diva360.sh 6 penguin 0218 0239 00