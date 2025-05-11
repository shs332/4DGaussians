object_name=$1
GPU_ID=$2

export CUDA_VISIBLE_DEVICES=$2&&python train.py -s data/Diva360/$object_name --expname "$Diva360/$object_name" --configs arguments/Diva360/$object_name.py