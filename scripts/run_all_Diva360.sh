#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

script_name=$1
wandb_group_name=$2

# script_name=finetuning_drag_diva360.sh
# wandb_group_name=tmp

bash scripts/${script_name} 0   wall_e          0222     0286   00      ${wandb_group_name} &&
bash scripts/${script_name} 0   blue_car        0142     0214   00      ${wandb_group_name} &&
bash scripts/${script_name} 0   k1_hand_stand   0412     0426   01      ${wandb_group_name} &

bash scripts/${script_name} 1   stirling        0000     0045   00      ${wandb_group_name} &&
bash scripts/${script_name} 1   world_globe     0020     0074   00      ${wandb_group_name} &&
bash scripts/${script_name} 1   music_box       0100     0125   00      ${wandb_group_name} &

bash scripts/${script_name} 2   trex            0135     0250   00      ${wandb_group_name} &&
bash scripts/${script_name} 2   k1_double_punch 0270     0282   01      ${wandb_group_name} &&
bash scripts/${script_name} 2   dog             0177     0279   00      ${wandb_group_name} &&
bash scripts/${script_name} 2   wolf            0357     1953   00      ${wandb_group_name} &

bash scripts/${script_name} 3   red_car         0042     0250   00      ${wandb_group_name} &&
bash scripts/${script_name} 3   tornado         0000     0456   00      ${wandb_group_name} &&
bash scripts/${script_name} 3   truck           0078     0171   00      ${wandb_group_name} &

bash scripts/${script_name} 4   clock           0000     1500   00      ${wandb_group_name} &&
bash scripts/${script_name} 4   horse           0120     0375   00      ${wandb_group_name} &&
bash scripts/${script_name} 4   bunny           0000     1000   00      ${wandb_group_name} &

bash scripts/${script_name} 5   hour_glass      0100     0200   00      ${wandb_group_name} &&
bash scripts/${script_name} 5   k1_push_up      0541     0557   01      ${wandb_group_name} &&
bash scripts/${script_name} 5   penguin         0217     0239   00      ${wandb_group_name} &


wait