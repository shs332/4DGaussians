#!/bin/bash

# 원하는 구조로 심볼릭 링크를 생성하는 스크립트입니다.
# 사용법: bash make_multipleview_symlinks.sh <output_dir>
# 예시: bash make_multipleview_symlinks.sh /data2/wlsgur4011/GESI/myfolder

set -e

# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <output_dir>"
#     exit 1
# fi

OUTPUT_DIR="/data2/wlsgur4011/GESI/4DGaussians/data"
SRC_ROOT="/data/rvi/dataset/Diva360_data/processed_data"

mkdir -p "$OUTPUT_DIR/multipleview"

for object_name in $(ls "$SRC_ROOT"); do
    OBJ_SRC="$SRC_ROOT/$object_name/frames_1"
    OBJ_DST="$OUTPUT_DIR/multipleview/$object_name"
    mkdir -p "$OBJ_DST"
    for cam_folder in $(ls "$OBJ_SRC"); do
        CAM_SRC="$OBJ_SRC/$cam_folder"
        CAM_DST="$OBJ_DST/$cam_folder"
        mkdir -p "$CAM_DST"
        for img in "$CAM_SRC"/*.png; do
            img_name=$(basename "$img")
            # frame_00001.jpg 형태로 변환
            frame_num=$(echo "$img_name" | sed -n 's/^.*\([0-9]\{5\}\)\.png$/\1/p')
            # echo "image : $img, link : $CAM_DST/frame_${frame_num}.png"
            ln -s "$img" "$CAM_DST/frame_${frame_num}.png"
        done
    done
done

echo "심볼릭 링크 생성 완료: $OUTPUT_DIR/multipleview"