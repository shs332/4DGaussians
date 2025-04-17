#!/bin/bash
set -e

OUTPUT_DIR="/data2/wlsgur4011/GESI/4DGaussians/data"
SRC_ROOT="/data/rvi/dataset/Diva360_data/processed_data"

mkdir -p "$OUTPUT_DIR/multipleview"

for object_name in $(ls "$SRC_ROOT"); do

    # if [[ "$object_name" < "poker" ]]; then # "poker"보다 작은 이름을 가진 객체는 건너뛰기
    #     echo "Skipping object: $object_name (comes before poker)"
    #     continue
    # fi

    echo -e "Processing object diva360: $object_name\n"

    OBJ_SRC="$SRC_ROOT/$object_name/frames_1"
    OBJ_DST="$OUTPUT_DIR/multipleview/$object_name"
    mkdir -p "$OBJ_DST"

    cam_idx=1
    for cam_folder in $(ls "$OBJ_SRC"); do # 원본 : cam00, cam01, cam04, ...
        # cam_idx를 이용해 cam01, cam02, cam03, ... 생성
        echo "Processing camera: $cam_folder"

        new_cam=$(printf "cam%02d" "$cam_idx")
        CAM_SRC="$OBJ_SRC/$cam_folder"
        CAM_DST="$OBJ_DST/$new_cam"
        mkdir -p "$CAM_DST"

        for img in "$CAM_SRC"/*.png; do
            img_name=$(basename "$img")
            # 10#을 써서 8진수 해석 방지, +1 후 5자리 0패딩
            raw_num=$(echo "$img_name" | sed -n 's/^\([0-9]\{8\}\)\.png$/\1/p')
            num=$((10#$raw_num + 1))
            frame_num=$(printf "%05d" "$num")

            # echo "image : $img, link : $CAM_DST/frame_${frame_num}.png"
            ln -sf "$img" "$CAM_DST/frame_${frame_num}.png" ## 원본 소프트링크 덮어쓰기
        done

        ((cam_idx++))
    done
done

echo "심볼릭 링크 생성 완료: $OUTPUT_DIR/multipleview_idxfrom1"
