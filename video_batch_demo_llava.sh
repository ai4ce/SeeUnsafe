#!/bin/bash
ROOT_DIR="PATH TO YOUR LLAVA-NEXT"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
POOL_MODE=$5
NEWLINE_POSITION=$6
OVERWRITE=$7
INPUT_DIR=$8
OUTPUT_DIR=$9

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory does not exist. Exiting the script."
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

for VIDEO_PATH in "$INPUT_DIR"/*.mp4; do
    if [ "$OVERWRITE" = False ]; then
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}
    else
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
    fi

    OUTPUT_FILE="$OUTPUT_DIR/$(basename "${VIDEO_PATH%.mp4}.txt")"

    python3 playground/demo/video_demo.py \
        --model-path $CKPT \
        --video_path ${VIDEO_PATH} \
        --output_dir ./work_dirs/video_demo/$SAVE_DIR \
        --output_name pred \
        --chunk-idx $(($IDX - 1)) \
        --overwrite ${OVERWRITE} \
        --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
        --for_get_frames_num $FRAMES \
        --conv-mode $CONV_MODE \
        --mm_spatial_pool_mode ${POOL_MODE:-average} \
        --mm_newline_position ${NEWLINE_POSITION:-grid} \
        --prompt "This video depicts a traffic event. Please choose one from normal, near-miss, and collision that can best describe the given video. If so, what types of road users are involved, and in what context do these traffic anomalies occur? Respond in the format of the following example without any additional information: Video Class: type of integer indicating the video class. For example: 0 for near-miss, 1 for collision, and 2 for normal. Answer with only one integer. Object Detail: type of string describing the appearance of road users. For example: The involved road users are a middle-aged male pedestrian and a white sedan. Scene Context: type of string describing the scene environment. For example: Sunny day with dry road surface in a curbside area near an intersection, under daytime lighting with moderate mixed traffic. Justification: type of string explaining the reason why a traffic anomaly occurs or why a traffic anomaly doesn't exist. For example: The pedestrian's pose changes from walking to a sudden stop, indicating an unexpected reaction to the vehicle. The vehicle shows a significant deviation from its original trajectory, indicating that it misjudged the pedestrian's path. This heavy deviation led to the vehicle failing to avoid the pedestrian, resulting in a collision." \
        > "$OUTPUT_FILE"

    echo "Processed $VIDEO_PATH -> $OUTPUT_FILE"
done