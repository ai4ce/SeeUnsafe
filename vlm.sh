#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_index> <end_index>"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2
INDEX_FILE="/home/bw2716/SeeUnsafe/index_file.csv"
OUTPUT_PATH="/home/bw2716/SeeUnsafe/answer.csv"


for ((i=START_INDEX; i<=END_INDEX; i++))
do
    INPUT_VIDEO="/home/bw2716/SeeUnsafe/data/video_${i}_track.mp4"

    LIST=$(grep "^$INPUT_VIDEO," "$INDEX_FILE" | sed -n 's/^[^,]*,\(.*\)/\1/p' | tr -d '"')


    if [ -z "$LIST" ]; then
        echo "No list found for $INPUT_VIDEO in $INDEX_FILE"
        continue
    fi

    echo "Running: python vlm.py --input $INPUT_VIDEO --list $LIST --output $OUTPUT_PATH"
    python vlm.py --input "$INPUT_VIDEO" --list "$LIST" --output "$OUTPUT_PATH"
done
