#!/bin/bash

# 参数检查，确保传入了范围参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_index> <end_index>"
    exit 1
fi

# 定义参数
START_INDEX=$1
END_INDEX=$2
INDEX_FILE="/home/bw2716/SeeUnsafe/index_file.csv"
OUTPUT_PATH="/home/bw2716/SeeUnsafe/answer.csv"

# 遍历指定范围的 i 值
for ((i=START_INDEX; i<=END_INDEX; i++))
do
    # 构建 input 视频文件路径
    INPUT_VIDEO="/home/bw2716/SeeUnsafe/data/video_${i}_track.mp4"

    # 从 CSV 文件中查找对应的 list
    LIST=$(grep "^$INPUT_VIDEO," "$INDEX_FILE" | sed -n 's/^[^,]*,\(.*\)/\1/p' | tr -d '"')


    # 检查是否找到了 list
    if [ -z "$LIST" ]; then
        echo "No list found for $INPUT_VIDEO in $INDEX_FILE"
        continue
    fi

    # 运行 Python 脚本
    echo "Running: python vlm.py --input $INPUT_VIDEO --list $LIST --output $OUTPUT_PATH"
    python vlm.py --input "$INPUT_VIDEO" --list "$LIST" --output "$OUTPUT_PATH"
done
