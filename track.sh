#!/bin/bash

# 设置索引范围，例如从1到10
start_index=1
end_index=9

# 遍历指定的索引范围
for i in $(seq $start_index $end_index); do
    # 设置输入视频路径
    input_video="/home/bw2716/SeeUnsafe/data/video_${i}.mp4"
    # 设置输出视频路径
    output_video="/home/bw2716/SeeUnsafe/data/video_${i}_track.mp4"
    
    # 运行 Python 脚本
    python track_objects.py --input "$input_video" --output "$output_video" --num_key_frames 9 --bbx_file /home/bw2716/SeeUnsafe/bbx_file.csv --index_file /home/bw2716/SeeUnsafe/index_file.csv
done
