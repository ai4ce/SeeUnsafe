#!/bin/bash

start_index=1
end_index=9

for i in $(seq $start_index $end_index); do
    input_video="/home/bw2716/SeeUnsafe/data/video_${i}.mp4"
    output_video="/home/bw2716/SeeUnsafe/data/video_${i}_track.mp4"
    
    python track_objects.py --input "$input_video" --output "$output_video" --num_key_frames 9 --bbx_file /home/bw2716/SeeUnsafe/bbx_file.csv --index_file /home/bw2716/SeeUnsafe/index_file.csv
done
