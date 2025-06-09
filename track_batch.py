import os

# Import the main function from track_objects.py
# Adjust the import statement based on the actual implementation
from track_objects import main as track_objects_main

def track_batch(original_dir, output_dir):
    """
    Process all videos in the original directory and run the `track_objects` function
    for each video, saving the output to the output directory with "_track" appended
    to the filename.

    Args:
        original_dir (str): Path to the directory containing the original video files.
        output_dir (str): Path to the output directory where processed videos will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process video files
    for filename in os.listdir(original_dir):
        if filename.endswith(".mp4"):
            original_video_path = os.path.join(original_dir, filename)
            output_filename = os.path.splitext(filename)[0] + "_track.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # Prepare arguments for track_objects_main
            args = [
                '--input', original_video_path,
                '--output', output_path,
                '--num_key_frames', '9',
                '--bbx_file', 'bbx.csv',
                '--index_file', 'index.csv'
            ]

            try:
                # Call the main function of track_objects.py directly
                track_objects_main(args)
                print(f"Processed: {original_video_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {original_video_path}: {e}")

if __name__ == "__main__":
    # Input and output directory paths
    original_folder = "/home/bw2716/SeeUnsafe/single_view_test_data/val_normal_reverse"
    output_folder = "/home/bw2716/SeeUnsafe/single_view_test_data/val_normal_reverse_track"

    track_batch(original_folder, output_folder)
