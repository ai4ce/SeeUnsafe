# When language and vision meet road safety: Leveraging multimodal large language models for video-based traffic accident analysis (SeeUnsafe)

**SeeUnsafe** is an MLLM-integrated framework for traffic accident analysis.

![main](https://github.com/ai4ce/SeeUnsafe/blob/main/media/main.jpg)

## Setup Instructions

SeeUnsafe relies on several open-source projects, including: GroundingDINO, SAM, SAM2, LLaVA-NeXT. The code has been tested on Ubuntu 22.04, with CUDA version 11.8, PyTorch version 2.3.1+cu118.

- Install SeeUnsafe and create a new environment

```python
git clone https://github.com/ai4ce/SeeUnsafe
conda create --name SeeUnsafe python=3.10
conda activate SeeUnsafe
cd SeeUnsafe
pip install -r requirements.txt
```

- Install PyTorch
```python
pip install torch==2.3.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install GroundingDINO, SAM, SAM2 in the same environment
```python
git clone https://github.com/IDEA-Research/GroundingDINO
git clone https://github.com/facebookresearch/segment-anything.git
git clone https://github.com/facebookresearch/segment-anything-2.git
```

Make sure GroundingDINO and SAM are installed in editable packages by `pip install -e .` Notice that in the the setup of SAM2 will automatically update PyTorch and CUDA. To maintain the required versions for SeeUnsafe, use `pip install -e . --no-deps`

- We have slightly modified the GroundingDINO

In `GroundingDINO/groundingdino/util/inference.py`, we add a function to help inference on an array of images. Please paste the following function into `inference.py`.

```python
def load_image_from_array(image_array: np.array) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image_array)
    image_transformed, _ = transform(image_source, None)
    return image_array, image_transformed
```

- The code still uses one checkpoint from segment-anything.

Make sure you download it in the SeeUnsafe folder.
**`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

- The current official checkpoint in segment-anything-2 uses sam2.1, while we are still using sam2.0. Therefore, you need to manually download the sam2.0 checkpoint into the segment-anything-2/checkpoints directory. Please do NOT use the download script in the folder checkpoints!
**sam2 ckpt: [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)**


- Obtain an OpenAI API key and create a `key.py` file under path SeeUnsafe/
```python
touch key.py
echo 'projectkey="YOUR_OPENAI_API_KEY"' > key.py
```

## Main Logic Pipeline

There are two critical parts of SeeUnsafe, *frame-wise information augmentation* part and *task-specific mLLM evaluation* part.

1. **Frame-Wise Information Augmentation**
   This part integrates various computer vision models to extract the locations of cars and pedestrians, segment them, and add visual prompts in different colors to facilitate understanding by the multimodal large language model (mLLM).
   
   `track_objects.py`: takes an MP4 video file and the number of key frames as input. It adds visual prompts to cars and pedestrians in the video, uniformly samples the specified number of key frames, and saves the indices of the key frames along with the center coordinates of the detected objects in those frames.
   
   `--input`: the input video path (mp4)
   
   `--output`: output path to the video with augmented visual prompt (mp4)
   
   `--num_key_frames`: number of keyframes for normal sampling (int)
   
   `--bbx_file`: file to store the center coordinates of the detected objects in keyframes (csv).

   This bbx storage is intended for further study and is not actually used in SeeUnsafe. Feel free to delete this input as needed.
   
   `--index_file`: file to store the indexes for keyframes (csv)

3. **Task-Specific mLLM Evaluation**
   This part takes the visually augmented video and key frame indices obtained from the frame-wise information augmentation stage, segments the enhanced video into multiple key clips, and calls the multimodal large language model (mLLM) to analyze the severity of the accident in each clip.
   
   `vlm.py`: calls openai api (gpt-4o default)
   
   `--input`: input video path (mp4)
   
   `--list`: list of key frame indexes (str)
   
   `--output`: output file path of responses by mLLM (txt)
   
   Note that the length of each clip can be specified in `def process_images`
   
   `vlm.sh`: A script that can batch process videos by calling `vlm.py`.

## LLaVA-NeXT Pipeline
The only difference between the llava-next pipeline and the main logic pipeline is that the Task-Specific mLLM Evaluation uses a different mLLM model to evaluate the results. We will use LLaVA-NeXT as an example to illustrate how this pipeline works.

- Download LLaVA-NeXT

```python
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
```
Follow its README to set up the environment.

- Navigate to the following path:

SeeUnsafe/LLaVA-NeXT/scripts/video/demo

- Copy `video_batch_demo_llava.sh` in SeeUnsafe to this path.

We have provided a sample script for batch processing videos. Feel free to modify it as needed.


## To-Do List

- [x] **Main logic**: Accident analysis pipeline using GPT-4o and GPT-4o mini

- [x] **Pipeline using other models**: LLaVA-NeXT, VideoCLIP

- [ ] **IMS calculation**: Information match score 

- [ ] **Other experimental functionalities**: RAG, key frame selection, trajectory-by-grounding, and maybe even more!

- [x] **Dataset preparation**: [Toyota Woven Traffic Safety Dataset](https://woven-visionai.github.io/wts-dataset-homepage/)

Stay tuned for updates!
## Citation
If you find this work useful for your research, please cite our paper:
```
@article{zhang2025language,
  title={When language and vision meet road safety: leveraging multimodal large language models for video-based traffic accident analysis},
  author={Zhang, Ruixuan and Wang, Beichen and Zhang, Juexiao and Bian, Zilin and Feng, Chen and Ozbay, Kaan},
  journal={arXiv preprint arXiv:2501.10604},
  year={2025}
}
```

## Acknowledgments
The codebase builds upon these wonderful projects [SeeDo](https://github.com/ai4ce/SeeDo), [SAM](https://github.com/facebookresearch/segment-anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [MASA](https://github.com/siyuanliii/masa). Feel free to play around with them!
