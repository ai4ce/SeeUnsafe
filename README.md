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

Make sure these models are installed in editable packages by `pip install -e .`

- The code still uses one checkpoint from segment-anything.

Make sure you download it in the SeeDo folder.
**`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

- Obtain an OpenAI API key and create a `key.py` file under path SeeUnsafe/
```python
touch key.py
echo 'projectkey="YOUR_OPENAI_API_KEY"' > key.py
```

## Pipeline

There are two critical parts of SeeUnsafe, *frame-wise information augmentation* part and *task-specific mLLM evaluation* part.

1. **Frame-Wise Information Augmentation**
   This part integrates various computer vision models to extract the locations of cars and pedestrians, segment them, and add visual prompts in different colors to facilitate understanding by the multimodal large language model (mLLM).
   
   `track_objects.py`: takes an MP4 video file and the number of key frames as input. It adds visual prompts to cars and pedestrians in the video, uniformly samples the specified number of key frames, and saves the indices of the key frames along with the center coordinates of the detected objects in those frames.
   
   `--input`: the input video path (mp4)
   
   `--output`: output path to the video with augmented visual prompt (mp4)
   
   `--num_key_frames`: number of keyframes for normal sampling (int)
   
   `--bbx_file`: file to store the center coordinates of the detected objects in keyframes (csv)
   
   `--index_file`: file to store the indexes for keyframes (csv)

3. **Task-Specific mLLM Evaluation**
   This part takes the visually augmented video and key frame indices obtained from the frame-wise information augmentation stage, segments the enhanced video into multiple key clips, and calls the multimodal large language model (mLLM) to analyze the severity of the accident in each clip.
   
   `vlm.py`: calls openai api (gpt-4o default)
   
   `--input`: input video path (mp4)
   
   `--list`: list of key frame indexes (str)
   
   `--output`: output file path of responses by mLLM (txt)
   
   Note that the length of each clip can be specified in `def process_images`
   
   `vlm.sh`: A script that can batch process videos by calling `vlm.py`.

## Future To-Do's
We will gradually open source the following contents.

-[x] **Main logic**: Accident analysis pipeline using GPT-4o

-[ ] **mLLM ablation study**: Accident analysis pipeline using LLaVA

-[ ] **Vehicle view analysis**: Accident analysis pipeline for vehicle view videos

-[ ] **RAG**: a pipeline integrating RAG methods for better performance

-[ ] **Dataset**: dataset of accident videos for statistical purposes


