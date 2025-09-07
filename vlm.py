import copy
import numpy as np
import cv2
import shapely
from shapely.geometry import *
from shapely.affinity import *
import matplotlib.pyplot as plt
from openai import OpenAI
from VLM_CaP.src.key import mykey, projectkey
import sys
from IPython.display import display, Image
import base64
from io import BytesIO
import os
import re
from PIL import Image
from collections import Counter
import argparse
import csv
import ffmpy
import ast
from vlm_video import extract_frame_list  # import extract_frames
# set up your openai api key
client = OpenAI(api_key=mykey)
# def for calling openai api with different prompts
def call_openai_api(prompt_messages):
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 400,
        "temperature": 0
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def extract_keywords_pick(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting pick keyword from response:", response)
        return None
def extract_keywords_drop(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting drop keyword from response:", response)
        return None
def extract_keywords_reference(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting reference object from response:", response)
        return None
def is_frame_relevant(response):
    return "hand is manipulating an object" in response.lower()
def parse_closest_object_and_relationship(response):
    pattern = r"Closest Object: ([^,]+), (.+)"
    match = re.search(pattern, response)
    if match:
        return match.group(1), match.group(2)
    print("Error parsing reference object and relationship from response:", response)
    return None, None

def save_results_to_txt(string_cache, output_file):
    """
    Save results to a text file.
    :param string_cache: String containing the processed results.
    :param output_file: Path to the output text file.
    """
    with open(output_file, mode='a') as file:
        file.write(string_cache + "\n")
    print(f"Results appended to {output_file}")

def process_images(selected_frames, time_stamps):
    string_cache = ""  # cache for CaP operations
    i = 0
    while i < len(selected_frames):
        # Notice that the number of frames for each batch is hard coded here. Feel free to modify.
        input_frame_pick = selected_frames[i:i+8]
        input_time_stamps = time_stamps[i:i+8]
        
        base_time_stamp = input_time_stamps[0]
        zeroed_time_stamps = [round(ts - base_time_stamp, 1) for ts in input_time_stamps]

        prompt_messages_relevance_pick = [
            {
                "role": "system",
                "content": [
                    "You are a traffic accident inspector. You need to determine which class (normal, near-miss, or collision among road users) the provided videos are. Notice that some but not all of the pedestrians are marked with green contour and some but not all of the cars are marked with red contour. Use that to help with your observation.",
                    "The normal class refers to no sudden deviations in speed, direction, or proximity between cars or between cars and pedestrians, or no objects are detected in the videos.",
                    "The near-miss class depicts situations where vehicles or pedestrians come extremely close to colliding but ultimately avoid impact. These events often involve abrupt changes in speed, direction, or proximity, indicating a high risk of an accident that was narrowly avoided.",
                    "The collision class depicts events where vehicles, pedestrians, or objects come into direct contact, resulting in an impact. These incidents often involve significant changes in speed or direction due to the collision, and can lead to visible damage or injury.",
                    "We use 0 for near-miss, 1 for collision, and 2 for normal to represent the video class.",
                    "You will receive a sequence of images as well as the corresponding time step as input. They are arranged in chronological order, representing events that are observed sequentially. The time step is 0.1 seconds. Based on this, determine the video class and answer questions.",
                    ],
            },
            {
                "role": "user",
                "content": [
                    "Three images are arranged in chronological order, depicting a traffic event. Please choose one from normal, near-miss, and collision that can best describe the given video.",
                    "If so, what types of road users are involved, and in what context do these traffic anomalies occur?",
                    "Respond in the format of the following example without any additional information:",
                    "Video Class: type of integer indicating the video class. For example: 0 for near-miss, 1 for collision, and 2 for normal.",
                    "Object Detail: type of string describing the appearance of road users. For example: The involved road users are a middle-aged male pedestrian and a white sedan",
                    "Scene Context: type of string describing the scene environment. For example: Sunny day with dry road surface in a curbside area near an intersection, under daytime lighting with moderate mixed traffic.",
                    "Justification: type of string explaining the reason why a traffic anomaly occurs or why a traffic anomaly doesn't exist. For example: The pedestrian's pose changes from walking to a sudden stop, indicating an unexpected reaction to the vehicle. The vehicle shows a significant deviation from its original trajectory, indicating that it misjudged the pedestrian's path. This heavy deviation led to the vehicle failing to avoid the pedestrian, resulting in a collision.",
                    "The images are as follows:",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_pick),
                    ],
            },
        ]
        response = call_openai_api(prompt_messages_relevance_pick)
        # print(prompt_messages_relevance_pick)
        print(response)
        string_cache += response + "\n"
        print(i)
        i += 8

    return string_cache

def save_results_to_csv(input_file, string_cache, output_file):
    file_exists = os.path.exists(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["demo", "response"])

        writer.writerow([f"{input_file}", string_cache])
    print(f"Results appended to {output_file}")

def main(input_video_path, frame_index_list, output_file):
    frame_index_list = ast.literal_eval(frame_index_list)
    # Calculate the timestamp for each key frame
    base_frame_index = frame_index_list[0]
    time_stamps = [(index - base_frame_index) * 0.1 for index in frame_index_list]
    print(time_stamps)
    
    # video path
    video_path = input_video_path
    selected_raw_frames1 = []
    selected_frame_index = frame_index_list
    cap = cv2.VideoCapture(video_path)
    actual_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        actual_frame_count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Actual frame count: {actual_frame_count}")
    for index in selected_frame_index:
        if index < actual_frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, cv2_image = cap.read()
            if ret:
                selected_raw_frames1.append(cv2_image)
            else:
                print(f"Failed to retrieve frame at index {index}")
        else:
            print(f"Frame index {index} is out of range for this video.")
    cap.release()
    selected_frames1 = extract_frame_list(selected_raw_frames1)
    string_cache = process_images(selected_frames1, time_stamps)
    print(string_cache)
    save_results_to_txt(string_cache, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and key frame extraction.")
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--list', type=str, required=True, help='List of key frame indexes')
    parser.add_argument('--output', type=str, required=True, help='Output file path (txt format)')
    args = parser.parse_args()
    # Call the main function with arguments
    main(args.input, args.list, args.output)
