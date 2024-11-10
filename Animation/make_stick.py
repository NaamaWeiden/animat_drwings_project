# Import necessary packages
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import ImageSequenceClip

# Define file paths (replace with your actual file paths)
video_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\Animation\drawings\WMb16Mya.mp4'  # Path to your video file
drawing_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\Animation\drawings\b1.png'  # Path to your drawing file

def get_pose_keypoints(video_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    keypoints_all_frames = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            keypoints = []

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    keypoints.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))
            else:
                keypoints = [(None, None)] * 33

            keypoints_all_frames.append(keypoints)

    cap.release()
    return keypoints_all_frames

def draw_stick_figure(ax, keypoints, drawing):
    keypoints = [(x, y + 100) for x, y in keypoints]

    head_center = keypoints[0]
    shoulder_left = keypoints[11]
    shoulder_right = keypoints[12]
    elbow_left = keypoints[13]
    elbow_right = keypoints[14]
    wrist_left = keypoints[15]
    wrist_right = keypoints[16]
    hip_left = keypoints[23]
    hip_right = keypoints[24]
    knee_left = keypoints[25]
    knee_right = keypoints[26]
    ankle_left = keypoints[27]
    ankle_right = keypoints[28]

    ax.imshow(drawing)

    if head_center:
        ax.plot(head_center[0], head_center[1], 'o', markersize=15, color='black')
    if shoulder_left and shoulder_right:
        ax.plot([shoulder_left[0], shoulder_right[0]], [shoulder_left[1], shoulder_right[1]], color='black', linewidth=5)
    if shoulder_left and elbow_left:
        ax.plot([shoulder_left[0], elbow_left[0]], [shoulder_left[1], elbow_left[1]], color='black', linewidth=5)
    if elbow_left and wrist_left:
        ax.plot([elbow_left[0], wrist_left[0]], [elbow_left[1], wrist_left[1]], color='black', linewidth=5)
    if shoulder_right and elbow_right:
        ax.plot([shoulder_right[0], elbow_right[0]], [shoulder_right[1], elbow_right[1]], color='black', linewidth=5)
    if elbow_right and wrist_right:
        ax.plot([elbow_right[0], wrist_right[0]], [elbow_right[1], wrist_right[1]], color='black', linewidth=5)
    if hip_left and hip_right:
        ax.plot([hip_left[0], hip_right[0]], [hip_left[1], hip_right[1]], color='black', linewidth=5)
    if hip_left and knee_left:
        ax.plot([hip_left[0], knee_left[0]], [hip_left[1], knee_left[1]], color='black', linewidth=5)
    if knee_left and ankle_left:
        ax.plot([knee_left[0], ankle_left[0]], [knee_left[1], ankle_left[1]], color='black', linewidth=5)
    if hip_right and knee_right:
        ax.plot([hip_right[0], knee_right[0]], [hip_right[1], knee_right[1]], color='black', linewidth=5)
    if knee_right and ankle_right:
        ax.plot([knee_right[0], ankle_right[0]], [knee_right[1], ankle_right[1]], color='black', linewidth=5)

def create_animation_video(keypoints_all_frames, drawing_path, output_path='output_video.mp4'):
    drawing = plt.imread(drawing_path)
    frames = []

    for keypoints in keypoints_all_frames:
        fig, ax = plt.subplots()
        draw_stick_figure(ax, keypoints, drawing)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    clip = ImageSequenceClip(frames, fps=10)  # Change fps as needed
    clip.write_videofile(output_path)

# Usage example
keypoints_all_frames = get_pose_keypoints(video_path)
create_animation_video(keypoints_all_frames, drawing_path)

# If you want to open the video file automatically after creation, use this (optional):
import os
os.startfile('output_video.mp4')
