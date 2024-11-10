import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from moviepy.editor import ImageSequenceClip
from PIL import Image
from Animation.media_cut import drawing_paths

# Function to add images to the axes
def add_image(ax, img_path, coordinates, size=None):
    if coordinates and img_path:
        img = Image.open(img_path)
        if size:
            img = img.resize(size, Image.LANCZOS)
        img = np.array(img)
        imagebox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(imagebox, coordinates, frameon=False, xybox=(0, 0), xycoords='data', boxcoords="offset points", pad=0)
        ax.add_artist(ab)

# Function to get pose keypoints from the video
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

# Function to draw the stick figure with images
def draw_stick_figure(ax, keypoints, drawing, knight_images):
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

    stomach_center = ((shoulder_left[0] + shoulder_right[0]) // 2, ((shoulder_left[1] + hip_left[1]) // 2)-20)
    shoulder_left = (shoulder_left[0]+30 , shoulder_left[1])
    shoulder_right = (shoulder_right[0]-30, shoulder_right[1])


    ax.imshow(drawing)

    # Define the sizes for the images (adjust these sizes to fit your needs)
    head_size = (70, 70)
    arm_size = (80, 70)
    leg_size = (35, 110)
    stomach_size = (80, 80)

    if head_center:
        add_image(ax, knight_images.get('head'), head_center, head_size)
    if shoulder_left:
        add_image(ax, knight_images.get('left_arm'), shoulder_left, arm_size)
    if shoulder_right:
        add_image(ax, knight_images.get('right_arm'), shoulder_right, arm_size)
    if hip_left:
        add_image(ax, knight_images.get('left_leg'), hip_left, leg_size)
    if hip_right:
        add_image(ax, knight_images.get('right_leg'), hip_right, leg_size)
    if stomach_center:
        add_image(ax, knight_images.get('stomach'), stomach_center, stomach_size)



# Function to create the animation video
def create_animation_video(keypoints_all_frames, drawing_path, drawing_paths, output_path='output_video.mp4'):
    frames = []
    drawing = plt.imread(drawing_path)
    figure_size = (10, 10)  # Adjust the figure size to be proportionate to the image sizes

    for keypoints in keypoints_all_frames:
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_xlim(0, 400)  # Adjust these limits based on your needs
        ax.set_ylim(0, 400)
        ax.invert_yaxis()
        draw_stick_figure(ax, keypoints, drawing, drawing_paths)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    clip = ImageSequenceClip(frames, fps=7)  # Change the frames per second as needed
    clip.write_videofile(output_path)

# # Example usage
# drawing_paths = {
#     'head': r'C:\path\to\head.png',       # Path to the head image
#     'left_arm': r'C:\path\to\left_arm.png',  # Path to the left arm image
#     'right_arm': r'C:\path\to\right_arm.png', # Path to the right arm image
#     'left_leg': r'C:\path\to\left_leg.png',   # Path to the left leg image
#     'right_leg': r'C:\path\to\right_leg.png', # Path to the right leg image
#     'stomach': r'C:\path\to\stomach.png'     # Path to the stomach image
# }

drawing_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\Animation\drawings\b1.png'     # Path to the drawing background image
video_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\Animation\drawings\WMb16Mya.mp4'         # Path to the video file

keypoints_all_frames = get_pose_keypoints(video_path)
create_animation_video(keypoints_all_frames, drawing_path, drawing_paths)

# Show the created video
video_output_path = 'output_video.mp4'
create_animation_video(keypoints_all_frames, drawing_path, drawing_paths, output_path=video_output_path)

cap = cv2.VideoCapture(video_output_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Generated Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
