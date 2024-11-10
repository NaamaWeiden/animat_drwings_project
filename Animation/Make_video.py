import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from moviepy.editor import ImageSequenceClip
from PIL import Image
# from google.colab import files
from Animation.media_cut import drawing_paths


def add_image(ax, img_path, coordinates, size=(10,10)):
    if coordinates:
        img = Image.open(img_path)
        img = img.resize(size, Image.LANCZOS)  # Resize image
        img = np.array(img)
        imagebox = OffsetImage(img, zoom=7)
        ab = AnnotationBbox(imagebox, coordinates, frameon=False, xybox=(0, 0), xycoords='data', boxcoords="offset points", pad=0)
        ax.add_artist(ab)



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

    stomach_center = ((shoulder_left[0]+shoulder_right[0])//2 , (shoulder_left[1]+hip_left[1])//2)
    #stomach_center = (((shoulder_left[0] + hip_left[0]) // 2) + ((shoulder_right[0] + hip_right[0]) //2) //2, ((shoulder_left[1] + hip_left[1]) // 2) + ((shoulder_right[1] + hip_right[1]) //2) //2) if hip_left and hip_right else (None, None)

    ax.imshow(drawing)
    if head_center:
        add_image(ax, knight_images.get('head'), head_center)
    if shoulder_left and wrist_left:
        arm_left_center = ((shoulder_left[0]), (shoulder_left[1]))
        add_image(ax, knight_images.get('left_arm'), arm_left_center)
    if shoulder_right:
        arm_right_center = ((shoulder_right[0]), (shoulder_right[1]))
        add_image(ax, knight_images.get('right_arm'), arm_right_center)
    if hip_left and ankle_left:
        leg_left_center = ((hip_left[0]), (hip_left[1]))
        add_image(ax, knight_images.get('left_leg'), leg_left_center)
    if hip_right and ankle_right:
        leg_right_center = ((hip_right[0]), (hip_right[1] ) )
        add_image(ax, knight_images.get('right_leg'), leg_right_center)
    if stomach_center:
        add_image(ax, knight_images.get('stomach'), stomach_center)  # Adjust size for the stomach

def create_animation_video(keypoints_all_frames, drawing_path, drawing_paths, output_path='output_video.mp4'):
    frames = []
    drawing = plt.imread(drawing_path)
    for keypoints in keypoints_all_frames:
        fig, ax = plt.subplots(figsize=(25, 35))
        draw_stick_figure(ax, keypoints, drawing, drawing_paths)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    clip = ImageSequenceClip(frames, fps=7)  # Change the frames per second as needed
    clip.write_videofile(output_path)



drawing_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\Animation\drawings\b1.png' # תמונה לבנה
video_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\Animation\drawings\unnamed+(1) (1).mp4'  # Provide the path to your video file
# drawing_paths = save_cropped_images(image_path, output_dir)

keypoints_all_frames = get_pose_keypoints(video_path)
create_animation_video(keypoints_all_frames,drawing_path, drawing_paths)

# In PyCharm, instead of downloading, we show the video directly
video_output_path = 'output_video.mp4'
create_animation_video(keypoints_all_frames, drawing_path, drawing_paths, output_path=video_output_path)

# Show the created video
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
