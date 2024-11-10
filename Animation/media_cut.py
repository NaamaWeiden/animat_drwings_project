import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = segmentation.process(image_rgb)
    mask = results.segmentation_mask
    condition = mask > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (255, 255, 255)
    output_image = np.where(condition[..., None], image, bg_image)
    return output_image

def get_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None, None
    keypoints = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in results.pose_landmarks.landmark]
    return keypoints, image

def crop_image(image, keypoints, point1_idx, point2_idx, padding=20):
    point1 = keypoints[point1_idx]
    point2 = keypoints[point2_idx]
    x_min = max(0, min(point1[0], point2[0]) - padding)
    x_max = min(image.shape[1], max(point1[0], point2[0]) + padding)
    y_min = max(0, min(point1[1], point2[1]) - padding)
    y_max = min(image.shape[0], max(point1[1], point2[1]) + padding)
    return image[y_min:y_max, x_min:x_max]

def crop_head(image, keypoints, point1_idx, point2_idx, padding=40):
    point1 = keypoints[point1_idx]
    point2 = keypoints[point2_idx]
    x_min = max(0, min(point1[0], point2[0]) - padding)
    x_max = min(image.shape[1], max(point1[0], point2[0]) + padding)
    y_min = max(0, min(point1[1], point2[1]) - padding)
    y_max = min(image.shape[0], max(point1[1], point2[1]) + padding)
    return image[y_min:y_max, x_min:x_max]

def crop_body_center(image, keypoints, left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx, padding=20):
    x_min = max(0, min(keypoints[left_shoulder_idx][0], keypoints[right_shoulder_idx][0]) - padding)
    x_max = min(image.shape[1], max(keypoints[left_hip_idx][0], keypoints[right_hip_idx][0]) + padding)
    y_min = max(0, min(keypoints[left_shoulder_idx][1], keypoints[right_shoulder_idx][1]) - padding)
    y_max = min(image.shape[0], max(keypoints[left_hip_idx][1], keypoints[right_hip_idx][1]) + padding)
    return image[y_min:y_max, x_min:x_max]

def save_cropped_images(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    keypoints, image = get_keypoints(image_path)
    if keypoints is None:
        print("No keypoints detected.")
        return None

    # Remove background from the original image
    image_no_bg = remove_background(image)

    # Crop individual parts based on keypoints
    head = crop_head(image, keypoints, 7, 8)
    left_arm = crop_image(image, keypoints, 11, 17)
    right_arm = crop_image(image, keypoints, 12, 18)
    left_leg = crop_image(image, keypoints, 23, 31)
    right_leg = crop_image(image, keypoints, 24, 32)
    body_center = crop_body_center(image, keypoints, 11, 12, 23, 24)  # From left shoulder to left hip, and right shoulder to right hip

    # Save the cropped images
    Image.fromarray(cv2.cvtColor(head, cv2.COLOR_BGR2RGB)).save(rf'{output_dir}\head.png')
    Image.fromarray(cv2.cvtColor(left_arm, cv2.COLOR_BGR2RGB)).save(rf'{output_dir}\left_arm.png')
    Image.fromarray(cv2.cvtColor(right_arm, cv2.COLOR_BGR2RGB)).save(rf'{output_dir}\right_arm.png')
    Image.fromarray(cv2.cvtColor(left_leg, cv2.COLOR_BGR2RGB)).save(rf'{output_dir}\left_leg.png')
    Image.fromarray(cv2.cvtColor(right_leg, cv2.COLOR_BGR2RGB)).save(rf'{output_dir}\right_leg.png')
    Image.fromarray(cv2.cvtColor(body_center, cv2.COLOR_BGR2RGB)).save(rf'{output_dir}\stomach.png')

    return {
        'head': rf'{output_dir}\head.png',
        'left_arm': rf'{output_dir}\left_arm.png',
        'right_arm': rf'{output_dir}\right_arm.png',
        'left_leg': rf'{output_dir}\left_leg.png',
        'right_leg': rf'{output_dir}\right_leg.png',
        'stomach': rf'{output_dir}\stomach.png'
    }
# Example usage
image_path = rf'C:\Users\tichnut\Desktop\my_project\final_project\connection_react\uploads\original_image.png'  # Path to your character image
output_dir = 'cut_draw'         # Directory to save the cropped images
drawing_paths = save_cropped_images(image_path, output_dir)
print(drawing_paths)  # Print the paths to the saved images
