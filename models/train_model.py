import os
import json
import numpy as np
from PIL import Image
from skimage.transform import resize
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Path to annotation file
annotation_file = r"amateur_drawings_annotations.json"
# Load annotations from JSON file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)
# Path to database directory
database_dir = r"amateur_drawings~"
target_size = (256, 256)
num_of_images = 20000
# Extract image paths and annotations
images = []
keypoints = []



for index,annotation in enumerate(annotations['annotations']):
    image_id = annotation['image_id']
    image_info = next((image for image in annotations['images'] if image['id'] == image_id), None)
    if image_info is None:
        # Skip if image info not found
        continue
    image_path = os.path.join(database_dir, image_info['file_name'])
    if not os.path.exists(image_path):
        # Skip if image file does not exist
        continue
        # Define target size for resizing

    origin_img = Image.open(image_path)
    x,y = origin_img.size
#     x = target_size[0]/x
#     y = target_size[0]/y
    img = origin_img.resize(target_size)
    image = np.array(img)  # Load image
    images.append(image)
    filtered_keypoints = [kp for i, kp in enumerate(annotation['keypoints']) if (i % 3 != 2)]#להחפיל את XY בכל חלק מהמערך
    keypoints.append(filtered_keypoints)
    if index == num_of_images:
        break

X = np.array(images)
y = np.array(keypoints)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

    # Define model architecture
    model = Sequential([
        Input(shape=(256, 256, 3)),  # Specify input shape using Input layer
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(34)  # Number of keypoints
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    checkpoint_path = "keypoint_detection_model_new_and_good2.h5"
    # Define the checkpoint callback to save the model
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

    # Train or continue training the model
    history = model.fit(
        np.array(images),
        np.array(keypoints),
        epochs=200,  # You can adjust the number of epochs as needed
        batch_size=32,
        validation_split=0.2,
    )
    model.save("keypoint_detection_model_new_and_good3.h5")
