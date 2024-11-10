import base64

import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load the model
model_path = "C:\\Users\\tichnut\\Desktop\\my_project\\final_project\\models\\keypoint_detection_model_new_and_good5.h5"
model = load_model(model_path)


# Function to preprocess the image
def preprocess_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image


# Function to post-process predictions
def postprocess_keypoints(keypoints, original_size, target_size=(256, 256)):
    keypoints = np.array(keypoints).reshape(-1, 2)
    keypoints[:, 0] *= target_size[0] / original_size[0]
    keypoints[:, 1] *= target_size[1] / original_size[1]
    return keypoints


# Main function to handle file and process it
def handle_file(image_path):
    # Load and preprocess the image
    preprocessed_image = preprocess_image(image_path)
    original_size = preprocessed_image.shape[1:3]

    # Predict keypoints
    predicted_keypoints = model.predict(preprocessed_image)
    predicted_keypoints = postprocess_keypoints(predicted_keypoints, original_size)

    # # Visualize the results
    # origin_img = Image.open(image_path)
    # img = origin_img.resize((256, 256))
    #
    # plt.imshow(img)
    # for (x, y) in predicted_keypoints:
    #     plt.plot(x+80, y+80, 'ro')
    # plt.axis('off')  # Turn off the axis
    # plt.show()

    target_size = (256, 256)
    origin_img = Image.open(image_path)
    img = origin_img.resize(target_size)
    image = np.array(img)  # Load image
    plt.imshow(image)
    arr = []
    for (x, y) in predicted_keypoints:
        plt.plot(x, y, 'ro')
        arr.append([x, y])

    line_arr = [[arr[3],arr[1]],[arr[1],arr[0]],[arr[0],arr[2]],[arr[2],arr[4]],[arr[5],arr[7]], [arr[7],arr[9]], [arr[6],arr[8]], [arr[8],arr[10]], [arr[11],arr[15]], [arr[12],arr[14]], [arr[13],arr[15]], [arr[14],arr[16]]]

    for start_point, end_point in line_arr:
        # חישוב ההפרשים בציר ה-X ובציר ה-Y
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        plt.quiver(start_point[0], start_point[1], dx, dy, angles='xy', scale_units='xy', scale=1, color='r')


    #plt.show()
    plt.axis('off')  # Turn off the axis
    plt.show()

    # buffer = BytesIO()
    # plt.savefig(buffer, format='PNG')
    # buffer.seek(0)
    # img_str = base64.b64encode(buffer.getvalue()).decode()

    # return img_str, arr
    # return image, predicted_keypoints
image_path = rf"C:\Users\tichnut\Desktop\my_project\final_project\pictures\y.png"
handle_file(image_path)