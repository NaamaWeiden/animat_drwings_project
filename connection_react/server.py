# from flask import Flask, request, jsonify
# import os
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# # Route to handle file upload
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({'error': 'No file selected for uploading'}), 400
#
#     # Save the file to the uploads folder
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
#
#     return jsonify({'message': 'File successfully uploaded'}), 200
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

# ---------------------------------------------------------------------------------

from flask import Flask, request, jsonify, send_file
import os

from flask_cors import cross_origin

from MediaPipe.mediaPipe import handle_file

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'processed_image.png')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to handle file upload
@app.route('/upload', methods=['POST'])
@cross_origin("*")
def upload_file():
    print('in server')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Save the file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, 'original_image.png')
    file.save(file_path)

    # Process the file with the handle_file function
    handle_file(file_path)

    return jsonify({'message': 'File successfully uploaded and processed'}), 200

@app.route('/processed-image', methods=['GET'])
@cross_origin("*")
def get_processed_image():
    try:
        return send_file(fr"C:\Users\tichnut\Desktop\my_project\final_project\connection_react\processed\processed_image.png", mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)





