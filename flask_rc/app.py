from flask import Flask, request, jsonify
from flask import request, jsonify
import joblib
import cv2
import numpy as np
from PIL import Image
from flask_cors import CORS, cross_origin
import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, request, send_file
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import subprocess
import os
import shutil


app = Flask(__name__)
CORS(app)
# Load the pre-trained model
model = joblib.load('roomclassifier.joblib')  # Replace 'your_model.joblib' with the actual path


model_path = r"C:\Users\Atiq ur Rehman\Desktop\fyp_models\diffusionModel"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe = pipe.to("cuda")  # moving to GPU

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt', '')

    try:
        image = generate_image_from_text(prompt)
        # Save the image to a BytesIO object
        image_io = BytesIO()
        image.save(image_io, format='PNG')
        image_io.seek(0)

        # Return the image as a binary stream
        return send_file(image_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')
    except Exception as e:
        print(f'Error generating image: {e}')
        return jsonify({'error': 'Failed to generate image'}), 500

    except Exception as e:
        print(f'Error generating image: {e}')
        return jsonify({'error': 'Failed to generate image'}), 500

def generate_image_from_text(prompt):
    # Use the diffusion model to generate an image from the prompt
    image = pipe(prompt).images[0]


    # image = cv2.imread('room_pic.png')
    # image = Image.fromarray(image)

    return image

# Function to preprocess an image before making predictions
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (150, 150))
    return img.flatten()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'})

    # Save the image temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    # Preprocess the image
    processed_image = preprocess_image(temp_path)
    if processed_image is None:
        return jsonify({'error': 'Unable to process the image'})

    # Make predictions
    prediction = model.predict([processed_image])[0]


    # Return the result
    result = {'prediction': 'Modern' if prediction == 1 else 'Traditional'}
    print(result)
    return jsonify(result)

import os

# ... (your existing code)

@app.route('/segment-image', methods=['POST'])
def segment_image():
    try:
        # Receive the generated image
        generated_image = request.files['file']
        if not generated_image:
            return jsonify({'error': 'No file provided'})

        # Save the generated image to a temporary file
        temp_generated_image_path = 'temp_generated_image.png'
        generated_image.save(temp_generated_image_path)
        custom_segmented_image_path = r"C:\Users\Atiq ur Rehman\Desktop\fyp_models\runs\segment\predict\temp_generated_image.png"
        # Apply YOLO segmentation
        yolo_command = [
            'yolo',
            'task=segment',
            'mode=predict',
            f'model=C:\\Users\\Atiq ur Rehman\\Desktop\\fyp_models\\best.pt',
            'conf=0.25',
            f'source={temp_generated_image_path}',
            'save=true'  
        ]

        subprocess.run(yolo_command, shell=True)

        segmented_folder = r'C:\Users\Atiq ur Rehman\Desktop\fyp_models\segmented'
        segmented_image_path = os.path.join(segmented_folder, 'segmented_image.png')
        prediction_folder = r'C:\Users\Atiq ur Rehman\Desktop\fyp_models\runs\segment\predict'

        # Move the segmented image to the 'segmented' folder
        shutil.move(custom_segmented_image_path, segmented_image_path)

        # Remove the 'runs' folder
        runs_folder = r'C:\Users\Atiq ur Rehman\Desktop\fyp_models\runs'
        shutil.rmtree(runs_folder, ignore_errors=True)

        # Return the segmented image as a binary stream
        return send_file(segmented_image_path, mimetype='image/png', as_attachment=True, download_name='segmented_image.png')


        # Return the segmented image as a binary stream
        
    except Exception as e:
        print(f'Error segmenting image: {e}')
        return jsonify({'error': 'Failed to segment image'}), 500



if __name__ == '__main__':
    app.run(debug=True)
