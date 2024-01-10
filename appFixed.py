from flask import Flask, request, jsonify
import joblib
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('roomclassifier.joblib')


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (150, 150))
    return img.flatten()


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'})

    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    processed_image = preprocess_image(temp_path)
    if processed_image is None:
        return jsonify({'error': 'Unable to process the image'})

    prediction = model.predict([processed_image])[0]

    result = {'prediction': 'Modern' if prediction == 1 else 'Traditional'}
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
