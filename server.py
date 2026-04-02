from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path='plant_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['guava', 'oregano', 'unknown']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(io.BytesIO(request.data)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        class_idx = int(np.argmax(output))
        confidence = float(np.max(output)) * 100

        return jsonify({
            'plant': CLASS_NAMES[class_idx],
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)