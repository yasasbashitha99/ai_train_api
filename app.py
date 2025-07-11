from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load trained TFLite model
interpreter = tf.lite.Interpreter(model_path="ai_train_model.tflite")
interpreter.allocate_tensors()

# Model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return "🚦 Emergency Brake AI API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from client
        data = request.get_json()
        distance = float(data['distance_cm'])
        speed = float(data['speed_rpm'])

        # Prepare model input
        input_data = np.array([[distance, speed]], dtype=np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Process output
        decision = int(round(output[0][0]))

        return jsonify({
            "brake": decision,
            "distance_cm": distance,
            "speed_rpm": speed
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

