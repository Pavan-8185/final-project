from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import StringLookup
import numpy as np
from PIL import Image
import io
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

# Load character vocabulary
with open("./characters", "rb") as fp:
    characters = pickle.load(fp)

char_to_num = StringLookup(vocabulary=characters, mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Model input parameters
image_width = 128
image_height = 32
max_len = 21

# Resize function
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    pad_height_top = pad_height // 2 + pad_height % 2
    pad_height_bottom = pad_height // 2
    pad_width_left = pad_width // 2 + pad_width % 2
    pad_width_right = pad_width // 2
    image = tf.pad(image, [[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

# Preprocess image for prediction
def preprocess_image(image_bytes):
    image = tf.image.decode_image(image_bytes, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    image = distortion_free_resize(image, (image_width, image_height))
    return tf.expand_dims(image, axis=0)

# Decode predictions to readable text
def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Define custom CTCLayer class
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

# Load the model
model = load_model("ocr_model_50_epoch.h5", custom_objects={"CTCLayer": CTCLayer})
prediction_model = tf.keras.Model(inputs=model.get_layer(name="image").input,
                                  outputs=model.get_layer(name="dense2").output)

# API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        image_tensor = preprocess_image(image_bytes)
        prediction = prediction_model.predict(image_tensor)
        decoded = decode_predictions(prediction)
        return jsonify({"prediction": decoded[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
