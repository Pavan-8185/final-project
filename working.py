import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import StringLookup
import pickle

# Load vocabulary and mappings
with open("./characters", "rb") as fp:
    characters = pickle.load(fp)

char_to_num = StringLookup(vocabulary=characters, mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Parameters
image_width = 128
image_height = 32
max_len = 21


# Resize and preprocess the image
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    pad_height_top = pad_height // 2 + pad_height % 2
    pad_height_bottom = pad_height // 2
    pad_width_left = pad_width // 2 + pad_width % 2
    pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


# Preprocess uploaded image bytes
def preprocess_image(image_bytes):
    if len(image_bytes) == 0:
        raise ValueError("Uploaded image is empty.")
    image = tf.image.decode_image(image_bytes, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    image = distortion_free_resize(image, (image_width, image_height))
    return image


# Decode the prediction output
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)

    return output_text


# Define CTCLayer (needed for model loading if trained with it)
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


# Load the trained model and extract the prediction model
@st.cache_resource
def load_prediction_model():
    model = tf.keras.models.load_model("./ocr_model_50_epoch.h5", custom_objects={"CTCLayer": CTCLayer})
    image_input = model.get_layer(name="image").input
    dense_output = model.get_layer(name="dense2").output
    prediction_model = tf.keras.models.Model(inputs=image_input, outputs=dense_output)
    return prediction_model


# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Handwritten Text Recognition", layout="centered")
st.title("📝 Handwritten Text Recognition")
st.markdown("Upload a **cropped word image** to recognize its handwritten text.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        img_bytes = uploaded_file.getvalue()
        img_tensor = preprocess_image(img_bytes)
        img_tensor = tf.expand_dims(img_tensor, axis=0)  # shape: (1, 128, 32, 1)

        # Predict
        model = load_prediction_model()
        prediction = model.predict(img_tensor)
        decoded = decode_batch_predictions(prediction)

        st.success("✅ Prediction:")
        st.code(decoded[0])

    except Exception as e:
        st.error(f"Error: {e}")
