import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="✍️",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)


model = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("✍️ Handwritten Digit Predictor")

st.write(
    "Draw a digit from **0 to 9** inside the box and let the neural network predict it."
)


# --------------------------------------------------
# Drawing Canvas
# --------------------------------------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,   # IMPORTANT
    key="canvas",
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Digit", use_container_width=True):

    if canvas_result.image_data is not None:

        # Get canvas image
        image = canvas_result.image_data.astype("uint8")

        # Convert RGBA -> grayscale
        image = Image.fromarray(image)
        image = image.convert("L")

        # Resize to MNIST size
        image = image.resize((28, 28))

        # Convert to NumPy array
        image_array = np.array(image)

        # Normalize just like MNIST
        image_array = image_array.astype("float32") / 255.0

        # Your model expects shape: (batch, 28, 28)
        image_array = np.expand_dims(image_array, axis=0)

        # Check if user actually drew something
        if np.max(image_array) < 0.05:
            st.warning("⚠️ Please draw a digit first.")

        else:
            # Prediction
            prediction = model.predict(image_array, verbose=0)

            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100

            st.success(f"## Predicted Digit: {predicted_digit}")

            st.write(f"### Confidence: {confidence:.2f}%")

            # Probability for all digits
            st.subheader("Prediction Probabilities")

            probabilities = {
                str(i): float(prediction[0][i])
                for i in range(10)
            }

            st.bar_chart(probabilities)
