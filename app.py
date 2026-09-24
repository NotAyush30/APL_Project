import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="✍️",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model.h5",
        compile=False
    )


model = load_model()


# --------------------------------------------------
# PREPROCESS DRAWING
# --------------------------------------------------
def preprocess_digit(canvas_image):

    # Convert canvas image to grayscale
    image = Image.fromarray(
        canvas_image.astype("uint8")
    ).convert("L")

    img = np.array(image)

    # Find pixels belonging to the digit
    coords = np.argwhere(img > 20)

    # No drawing detected
    if coords.size == 0:
        return None

    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    cropped = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    h, w = cropped.shape

    # ----------------------------------------------
    # Resize while maintaining aspect ratio
    # MNIST digits normally occupy around 20x20
    # inside a 28x28 image
    # ----------------------------------------------
    if h > w:

        new_h = 20
        new_w = max(
            1,
            int(w * (20 / h))
        )

    else:

        new_w = 20
        new_h = max(
            1,
            int(h * (20 / w))
        )

    # Resize cropped digit
    cropped_img = Image.fromarray(cropped)

    cropped_img = cropped_img.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    resized = np.array(cropped_img)

    # ----------------------------------------------
    # Create MNIST-style 28x28 black image
    # ----------------------------------------------
    final_img = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Center digit
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    final_img[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # ----------------------------------------------
    # Normalize 0-255 -> 0-1
    # ----------------------------------------------
    final_img = final_img.astype("float32") / 255.0

    return final_img


# --------------------------------------------------
# APP TITLE
# --------------------------------------------------
st.title("✍️ Handwritten Digit Predictor")

st.write(
    """
    Draw any digit from **0 to 9** in the box below.

    The drawing will be converted into MNIST format and
    predicted using the trained neural network.
    """
)


# --------------------------------------------------
# DRAWING CANVAS
# --------------------------------------------------
canvas_result = st_canvas(

    fill_color="black",

    stroke_width=20,

    stroke_color="white",

    background_color="black",

    width=280,

    height=280,

    drawing_mode="freedraw",

    return_image_data=True,

    key="canvas"
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
if st.button(
    "🔍 Predict Digit",
    use_container_width=True
):

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        processed_image = preprocess_digit(
            canvas_result.image_data
        )

        if processed_image is None:

            st.warning(
                "⚠️ Please draw a digit before prediction."
            )

        else:

            # --------------------------------------
            # Show model input
            # --------------------------------------
            st.subheader("Image sent to the model")

            st.image(
                processed_image,
                width=150,
                clamp=True
            )


            # --------------------------------------
            # Prepare model input
            # --------------------------------------

            # Your current model expects:
            # (batch, 28, 28)

            model_input = np.expand_dims(
                processed_image,
                axis=0
            )


            # --------------------------------------
            # Predict
            # --------------------------------------
            prediction = model.predict(
                model_input,
                verbose=0
            )

            predicted_digit = int(
                np.argmax(prediction[0])
            )

            confidence = float(
                np.max(prediction[0]) * 100
            )


            # --------------------------------------
            # RESULT
            # --------------------------------------
            st.success(
                f"### Predicted Digit: {predicted_digit}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # --------------------------------------
            # PROBABILITIES
            # --------------------------------------
            st.subheader(
                "Prediction probabilities"
            )

            probabilities = {
                str(i): float(prediction[0][i])
                for i in range(10)
            }

            st.bar_chart(
                probabilities
            )


            # --------------------------------------
            # Top 3 Predictions
            # --------------------------------------
            st.subheader(
                "Top Predictions"
            )

            top_3 = np.argsort(
                prediction[0]
            )[-3:][::-1]

            for rank, digit in enumerate(
                top_3,
                start=1
            ):

                probability = (
                    prediction[0][digit] * 100
                )

                st.write(
                    f"**{rank}. Digit {digit}** "
                    f"— {probability:.2f}%"
                )


# --------------------------------------------------
# INFO
# --------------------------------------------------
with st.expander(
    "ℹ️ How prediction works"
):

    st.write(
        """
        Your drawing goes through these steps:

        1. Convert drawing to grayscale
        2. Detect the handwritten digit
        3. Crop unnecessary empty space
        4. Resize the digit while keeping its shape
        5. Center it inside a 28 × 28 image
        6. Normalize pixel values between 0 and 1
        7. Send it to the MNIST neural network
        8. Select the digit with the highest probability
        """
    )
