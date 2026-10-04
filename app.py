import streamlit as st
import numpy as np
import cv2
from model import predict_digit
from streamlit_drawable_canvas import st_canvas

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below")

# Initialize session state
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas"

# Create canvas with dynamic key
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key,
    return_image_data=True,
)

# Buttons
col1, col2 = st.columns([1,1], gap="small")

with col1:
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
            img = cv2.resize(img, (28, 28))
            predicted_digit, confidence, probabilities = predict_digit(img)
            st.success(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f}%)")
            st.bar_chart(probabilities)
        else:
            st.warning("Please draw a digit before predicting.")

with col2:
    if st.button("Clear"):
        # Change the key to force a fresh canvas
        st.session_state.canvas_key = f"canvas_{np.random.randint(0,100000)}"
        st.rerun()

st.markdown("**Developed by: Aashish**")

