from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load trained model
model = load_model("digit_model.h5")

def predict_digit(img):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert colors: MNIST digits are white on black
    img = cv2.bitwise_not(img)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Normalize pixel values
    img = img.astype("float32") / 255.0

    # Reshape for CNN input
    img = img.reshape(1, 28, 28, 1)

    # # Predict digit
    # prediction = model.predict(img)
    # return prediction.argmax(axis=1)[0]

     # Predict probabilities
    probabilities = model.predict(img)[0]

    # Get predicted digit and confidence
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit] * 100  # percentage

    return predicted_digit, confidence, probabilities
