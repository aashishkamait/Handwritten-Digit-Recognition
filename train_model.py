import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load MNIST Dataset

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# 2. Normalize Pixel Values

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 3. Reshape Data for CNN

# CNN expects:
# (number of images, height, width, channels)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("After reshaping:")
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)


# 4. Display Some Images

plt.figure(figsize=(8, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.title("Label: " + str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()


# 5. Create CNN Model

model = Sequential([

    # First convolution layer
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # Reduce image size
    MaxPooling2D((2, 2)),

    # Second convolution layer
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Reduce image size again
    MaxPooling2D((2, 2)),

    # Convert feature maps into a vector
    Flatten(),

    # Fully connected layer
    Dense(128, activation="relu"),

    # Reduce overfitting
    Dropout(0.5),

    # Output layer: 10 digits (0-9)
    Dense(10, activation="softmax")
])


# 6. Display Model Structure

model.summary()


# 7. Compile Model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 8. Train Model

print("\nStarting model training...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)


# 9. Evaluate Model

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)


# 10. Make Predictions

predictions = model.predict(x_test)

predicted_digits = np.argmax(predictions, axis=1)

print("\nFirst 10 predictions:")
print(predicted_digits[:10])

print("\nActual labels:")
print(y_test[:10])


# 11. Display Predictions

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        "Pred: " + str(predicted_digits[i]) +
        "\nActual: " + str(y_test[i])
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# 12. Save Trained Model

model.save("digit_model.keras")

print("\n================================")
print("Model saved successfully!")
print("File: digit_model.keras")
print("================================")