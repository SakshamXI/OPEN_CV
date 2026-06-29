"""
Predict the class of an image using the trained
Cats vs Dogs MobileNetV2 model.
"""

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =============================================================================
# Configuration
# =============================================================================

IMAGE_SIZE = (224, 224)
MODEL_PATH = "dogs_vs_cats.keras"
IMAGE_PATH = "sample_images/cat.jpg"  # Change image path here

CLASS_NAMES = ["Cat", "Dog"]

# =============================================================================
# Load Trained Model
# =============================================================================

model = tf.keras.models.load_model(MODEL_PATH)

# =============================================================================
# Load and Preprocess Image
# =============================================================================

# Read image
image = cv2.imread(IMAGE_PATH)

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
image = cv2.resize(image, IMAGE_SIZE)

# Convert to batch
image = np.expand_dims(image, axis=0)

# Preprocess for MobileNetV2
image = preprocess_input(image)

# =============================================================================
# Make Prediction
# =============================================================================

prediction = model.predict(image, verbose=0)

predicted_index = np.argmax(prediction[0])
confidence = prediction[0][predicted_index]

print(f"Prediction : {CLASS_NAMES[predicted_index]}")
print(f"Confidence : {confidence * 100:.2f}%")