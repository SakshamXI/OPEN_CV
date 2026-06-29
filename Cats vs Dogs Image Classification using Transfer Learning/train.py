"""
Cats vs Dogs Image Classification using Transfer Learning (MobileNetV2)

This script downloads the Cats vs Dogs dataset using KaggleHub,
creates training, validation, and test datasets, applies the
required MobileNetV2 preprocessing, trains a transfer learning
model, and saves the trained model.
"""

import tensorflow as tf
import kagglehub

from tensorflow.keras import Model, layers
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)

# =============================================================================
# Configuration
# =============================================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 10

# =============================================================================
# Dataset
# =============================================================================

# Download dataset
dataset_path = (
    kagglehub.dataset_download("salader/dogsvscats")
    + "/catsvsdogs"
)

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/test",
    validation_split=0.5,
    subset="training",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/test",
    validation_split=0.5,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

# =============================================================================
# Data Preprocessing
# =============================================================================

train_dataset = train_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

test_dataset = test_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

# =============================================================================
# Model Construction
# =============================================================================

# Load pretrained MobileNetV2
base_model = MobileNetV2(weights="imagenet")

# Add custom classification layer
classifier = layers.Dense(
    units=2,
    activation="softmax"
)(base_model.layers[-2].output)

model = Model(
    inputs=base_model.input,
    outputs=classifier
)

# Freeze pretrained layers
for layer in model.layers[:-1]:
    layer.trainable = False

# =============================================================================
# Model Compilation
# =============================================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# =============================================================================
# Model Training
# =============================================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
)

# =============================================================================
# Save Model
# =============================================================================

model.save("dogs_vs_cats.keras")