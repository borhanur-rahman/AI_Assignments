"""
Assignment 14
Effect of:
1. Different activation functions in hidden layers
2. Different loss functions
on CNN classifier performance
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ======================================================
# 1. LOAD DATASET (CIFAR-10)
# ======================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten labels for sparse loss
y_train_sparse = y_train.flatten()
y_test_sparse  = y_test.flatten()

# One-hot encode for categorical loss
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

# ======================================================
# 2. BUILD CNN MODEL FUNCTION
# ======================================================

def build_model(activation="relu", num_classes=10):

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation=activation, input_shape=(32,32,3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation=activation),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation=activation),

        layers.Flatten(),
        layers.Dense(128, activation=activation),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model

# ======================================================
# 3. TRAINING FUNCTION
# ======================================================

def train_and_evaluate(activation, loss_type):

    print("\n========================================")
    print(f"Activation: {activation}")
    print(f"Loss: {loss_type}")
    print("========================================")

    model = build_model(activation=activation)

    if loss_type == "sparse":
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(
            x_train, y_train_sparse,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
        test_loss, test_acc = model.evaluate(x_test, y_test_sparse, verbose=0)

    else:  # categorical
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(
            x_train, y_train_cat,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
        test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

    print("Test Accuracy:", round(test_acc,4))

    return test_acc, history

# ======================================================
# 4. RUN EXPERIMENTS
# ======================================================

activations = ["relu", "tanh", "sigmoid"]
losses = ["sparse", "categorical"]

results = {}

for act in activations:
    for loss in losses:
        acc, hist = train_and_evaluate(act, loss)
        results[(act, loss)] = acc

# ======================================================
# 5. PRINT FINAL COMPARISON
# ======================================================

print("\n========== FINAL RESULTS ==========")

for key, value in results.items():
    print("Activation:", key[0],
          "| Loss:", key[1],
          "| Test Accuracy:", round(value,4))
