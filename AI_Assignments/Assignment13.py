"""
Assignment 13
Effect of Dropout and Data Augmentation on Overfitting
CNN Classifier on CIFAR-10
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ============================================
# 1. Load Dataset
# ============================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Split validation set
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

# ============================================
# 2. Base CNN Model Builder
# ============================================

def build_model(use_dropout=False, use_augmentation=False):

    model = models.Sequential()

    # Data Augmentation Layer
    if use_augmentation:
        model.add(layers.RandomFlip("horizontal"))
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    if use_dropout:
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================
# 3. Training Function
# ============================================

def train_model(title, use_dropout=False, use_augmentation=False):

    print("\nTraining:", title)

    model = build_model(use_dropout, use_augmentation)

    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("Test Accuracy:", round(test_acc,4))

    return history


# ============================================
# 4. Run Experiments
# ============================================

hist_baseline = train_model("Baseline (No Dropout, No Augmentation)")
hist_dropout  = train_model("With Dropout", use_dropout=True)
hist_aug      = train_model("With Data Augmentation", use_augmentation=True)
hist_both     = train_model("Dropout + Augmentation", use_dropout=True, use_augmentation=True)


# ============================================
# 5. Plot Training vs Validation Accuracy
# ============================================

def plot_history(history, title):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

plot_history(hist_baseline, "Baseline")
plot_history(hist_dropout, "Dropout")
plot_history(hist_aug, "Data Augmentation")
plot_history(hist_both, "Dropout + Augmentation")
