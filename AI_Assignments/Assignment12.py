"""
Assignment 12
Effect of Data Augmentation on CNN Classifier
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time

# ============================================================
# 1. LOAD CIFAR-10 DATASET
# ============================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Validation split
x_val = x_train[-5000:]
y_val = y_train[-5000:]

x_train = x_train[:-5000]
y_train = y_train[:-5000]

# ============================================================
# 2. DATA AUGMENTATION FUNCTION
# ============================================================

def get_augmentation(mode):

    if mode == "none":
        return None

    elif mode == "flip":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal")
        ])

    elif mode == "flip_rotation":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1)
        ])

    elif mode == "full":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2)
        ])

# ============================================================
# 3. BUILD MODEL FUNCTION
# ============================================================

def build_model(augmentation_layer=None):

    inputs = layers.Input(shape=(32,32,3))

    x = inputs

    if augmentation_layer:
        x = augmentation_layer(x)

    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================
# 4. TRAIN FUNCTION
# ============================================================

def train_model(mode):

    print("\nTraining with:", mode)

    augmentation = get_augmentation(mode)
    model = build_model(augmentation)

    print("Total Parameters:", model.count_params())

    start = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_val, y_val),
        verbose=1
    )

    end = time.time()

    val_acc = history.history['val_accuracy'][-1]

    print("Final Validation Accuracy:", round(val_acc,4))
    print("Training Time:", round(end-start,2), "seconds")

    return val_acc, history

# ============================================================
# 5. RUN EXPERIMENTS
# ============================================================

modes = ["none", "flip", "flip_rotation", "full"]
results = {}

for mode in modes:
    acc, hist = train_model(mode)
    results[mode] = acc

# ============================================================
# 6. PLOT COMPARISON
# ============================================================

plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.title("Effect of Data Augmentation on Validation Accuracy")
plt.ylabel("Validation Accuracy")
plt.show()

print("\nFinal Comparison:")
for k,v in results.items():
    print(k, ":", round(v,4))
