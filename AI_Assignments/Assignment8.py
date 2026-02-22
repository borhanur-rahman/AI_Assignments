import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)   # VGG usually uses 224x224
NUM_CLASSES = 10        # Change according to your dataset

# ---------------- BUILD VGG16-LIKE MODEL ----------------
def build_vgg16_like():

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=IMG_SIZE+(3,)))
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Block 3
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Block 4
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Block 5
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Fully Connected Layers (VGG style)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    return model


# ---------------- CREATE MODEL ----------------
model = build_vgg16_like()

# ---------------- COMPILE ----------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- SUMMARY ----------------
model.summary()
