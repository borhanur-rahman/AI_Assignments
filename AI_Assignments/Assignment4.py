import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10


# ==========================================
# 1️⃣ Build FCFNN Model
# ==========================================

def build_fcf_nn(input_shape, num_classes):

    inputs = Input(shape=input_shape, name='input_layer')

    x = Flatten()(inputs)

    x = Dense(512, activation='relu', name='hidden1')(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dense(64, activation='relu', name='hidden4')(x)
    x = Dense(32, activation='relu', name='hidden5')(x)

    outputs = Dense(num_classes, activation='softmax',
                    name='output_layer')(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name="FCFNN_Classifier")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # 🔥 changed here
        metrics=['accuracy']
    )

    return model


# ==========================================
# 2️⃣ Training Function
# ==========================================

def train_and_evaluate(dataset_name, x_train, y_train,
                       x_test, y_test, input_shape):

    print(f"\nTraining on {dataset_name}")

    num_classes = 10

    # Normalize images
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = build_fcf_nn(input_shape, num_classes)

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"{dataset_name} Test Accuracy:", test_acc)


# ==========================================
# 3️⃣ MNIST
# ==========================================

(x_train_m, y_train_m), (x_test_m, y_test_m) = mnist.load_data()

train_and_evaluate(
    "MNIST",
    x_train_m, y_train_m,
    x_test_m, y_test_m,
    input_shape=(28, 28)
)


# ==========================================
# 4️⃣ Fashion MNIST
# ==========================================

(x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()

train_and_evaluate(
    "Fashion MNIST",
    x_train_f, y_train_f,
    x_test_f, y_test_f,
    input_shape=(28, 28)
)


# ==========================================
# 5️⃣ CIFAR-10
# ==========================================

(x_train_c, y_train_c), (x_test_c, y_test_c) = cifar10.load_data()

# CIFAR labels shape (n,1) → flatten
y_train_c = y_train_c.flatten()
y_test_c = y_test_c.flatten()

train_and_evaluate(
    "CIFAR-10",
    x_train_c, y_train_c,
    x_test_c, y_test_c,
    input_shape=(32, 32, 3)
)
