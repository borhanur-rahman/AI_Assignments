"""
11. Discuss the feature extraction power of your favorite CNN pretrained by the ImageNet 
dataset before and after transfer learning by the MNIST digit dataset after plotting high 
dimensional feature vectors on 2D plane using the following two dimension reduction 
techniques: 
● Principal Component Analysis (PCA) 
● t-distributed Stochastic Neighbor Embedding (t-SNE)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =====================================================
# 1️⃣ Load MNIST Dataset
# =====================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert to 3 channels (because MobileNet expects RGB)
x_train = np.stack([x_train]*3, axis=-1)
x_test = np.stack([x_test]*3, axis=-1)

# Resize to 96x96 (MobileNetV2 minimum size)
x_train = tf.image.resize(x_train, (96,96)).numpy()
x_test = tf.image.resize(x_test, (96,96)).numpy()

# Use smaller subset for faster PCA & t-SNE
x_subset = x_test[:2000]
y_subset = y_test[:2000]

# =====================================================
# 2️⃣ Load Pretrained MobileNetV2 (ImageNet)
# =====================================================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

base_model.trainable = False

# Global pooling for feature extraction
feature_extractor = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D()
])

# =====================================================
# 3️⃣ Extract Features BEFORE Transfer Learning
# =====================================================

features_before = feature_extractor.predict(x_subset)

# =====================================================
# 4️⃣ Transfer Learning on MNIST
# =====================================================

# Add classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train only classifier head
model.fit(x_train[:10000], y_train[:10000],
          epochs=3,
          batch_size=32,
          validation_split=0.1)

# =====================================================
# 5️⃣ Extract Features AFTER Transfer Learning
# =====================================================

features_after = feature_extractor.predict(x_subset)

# =====================================================
# 6️⃣ PCA Visualization
# =====================================================

pca = PCA(n_components=2)

pca_before = pca.fit_transform(features_before)
pca_after = pca.fit_transform(features_after)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(pca_before[:,0], pca_before[:,1], c=y_subset, cmap='tab10', s=5)
plt.title("PCA Before Transfer Learning")

plt.subplot(1,2,2)
plt.scatter(pca_after[:,0], pca_after[:,1], c=y_subset, cmap='tab10', s=5)
plt.title("PCA After Transfer Learning")

plt.show()

# =====================================================
# 7️⃣ t-SNE Visualization
# =====================================================

tsne = TSNE(n_components=2, random_state=42)

tsne_before = tsne.fit_transform(features_before)
tsne_after = tsne.fit_transform(features_after)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(tsne_before[:,0], tsne_before[:,1], c=y_subset, cmap='tab10', s=5)
plt.title("t-SNE Before Transfer Learning")

plt.subplot(1,2,2)
plt.scatter(tsne_after[:,0], tsne_after[:,1], c=y_subset, cmap='tab10', s=5)
plt.title("t-SNE After Transfer Learning")

plt.show()
