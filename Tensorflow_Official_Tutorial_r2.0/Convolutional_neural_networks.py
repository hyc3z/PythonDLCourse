
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python.keras import datasets, layers, models

(train_images, train_labels),(test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    # Convolutional base
    # 28 * 28 * 1 -> 26 * 26 * 1 * 32
    # Params = (3*3*1)*32 + 32 = 320
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 26 * 26 * 1 * 32 -> 13 * 13 * 1 * 32
    layers.MaxPool2D((2, 2)),
    # 13 * 13 * 1 * 32 -> 11 * 11 * 1 * 64
    # Params = (3*3*32)*64 + 64 = 18496
    layers.Conv2D(64, (3, 3), activation='relu'),
    # 11 * 11 * 1 * 64 -> 5 * 5 * 1 * 64
    layers.MaxPool2D((2, 2)),
    # 5 * 5 * 1 * 64 -> 3 * 3 * 1 * 64
    # Params = (3*3*64)*64 + 64 = 36928
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Dense layers
    # 3 * 3 * 1 * 64 -> 576 (1D Array)
    layers.Flatten(),
    # Params = 576 * 64 + 64 = 36928
    layers.Dense(64, activation='relu'),
    # Params = 64 * 10 + 10 = 650
    layers.Dense(10, activation='softmax'),
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
              )

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

