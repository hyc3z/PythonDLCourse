from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
import glob

train_dir='/home/hu/Downloads/train_samples/'
val_dir='/home/hu/Downloads/val_samples'
save_dir='./'

image_size = 224 # All images will be resized to 160x160
batch_size = 20

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    val_dir, # Source directory for the validation images
    target_size=(image_size, image_size),
    batch_size=batch_size,
)

IMG_SHAPE = (image_size, image_size, 3)
np.random.seed(1000)

checkpoint_path = "./cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def load_ckpt(callbacks):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = callbacks
    model.load_weights(latest)
    return model



model =  keras.models.load_model('./model2.h5')
# validation_steps = validation_generator.n // batch_size
# predictions = model.evaluate_generator(
#     validation_generator,
#     verbose=1,
# )
#
model.summary()
for i in os.listdir(val_dir):
    print('-------------------')
    for j in os.listdir(os.path.join(val_dir,i)):
        print(j)
        img = keras.preprocessing.image.load_img(os.path.join(val_dir, i, j), target_size=(224, 224))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, 0)
        # print(x.shape)
        print(model.predict(x))

