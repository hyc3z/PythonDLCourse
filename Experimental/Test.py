from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import os
base_dir = '/home/hu/tensorflow_datasets/cats_vs_dogs/PetImages'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
print(train_dir)
# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
print ('Total training cat images:', len(os.listdir(train_cats_dir)))

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
print ('Total training dog images:', len(os.listdir(train_dogs_dir)))

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
print ('Total validation cat images:', len(os.listdir(validation_cats_dir)))

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print ('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

image_size = 224 # All images will be resized to 160x160
batch_size = 32

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                validation_dir, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='binary')

IMG_SHAPE = (image_size, image_size, 3)
np.random.seed(1000)


def AlexNet():
    AlexNet = Sequential()
    AlexNet.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    AlexNet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    AlexNet.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    AlexNet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    AlexNet.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    AlexNet.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    AlexNet.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    AlexNet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    AlexNet.add(Flatten())
    AlexNet.add(Dense(4096,activation='relu'))
    AlexNet.add(Dropout(0.5))
    AlexNet.add(Dense(4096,activation='relu'))
    AlexNet.add(Dropout(0.5))
    AlexNet.add(Dense(1000,activation='softmax'))
    AlexNet.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    AlexNet.summary()
    return AlexNet


def ZFNet():
    ZFNet = Sequential()
    ZFNet.add(Conv2D(96,(7,7),strides=(2,2),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    ZFNet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    ZFNet.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))
    ZFNet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    ZFNet.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    ZFNet.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    ZFNet.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    ZFNet.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    ZFNet.add(Flatten())
    ZFNet.add(Dense(4096,activation='relu'))
    ZFNet.add(Dropout(0.5))
    ZFNet.add(Dense(4096,activation='relu'))
    ZFNet.add(Dropout(0.5))
    ZFNet.add(Dense(1000,activation='softmax'))
    ZFNet.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    ZFNet.summary()
    return ZFNet


def VGG16():
    VGG16 = Sequential()
    VGG16.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(MaxPooling2D(pool_size=(2,2)))
    VGG16.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(MaxPooling2D(pool_size=(2,2)))
    VGG16.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(MaxPooling2D(pool_size=(2,2)))
    VGG16.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(MaxPooling2D(pool_size=(2,2)))
    VGG16.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    VGG16.add(MaxPooling2D(pool_size=(2,2)))
    VGG16.add(Flatten())
    VGG16.add(Dense(4096,activation='relu'))
    VGG16.add(Dropout(0.5))
    VGG16.add(Dense(4096,activation='relu'))
    VGG16.add(Dropout(0.5))
    VGG16.add(Dense(1000,activation='softmax'))
    VGG16.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    VGG16.summary()
    return VGG16

