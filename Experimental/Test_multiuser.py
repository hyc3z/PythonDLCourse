from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
# import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

train_dir='/home/hu/Downloads/ILSVRC/Data/CLS-LOC/train/'
val_dir='/home/hu/Downloads/ILSVRC/Data/CLS-LOC/val'
save_dir='./'

image_size = 224 # All images will be resized to 160x160
batch_size = 512

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
    )

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    val_dir, # Source directory for the validation images
    target_size=(image_size, image_size),
    batch_size=batch_size,
)

IMG_SHAPE = (image_size, image_size, 3)
np.random.seed(1000)


def AlexNet(save_dir):
    AlexNet = Sequential()
    AlexNet.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))
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
    save_dir += 'AlexNet.h5'
    return AlexNet


def ZFNet(save_dir):
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
    save_dir += 'ZFNet.h5'
    return ZFNet


def VGG16(save_dir):
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
    save_dir += 'VGG16.h5'
    return VGG16


checkpoint_path = "./cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def load_ckpt(callbacks):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = callbacks
    model.load_weights(latest)
    return model



# model =  load_ckpt(AlexNet(save_dir))
model = AlexNet(save_dir)
epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size






cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every epoch.
    period=1)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    workers=8,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[cp_callback],
)



model.save(save_dir)
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,max(plt.ylim())])
# plt.title('Training and Validation Loss')
# plt.show()
#
