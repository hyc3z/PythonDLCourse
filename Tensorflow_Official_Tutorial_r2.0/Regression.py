from __future__ import absolute_import

import pathlib
# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys,time

print(tf.__version__)

dataset_path = './auto-mpg.data'
# dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail(10))
print(dataset.isna().sum())
dataset = dataset.dropna()

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail(10))

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]])
# plt.show()

train_stats = train_dataset.describe()
print(train_stats)
train_stats.pop("MPG")
print(train_stats)
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


model = build_model()

model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


EPOCHS = 1000


class PrintDot(keras.callbacks.Callback):
    LastTime = time.time()
    Remain = 0
    LastDelta = 0
    def on_epoch_end(self, epoch, logs):
        delta = time.time() - self.LastTime
        self.LastTime = time.time()
        self.Remain += delta
        if self.LastDelta == 0:
            self.LastDelta = delta
        percentile = epoch/EPOCHS*100
        num = int(percentile/5)
        sys.stdout.write(' ' * 30 + '\r')
        sys.stdout.flush()
        if self.Remain >= 1:
            sys.stdout.write('Training... '+str(int(percentile))+'% ['+'='*num + '>'+'-'*(20-num)+'] ('+str(epoch)+'/'+str(EPOCHS)+')'+str(round(delta*1000,2))+'ms/epoch \r')
            self.LastDelta = delta
            self.Remain -= 1
        else:
            sys.stdout.write('Training... '+str(int(percentile))+'% ['+'='*num + '>'+'-'*(20-num)+'] ('+str(epoch)+'/'+str(EPOCHS)+')'+str(round(self.LastDelta*1000,2))+'ms/epoch \r')
        sys.stdout.flush()


history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


