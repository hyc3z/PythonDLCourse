import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

# URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
URL = './heart.csv'
dataframe = pd.read_csv(URL)
print(dataframe.head())
print(len(dataframe), 'total')
print()
train, test = train_test_split(dataframe, test_size=0.2)
print(len(train), 'train+val examples')
print(len(test), 'test examples')
print()
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    # pop:Return item and drop from frame. Raise KeyError if not found.
    labels = dataframe.pop('target')
    # print(labels)
    # Creates a Dataset whose elements are slices of the given tensors.
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # print(ds)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = dataframe_to_dataset(train, batch_size=batch_size)
val_ds = dataframe_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = dataframe_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  #   Creates a Dataset with at most count elements from this dataset.
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )


example_batch = next(iter(train_ds))[0]


def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())


age = feature_column.numeric_column("age")
demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# Key point: using an embedding column is best when a categorical column has many possible values. We are using one here for demonstration purposes, so you have a complete example you can modify for a different dataset in the future.
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000
)
demo(feature_column.indicator_column(thal_hashed))

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

# Choose which column to use
feature_columns = []
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = dataframe_to_dataset(train, batch_size=batch_size)
val_ds = dataframe_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = dataframe_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=5
)

loss, accuracy = model.evaluate(test_ds)
model.save('./Classify_structured_data')
print("Accuracy", accuracy)


