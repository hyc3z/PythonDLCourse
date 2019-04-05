from __future__ import absolute_import, division, print_function

import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

print(train_data[0])

word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])

maxlen = 500

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=maxlen)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=maxlen)

embedding_dim=16

model = keras.Sequential([
    #     layers.Embedding(10000, 16, input_length=500),
    # Params = 10000 (vectors) * 16 (size of each vector) = 160000
    # Output : 500 vectors (mathematical average of each embedded vector)* 16(size)
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    # I think it's crazy to dramatically shrink a 500*16 sized tensor to a 1D Array with only 16 elements
    layers.GlobalAveragePooling1D(),
    # Params: 16*16 + 16=272
    layers.Dense(16, activation='relu'),
    # Params: 16*1 + 1 = 17
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2
)

import matplotlib.pyplot as plt

print(history.history)
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.figure(figsize=(16,9))

plt.show()
