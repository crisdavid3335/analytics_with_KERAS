"""
Created on Tue Nov 30 11:49:51 2021

@author: Christian
"""
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words = 10000)

train_data[0]
train_labels[0]

max([max(sequense) for sequense in train_data])

word_index = imdb.get_word_index()                               # 1
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])       # 2
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]) # 3

import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))      # 1
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.                           # 2
    return results

x_train = vectorize_sequences(train_data)                # 3 
x_test = vectorize_sequences(test_data)                  # 4

x_train[0]

y_train = np.asarray(train_labels).astype(np.float_)
y_test = np.asarray(test_labels).astype(np.float_)

from tensorflow import keras 
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation = 'relu'),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')])

model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

x_val = x_train[: 10000]
partial_x_train = x_train[10000 :]

y_val = y_train[: 10000]
partial_y_train = y_train[10000 : ]

history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'b', label = 'Training loss')
plt.plot(epochs, val_loss_values, '+r', label = 'Validation loss')
plt.title('Training and validations loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()                                                        # 1 
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'b', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'or', label = 'Validation acc')
plt.title('Training and validations acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


model = keras.Sequential([
    layers.Dense(16, activation = 'relu'),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')])

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 4, batch_size = 512)
results = model.evaluate(x_test, y_test)

results # 1

model.predict(x_test)
