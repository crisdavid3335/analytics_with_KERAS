"""
Created on Fri Dec  3 13:46:31 2021

@author: Christian
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

def vectorise_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train_validation = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)

from tensorflow.keras.utils import to_categorical

y_train_validation = to_categorical(train_labels)
y_test = to_categorical(test_labels)

indices = np.random.permutation(x_train_validation.shape[0])
num_val_samples = int(0.2 * indices.shape[0])

x_train_validation = x_train_validation[indices]
y_train_validation = y_train_validation[indices]

x_val = x_train_validation[: num_val_samples]
x_train = x_train_validation[num_val_samples :]

y_val = y_train_validation[: num_val_samples]
y_train = y_train_validation[num_val_samples :]


import tensorflow.keras as keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(46, activation = 'softmax')])

model.compile(optimizer = tf.keras.optimizers.RMSprop(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

history = model.fit(x_train, y_train,
                    epochs = 20, batch_size = 512, 
                    validation_data = (x_val, y_val))

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'y', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label= 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

results = model.evaluate(x_test, y_test)
