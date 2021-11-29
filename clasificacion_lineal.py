"""
Created on Mon Nov 29 17:07:53 2021

@author: Christian
"""
#Librerias
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Datos
num_samples = 5000

negative_class = np.random.multivariate_normal(
    mean = [5 , 10], 
    cov = [[1.5, 0.5], [0.5, 1.5]],
    size = num_samples)

positive_class = np.random.multivariate_normal(
    mean = [10 , 5], 
    cov = [[1.5, 0.5], [0.5, 1.5]],
    size = num_samples)

inputs = np.vstack((negative_class, positive_class)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples, 1), dtype = np.float_),
                    np.ones((num_samples, 1), dtype = np.float_)))

# grafico
plt.scatter(inputs[:, 0], inputs[:, 1], c = targets[:, 0])
plt.show()

# modelo
input_dim = 2
output_dim = 1
# parametros 
w = tf.Variable(initial_value = tf.random.uniform(shape = (input_dim, output_dim)))
b = tf.Variable(initial_value = tf.zeros(shape = (output_dim, )))

def model(inputs):
    return tf.matmul(inputs, w) + b

def square_loss(targets, predictions):
    per_sample_loss = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_loss)


lr = 0.001

def training(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_w, grad_loss_b = tape.gradient(loss, [w, b])
    w.assign_sub(grad_loss_w * lr)
    b.assign_sub(grad_loss_b * lr)
    return loss

for step in range(50):
    loss = training(inputs, targets)
    print(f'Loss at step {step}: {loss:.4f}')
    
predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c = predictions[:, 0] > 0.5)
plt.show()


x = np.linspace(2, 13, 100)
y = -w[0] / w[1] * x + (0.5 - b) / w[1]
plt.plot(x, y, '-r')
plt.scatter(inputs[:, 0], inputs[:, 1], c = predictions[:, 0] > 0.5)
plt.show()
