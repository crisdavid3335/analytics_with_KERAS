"""
Created on Mon Nov 29 22:49:57 2021

@author: Christian
"""

import numpy as np
np.set_printoptions(suppress = True, linewidth = 100, precision = 2)

# Cargamos los datos 
raw_data_np = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\train.csv",
                            delimiter = ',',
                            dtype = np.str_,
                            autostrip = True,
                            usecols = np.arange(0,12),)

raw_data_np

# Eliminamos los titulos
raw_data_np = np.genfromtxt('C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\train.csv',
                             delimiter = ',',
                             skip_header = 1,
                             autostrip = True,
                             usecols = np.arange(0,13))

raw_data_np

# Datos faltantes
np.isnan(raw_data_np).sum()

# Estadisticas de reemplazo 
temporary_fill = np.nanmax(raw_data_np) + 1
temporary_mean = np.nanmean(raw_data_np, axis = 0)

temporary_mean

# Variable estadistica
temporary_stats = np.array([np.nanmin(raw_data_np, axis = 0),
                            temporary_mean,
                            np.nanmax(raw_data_np, axis = 0)])

temporary_stats = np.delete(temporary_stats, 4, axis = 1)
temporary_stats[:, 8]

# columnas no numericas
columns_strings1 = np.argwhere(np.isnan(temporary_mean)).squeeze()
columns_strings2 = np.copy(columns_strings1) 
columns_strings2 = np.delete(columns_strings2, 2)
columns_strings2[2:] = columns_strings2[2:] - 1
columns_strings2

# columnas numericas 
columns_numeric1 = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
columns_numeric2 = np.copy(columns_numeric1)
columns_numeric2[3:] = columns_numeric2[3:] - 1
columns_numeric2

# datos no numericos y numericos
Loan_data_strings = np.genfromtxt('C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\train.csv',
                                  delimiter = ',',
                                  skip_header = 1,
                                  autostrip = True,
                                  usecols = columns_strings1,
                                  dtype = np.str_)


Loan_data_numeric = np.genfromtxt('C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\train.csv',
                                  delimiter = ',',
                                  skip_header = 1,
                                  autostrip = True,
                                  usecols = columns_numeric1,
                                  filling_values = temporary_fill)

Loan_data_strings[0]
Loan_data_numeric[0]

Loan_data_strings_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\test.csv",
                                  delimiter = ',',
                                  skip_header = 1,
                                  autostrip = True,
                                  usecols = range(0, 12),
                                  dtype = np.str_)

Loan_data_strings_test = np.delete(Loan_data_strings_test, columns_numeric2, axis = 1)

columns_numeric_test = np.array([0,1,5,6,7,8,9])

Loan_data_numeric_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\test.csv",
                                  delimiter = ',',
                                  skip_header = 1,
                                  autostrip = True,
                                  usecols = columns_numeric_test)

Names_full = np.genfromtxt('C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\train.csv',
                           delimiter = ',',
                           autostrip = True,
                           skip_footer = raw_data_np.shape[0],
                           dtype = np.str_)
Names_full

Names_full_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\titanic\\test.csv",
                           delimiter = ',',
                           autostrip = True,
                           skip_footer = Loan_data_numeric_test.shape[0],
                           dtype = np.str_)
Names_full_test
columns_strings_test = np.array([2,3,9,10])
columns_numeric_test = np.array([0,1,4,5,6,7,8])

Loan_data_strings[:, 0] = np.char.add(Loan_data_strings[:, 0], ' ; ')
Loan_data_strings[:, 0] = np.char.add(Loan_data_strings[:, 0], Loan_data_strings[:, 1])
Loan_data_strings = np.delete(Loan_data_strings, 1, axis = 1)


# Dividimos los nombres por el tipo de columnas
Names_strings, Names_numeric = Names_full[columns_strings2], Names_full[columns_numeric2]

Names_strings_test, Names_numeric_test = Names_full_test[columns_strings_test], Names_full_test[columns_numeric_test]

Names_strings
Names_numeric

Names_strings_test
Names_numeric_test

# Los nombres
Loan_data_strings[:, 0]
Loan_data_strings_test[:, 0]
np.unique(Loan_data_strings[:, 0], return_counts = True)
# poseen un " por lo que podemos eliminarlos
Loan_data_strings[:, 0] = np.chararray.strip(Loan_data_strings[:, 0], '"')
# reemplazamos los titulos por numeros
identifiers = np.array(['Master', 'Miss', 'Ms', 'Mme', 'Mlle', 'Mrs', 'Mr', 'Rare', 'Dr'])

for i in range(0, len(identifiers)):
    Loan_data_strings[:, 0] = np.where(np.char.count(Loan_data_strings[:, 0], identifiers[i]), 
                                       i + 1, 
                                       Loan_data_strings[:, 0])

Loan_data_strings[:, 0] = np.where(np.char.isdigit(Loan_data_strings[:, 0]), 
                                       Loan_data_strings[:, 0], 
                                       0)

for i in range(0, len(identifiers)):
    Loan_data_strings_test[:, 0] = np.where(np.char.count(Loan_data_strings_test[:, 0], identifiers[i]), 
                                       i + 1, 
                                       Loan_data_strings_test[:, 0])

Loan_data_strings_test[:, 0] = np.where(np.char.isdigit(Loan_data_strings_test[:, 0]), 
                                       Loan_data_strings_test[:, 0], 
                                       0)

Names_strings[0] = 'Identifier'
Names_strings

Names_strings_test[0] = 'Identifier'
Names_strings_test


# El sexo
Loan_data_strings[:, 1]
Loan_data_strings_test[:, 1]
np.unique(Loan_data_strings[:, 1], return_counts = True)
# dummy mujer = 1, hombre = 0
Loan_data_strings[:, 1] = np.where(Loan_data_strings[:,1] == 'female', 1, 0)

Loan_data_strings_test[:, 1] = np.where(Loan_data_strings_test[:,1] == 'female', 1, 0)


# Cabina
Loan_data_strings[:, 2]
Loan_data_strings_test[:, 2]
np.unique(Loan_data_strings[:, 2], return_counts = True)
# reemplazamos los datos faltantes con x    
Loan_data_strings[:, 2] = np.where(Loan_data_strings[:, 2] == '', 
                                   0, Loan_data_strings[:, 2])

Loan_data_strings_test[:, 2] = np.where(Loan_data_strings_test[:, 2] == '', 
                                   0, Loan_data_strings_test[:, 2])

# reemplazamos las letras por numeros
letters = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'T'])

for i in range(0, len(letters)):
    Loan_data_strings[:, 2] = np.where(np.char.startswith(Loan_data_strings[:, 2], letters[i]),
                                       i + 1,
                                       Loan_data_strings[:, 2])

for i in range(0, len(letters)):
    Loan_data_strings_test[:, 2] = np.where(np.char.startswith(Loan_data_strings_test[:, 2], letters[i]),
                                       i + 1,
                                       Loan_data_strings_test[:, 2])

# Embarque
Loan_data_strings[:, 3]
np.unique(Loan_data_strings[:, 3], return_counts = True)
# reemplazamos los datos faltantes con numeros
Loan_data_strings[:, 3] = np.where((Loan_data_strings[:, 3] == '') | (Loan_data_strings[:, 3] == 'S'), 1, 
                                   Loan_data_strings[:, 3])

Loan_data_strings[:, 3] = np.where(Loan_data_strings[:, 3] == 'C', 2, 
                                   Loan_data_strings[:, 3])

Loan_data_strings[:, 3] = np.where(Loan_data_strings[:, 3] == 'Q', 3, 
                                   Loan_data_strings[:, 3])

Loan_data_strings_test[:, 3]
Loan_data_strings_test[:, 3] = np.where((Loan_data_strings_test[:, 3] == '') | (Loan_data_strings_test[:, 3] == 'S'), 1, 
                                   Loan_data_strings_test[:, 3])

Loan_data_strings_test[:, 3] = np.where(Loan_data_strings_test[:, 3] == 'C', 2, 
                                   Loan_data_strings_test[:, 3])

Loan_data_strings_test[:, 3] = np.where(Loan_data_strings_test[:, 3] == 'Q', 3, 
                                   Loan_data_strings_test[:, 3])

Loan_data_strings.astype(np.int_)
Loan_data_strings_test.astype(np.int_)

# ID del pasagero
np.isin(Loan_data_numeric[:, 0], temporary_fill).sum()
np.isin(Loan_data_numeric_test[:, 0], temporary_fill).sum()

# Supervivencia
np.isin(Loan_data_numeric[:, 1], temporary_fill).sum()

# Vamos a reemplazar todos los valores de las numericas
Names_numeric
Names_numeric_test

for i in range(0, Loan_data_numeric.shape[1]):
    Loan_data_numeric[:, i] = np.where((Loan_data_numeric[:, i] == temporary_fill),
                                      temporary_stats[2, columns_numeric2[i]],
                                      Loan_data_numeric[:, i])

temporary_mean_test = np.nanmean(Loan_data_numeric_test, axis = 0)
temporary_mean_test

for i in range(0, Loan_data_numeric_test.shape[1]):
    Loan_data_numeric_test[:, i] = np.where(np.isnan(Loan_data_numeric_test[:, i]),
                                      temporary_mean_test[i],
                                      Loan_data_numeric_test[:, i])

# Se reemplaza con la media de los datos de training porque en principio se espera
# que no se posean las medias de los datos nuevos
# Revizamos
np.isin(Loan_data_numeric, temporary_fill).sum()
Loan_data_numeric.shape

np.isin(Loan_data_numeric_test, temporary_fill).sum()
Loan_data_numeric_test.shape

# eliminamos el id
id_test = Loan_data_numeric_test[:, 0]
Loan_data_numeric = np.delete(Loan_data_numeric, 0, axis = 1)
Loan_data_numeric_test = np.delete(Loan_data_numeric_test, 0, axis = 1)

Names_numeric = np.delete(Names_numeric, 0)
Names_numeric_test = np.delete(Names_numeric_test, 0)
Full_names = np.hstack([Names_numeric, Names_strings])
Full_names

Full_names_test = np.hstack([Names_numeric_test, Names_strings_test])
Full_names_test

train_data = np.hstack((Loan_data_numeric, Loan_data_strings))
test_data = np.hstack((Loan_data_numeric_test, Loan_data_strings_test))
train_data = np.vstack((Full_names, train_data))

y_train = np.copy(train_data[:,0])
x_train = np.delete(train_data, 0, axis = 1)
x_test = np.vstack((Full_names_test, test_data))

x_train = x_train[1:].astype(np.float_)
x_test = x_test[1:].astype(np.float_)
y_train = y_train[1:].astype(np.float_)

x_mean = np.mean(x_train, axis = 0)
x_dist = np.std(x_train, axis = 0)

x_mean_test = np.mean(x_test, axis = 0)
x_dist_test = np.std(x_test, axis = 0)

x_train = (x_train - x_mean)/ x_dist
x_test = (x_test - x_mean_test)/ x_dist_test

# Modelo
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf 

model = keras.Sequential([
    layers.Dense(16, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(6, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')])

custom_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

model.compile(optimizer = custom_optimizer, 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

x_train.shape[0]*0.2

x_val = x_train[: 178]
partial_x_train = x_train[178 :]

y_val = y_train[: 178]
partial_y_train = y_train[178 : ]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',    
    min_delta = 0,
    patience = 5,
    verbose = 1, 
    restore_best_weights = True)

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 200,
                    batch_size = 30,
                    callbacks = [early_stopping],
                    validation_data = (x_val, y_val))

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'b', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'r', label = 'Validation loss')
plt.title('Training and validations loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()                                                        # 1 
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'b', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'r', label = 'Validation acc')
plt.title('Training and validations acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

summit= model(x_test)
summit.shape
summit = np.where(summit >= 0.50, 1, 0)
summit = np.reshape(summit, (418,))

import pandas as pd

submission = pd.DataFrame({"PassengerId": id_test,"Survived": summit})
submission = submission.astype('int32')
submission.to_csv('C:\\Users\\Christian\\Desktop\\submission2.csv', index=False)

