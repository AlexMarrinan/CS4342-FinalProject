#SOURCES
#https://keras.io/guides/sequential_model/
#https://keras.io/api/losses/
#https://keras.io/api/optimizers/

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers


trainingData = pd.read_csv("train_CS.csv", header=None)

sensor_data = np.split(trainingData, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((25968, 60, 13))

sensor2 = reshaped_data[:,:,2]
sensor4 = reshaped_data[:,:,4]


trainingData =  np.column_stack((sensor4,sensor2))
y = pd.read_csv("labels_CS.csv",header = None)
y = y.values.reshape(-1)

testingData = pd.read_csv("test_CS.csv", header=None)

sensor_data = np.split(testingData, indices_or_sections=13, axis=1)

stacked_data = np.hstack(sensor_data)

reshaped_data = stacked_data.reshape((12218, 60, 13))


sensor2 = reshaped_data[:,:,2]
sensor4 = reshaped_data[:,:,4]


testingData = np.column_stack((sensor4,sensor2))

model = keras.Sequential()


model.add(layers.Dense(12, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_dim=120))
model.add(layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(layers.Dense(3, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


history = model.fit(trainingData, y, epochs=500, batch_size=256)

y_train = model.predict(trainingData)

y_train= y_train.ravel()

y_train = (y_train >= 0.5).astype(int)

match_percentage = np.mean(y == y_train) * 100
    
print("Training Accuracy:")
print(match_percentage)

y_test = model.predict(testingData)

y_test= y_test.ravel()

y_test = (y_test >= 0.5).astype(int)

output = pd.DataFrame({'sequence': range(25968, 25968+len(y_test)), 'state': y_test})


output.to_csv('output.csv', index=False)
