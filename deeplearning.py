#SOURCES
#https://keras.io/guides/sequential_model/
#https://keras.io/api/losses/
#https://keras.io/api/optimizers/

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



trainingData = pd.read_csv("train_CS.csv", header=None)

sensor_data = np.split(trainingData, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((25968, 60, 13))

sensor0 = reshaped_data[:,:,0]
sensor1 = reshaped_data[:,:,1]
sensor2 = reshaped_data[:,:,2]
sensor3 = reshaped_data[:,:,3]
sensor4 = reshaped_data[:,:,4]
sensor5 = reshaped_data[:,:,5]
sensor6 = reshaped_data[:,:,6]
sensor7 = reshaped_data[:,:,7]
sensor8 = reshaped_data[:,:,8]
sensor9 = reshaped_data[:,:,9]
sensor10 = reshaped_data[:,:,10]
sensor11 = reshaped_data[:,:,11]
sensor12 = reshaped_data[:,:,12]

trainingData = np.column_stack((sensor0,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8,sensor9,sensor10,sensor11,sensor12))
y = pd.read_csv("labels_CS.csv",header = None)
y = y.values.reshape(-1)

testingData = pd.read_csv("test_CS.csv", header=None)

sensor_data = np.split(testingData, indices_or_sections=13, axis=1)

stacked_data = np.hstack(sensor_data)

reshaped_data = stacked_data.reshape((12218, 60, 13))

sensor0 = reshaped_data[:,:,0]
sensor1 = reshaped_data[:,:,1]
sensor2 = reshaped_data[:,:,2]
sensor3 = reshaped_data[:,:,3]
sensor4 = reshaped_data[:,:,4]
sensor5 = reshaped_data[:,:,5]
sensor6 = reshaped_data[:,:,6]
sensor7 = reshaped_data[:,:,7]
sensor8 = reshaped_data[:,:,8]
sensor9 = reshaped_data[:,:,9]
sensor10 = reshaped_data[:,:,10]
sensor11 = reshaped_data[:,:,11]
sensor12 = reshaped_data[:,:,12]

testingData = np.column_stack((sensor0,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8,sensor9,sensor10,sensor11,sensor12))

model = keras.Sequential()


model.add(layers.Dense(100, activation='relu', input_dim=780))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


history = model.fit(trainingData, y, epochs=100, batch_size=32)

test_loss, test_acc = model.evaluate(trainingData, y)

y_test = model.predict(testingData)

y_test= y_test.ravel()

y_test = (y_test >= 0.5).astype(int)

output = pd.DataFrame({'sequence': range(25968, 25968+len(y_test)), 'state': y_test})


output.to_csv('output.csv', index=False)

print('Test accuracy:', test_acc)

