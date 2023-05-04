#SOURCES
#https://keras.io/guides/sequential_model/
#https://keras.io/api/losses/
#https://keras.io/api/optimizers/

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the training data
training_data = pd.read_csv("train_CS.csv", header=None)

# Split the sensor data and reshape it
sensor_data = np.split(training_data, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((25968, 60, 13))

# Split the reshaped data into individual sensors
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

# Combine the sensor data into a single array
training_data = np.column_stack((sensor0, sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8, sensor9, sensor10, sensor11, sensor12))

# Load the training labels
y = pd.read_csv("labels_CS.csv", header=None)
y = y.values.reshape(-1)

# Load the testing data
testing_data = pd.read_csv("test_CS.csv", header=None)

# Split the sensor data and reshape it
sensor_data = np.split(testing_data, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((12218, 60, 13))

# Split the reshaped data into individual sensors
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

# Combine the sensor data into a single array
testing_data = np.column_stack((sensor0, sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8, sensor9, sensor10, sensor11, sensor12))

# Create the neural network model
model = keras.Sequential()
model.add(layers.Dense(100, activation='relu', input_dim=780))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Train the model
history = model.fit(training_data, y, epochs=100, batch_size=32)

# Evaluate the model on the training data
test_loss, test_acc = model.evaluate(training_data, y)

# Make predictions on the testing data
y_test = model.predict(testing_data)
y_test= y_test.ravel()
y_test = (y_test >= 0.5).astype(int)

# Save the predictions to a CSV file
output = pd.DataFrame({'sequence': range(25968, 25968+len(y_test)), 'state': y_test})

output.to_csv('output.csv', index=False)

print('Test accuracy:', test_acc)

