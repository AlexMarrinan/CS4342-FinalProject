import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers

# Load training data
training_data = pd.read_csv("train_CS.csv", header=None)

sensor_data = np.split(training_data, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((25968, 60, 13))

sensor2 = reshaped_data[:, :, 2]
sensor4 = reshaped_data[:, :, 4]
training_data = np.column_stack((sensor4, sensor2))

# Load training labels
y_train = pd.read_csv("labels_CS.csv", header=None).values.reshape(-1)

# Load testing data
testing_data = pd.read_csv("test_CS.csv", header=None)
sensor_data = np.split(testing_data, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((12218, 60, 13))

sensor2 = reshaped_data[:, :, 2]
sensor4 = reshaped_data[:, :, 4]
testing_data = np.column_stack((sensor4, sensor2))

# Define the model
model = keras.Sequential([
    layers.Dense(12, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_dim=120),
    layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.Dense(3, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(training_data, y_train, epochs=500, batch_size=256)

# Predict training labels
y_train_pred = model.predict(training_data).ravel()
y_train_pred = (y_train_pred >= 0.5).astype(int)
train_accuracy = np.mean(y_train == y_train_pred) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

# Predict testing labels
y_test_pred = model.predict(testing_data).ravel()
y_test_pred = (y_test_pred >= 0.5).astype(int)
output = pd.DataFrame({'sequence': range(25968, 25968 + len(y_test_pred)), 'state': y_test_pred})
output.to_csv('output.csv', index=False)
