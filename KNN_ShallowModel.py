import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load training data
training_data = pd.read_csv("train_CS.csv", header=None)
sensor_data = np.split(training_data, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((25968, 60, 13))
sensor2 = reshaped_data[:,:,2]
sensor4 = reshaped_data[:,:,4]
training_data = np.column_stack((sensor2,sensor4))
y = pd.read_csv("labels_CS.csv", header=None).values.reshape(-1)

# Load testing data
testing_data = pd.read_csv("test_CS.csv", header=None)
sensor_data = np.split(testing_data, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((12218, 60, 13))
sensor2 = reshaped_data[:,:,2]
sensor4 = reshaped_data[:,:,4]
testing_data = np.column_stack((sensor2,sensor4))

# Create KNN classifier model and fit it to the training data
model = KNeighborsClassifier(n_neighbors=3)
model.fit(training_data, y)

# Evaluate training accuracy
y_pred_train = model.predict(training_data)
accuracy_train = accuracy_score(y, y_pred_train)
print("Training Accuracy:", accuracy_train)

# Make predictions on the testing data and save output
y_pred_test = model.predict(testing_data)
output = pd.DataFrame({'sequence': range(25968, 25968+len(y_pred_test)), 'state': y_pred_test})
output.to_csv('output.csv', index=False)
