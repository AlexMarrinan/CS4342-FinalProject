import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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

trainingData = np.column_stack((sensor2,sensor4))
y = pd.read_csv("labels_CS.csv", header=None)
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

testingData = np.column_stack((sensor2,sensor4))
# Create KNN classifier model
model = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
model.fit(trainingData, y)

y_pred = model.predict(trainingData)

# Calculate the accuracy of the model
accuracy = accuracy_score(y, y_pred)

print("Training Accuracy:")
print(accuracy)



y_test = model.predict(testingData)


output = pd.DataFrame({'sequence': range(25968, 25968+len(y_test)), 'state': y_test})


output.to_csv('output.csv', index=False)
