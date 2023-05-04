import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load training data
trainingData = pd.read_csv("train_CS.csv")
sensor_data = np.split(trainingData, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((25967, 60, 13))
sensor0 = reshaped_data[:,:1,0]
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
trainingDataAll = np.column_stack((sensor0,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8,sensor9,sensor10,sensor11,sensor12))
y = pd.read_csv("labels_CS.csv")
y = y.values.reshape(-1)

# Reduce dimensionality with PCA
pcaAll = PCA(n_components=2)
pca_dataAll = pcaAll.fit_transform(trainingDataAll)

# Visualize data and labels in PCA space for all sensors
colorsAll = ['blue' if label == 0 else 'red' for label in y]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_dataAll[:, 0], pca_dataAll[:, 1], c=colorsAll)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA visualization of training data for all sensors")

# Reduce dimensionality with PCA for sensors 2 and 4
trainingDataReduced = np.column_stack((sensor2, sensor4))
pcaReduced = PCA(n_components=2)
pca_dataReduced = pcaReduced.fit_transform(trainingDataReduced)

# Visualize data and labels in PCA space for sensors 2 and 4
colorsReduced = ['blue' if label == 0 else 'red' for label in y]

plt.subplot(1, 2, 2)
plt.scatter(pca_dataReduced[:, 0], pca_dataReduced[:, 1], c=colorsReduced)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA visualization of training data with sensor 2 and sensor 4")
plt.tight_layout()
plt.show()
