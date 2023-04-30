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
trainingData = np.column_stack((sensor0,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8,sensor9,sensor10,sensor11,sensor12,sensor2,sensor4))
y = pd.read_csv("labels_CS.csv")
y = y.values.reshape(-1)

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(trainingData)

# Visualize data and labels in PCA space
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA visualization of training data")
plt.show()
