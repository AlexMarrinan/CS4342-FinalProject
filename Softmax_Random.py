
#SOURCES: This code is a modified version of HW3 for CS4342
import pandas as pd
import numpy as np

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix m by 2. Where m is the features.(example:sensor data)
# and 2 is the diminsion of whether they seeked help or not
# Then return Wtilde.
def softmaxRegression(trainingData, trainingLabels, epsilon, batchSize, alpha):
    
    m = trainingData.shape[1] # this is the number of features in the data plus one for the bias term.
    Wtilde = np.random.randn(m, 2) * 1e-5 # this produces features + bias term by the number of classes with random number.
    # This initializes the weight vectors
    
    n = trainingData.shape[0] # number of sequences in the set
    index = np.random.permutation(n) # get random permutation of size n so we can have random batches
    
    batches = int(n / batchSize) # number of sequences / by batch size gives us how many batches are needed
    
    for i in range(1000): # this is for epochs we can change this number
        for j in range(batches):
            indices = index[j * batchSize:(j + 1) * batchSize] # get the corresponding indices that we will need for the corresponding trainingData and Labels
            X = trainingData[indices] # get the data
            Y = trainingLabels[indices] # get the labels
            
            W_NoBias = np.vstack((Wtilde[:-1], np.zeros((1,2)))) # W without the bias. Making the last row zeros so it has no effect on bias. Using this for regularization term.
             
            yhat = np.exp(X.dot(Wtilde)) / np.sum(np.exp(X.dot(Wtilde)), axis=1, keepdims=True) # this is the squashing function making everything between 0 and 1 and adding to one as a sum
            
            gradient = X.T.dot(yhat - Y) / batchSize + (alpha / (2 * n)) * np.sum(W_NoBias ** 2) # gradient 1/n(X(yhat-Y)) The same thing here as above X was already in transposed format so X.T would give non transposed format
            Wtilde -= epsilon * gradient # updating weights
            
    return Wtilde # return weights

if __name__ == "__main__":
# Load training data
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


# Load testing data
testingData = pd.read_csv("test_CS.csv", header=None)

# Split testing data into 13 sensor inputs, stack them together, and reshape them for training and testing
sensor_data = np.split(testingData, indices_or_sections=13, axis=1)
stacked_data = np.hstack(sensor_data)
reshaped_data = stacked_data.reshape((12218, 60, 13))

# Use sensor data from only sensor 2 and sensor 4 for testing
testingData = np.column_stack((reshaped_data[:,:,2],reshaped_data[:,:,4]))

# One-hot encode the output values for training the logistic regression model
OneHotTraining = np.eye(2)[y]

# Train the model using softmax regression
Wtilde = softmaxRegression(trainingData, OneHotTraining, 0.001, 375, 0.1)

# Calculate probabilities of the output for the training data
training_probabilities =  np.exp(trainingData.dot(Wtilde)) / np.sum(np.exp(trainingData.dot(Wtilde)), axis=1, keepdims=True)

# Convert probabilities to predictions for the training data
training_predictions = np.zeros(training_probabilities.shape[0], dtype=int)
for i in range(training_probabilities.shape[0]):
    if training_probabilities[i, 1] > training_probabilities[i, 0]:
        training_predictions[i] = 1

# Calculate the training accuracy
training_accuracy = np.mean(y == training_predictions) * 100
print("Training Accuracy: %.2f%%" % training_accuracy)

# Calculate probabilities of the output for the testing data
testing_probabilities =  np.exp(testingData.dot(Wtilde)) / np.sum(np.exp(testingData.dot(Wtilde)), axis=1, keepdims=True)

# Convert probabilities to predictions for the testing data
testing_predictions = np.zeros(testing_probabilities.shape[0], dtype=int)
for i in range(testing_probabilities.shape[0]):
    if testing_probabilities[i, 1] > testing_probabilities[i, 0]: 
        testing_predictions[i] = 1 

# Generate output file for the predicted values of the testing data
output = pd.DataFrame({'sequence': range(25968, 25968+len(testing_predictions)), 'state': testing_predictions})
output.to_csv('output.csv', index=False)

# Generate random predicted values and output to file
randomPredictions = np.random.randint(2, size=y.shape[0])
randomOutput = pd.DataFrame({'sequence': range(25968, 25968+len(randomPredictions[:12218])), 'state': randomPredictions[:12218]})
randomOutput.to_csv('random.csv', index=False)

# Calculate random guess accuracy
random_guess_accuracy = np.mean(y == randomPredictions) * 100
print("Random Guess Accuracy: %.2f%%" % random_guess_accuracy)
    





