
#SOURCES: This code is a modified version of work I did in CS4342



import pandas as pd
import numpy as np
import csv



# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix m by 2. Where m is the features.(example:sensor data)
# and 2 is the diminsion of whether they seeked help or not
# Then return Wtilde.




def softmaxRegression(trainingData, trainingLabels, epsilon, batchSize, alpha):
    
    
    m = trainingData.shape[1]# this is the number of features in the data plus one for the bias term.
    Wtilde = np.random.randn(m, 2) * 1e-5 #this produces features + bias term by the number of classes with random number.
    #This initalize the weight vectors
   
    n=trainingData.shape[0]; # number of sequences in the set
 


    index = np.random.permutation(n) #get random permutation of size n so we can have random batchs
    
    batches = int( n / batchSize)# number of sequences / by batch size gives us how many batches are needed
    
    
    for i in range(1000): # this is for epochs we can change this number
        
                 
        for j in range(batches):
            
           
            indices = index[j * batchSize:(j + 1) * batchSize]#get the corresponding indices that we will need for the
            #corresponding trainingDate and Labels

            X = trainingData[indices]# get the data
            Y = trainingLabels[indices]#get the labels

            
            
           
            W_NoBias= np.vstack((Wtilde[:-1], np.zeros((1,2))))#W without the bias. Maxing the last row zeros so it 
            #has no effect on bias. Using this for regulization term
            
             
            #this is the sqaushing function making everything between 0 and 1 and adding to one as a sum
            yhat = np.exp(X.dot(Wtilde)) / np.sum(np.exp(X.dot(Wtilde)), axis=1, keepdims=True) 
            
           
            #gradient 1/n(X(yhat-Y)) The same thing here as above X was already in tranposed  format so
            #X.T would give non transposed format
            gradient = X.T.dot(yhat - Y)/batchSize + (alpha/(2*n)) * np.sum(W_NoBias**2) 
            Wtilde -= epsilon * gradient#updating weights
            

    return Wtilde #return weights





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

    
    
    OneHotTraining = np.eye(2)[y]
    
   

    #training model
    
    Wtilde = softmaxRegression(trainingData, OneHotTraining, 0.001, 375, 0.1)




    probabilities =  np.exp(trainingData.dot(Wtilde)) / np.sum(np.exp(trainingData.dot(Wtilde)), axis=1, keepdims=True)


    # probabilities to predictions
    predictions = np.zeros(probabilities.shape[0],dtype = int)#to store predictions
    for i in range(probabilities.shape[0]):# for each yhat probabilities
        if probabilities[i, 1] > probabilities[i, 0]: # if the probability of surviving is greater then not surviving
            predictions[i] = 1 # if so then say person survived


    pc = np.mean(y == predictions) * 100

    print("Training Accuracy:")
    print(pc)
    
    
    probabilities =  np.exp(testingData.dot(Wtilde)) / np.sum(np.exp(testingData.dot(Wtilde)), axis=1, keepdims=True)


    # probabilities to predictions
    predictions = np.zeros(probabilities.shape[0],dtype = int)#to store predictions
    for i in range(probabilities.shape[0]):# for each yhat probabilities
        if probabilities[i, 1] > probabilities[i, 0]: 
            predictions[i] = 1 


    output = pd.DataFrame({'sequence': range(25968, 25968+len(predictions)), 'state': predictions})


    output.to_csv('output.csv', index=False)
    
    
    randomPredictions = np.random.randint(2, size=y.shape[0])
    
    
    randomOutput = pd.DataFrame({'sequence': range(25968, 25968+len(randomPredictions[:12218])), 'state': randomPredictions[:12218]})


    randomOutput.to_csv('random.csv', index=False)

    # Calculate random guess accuracy
    random_guess_accuracy = np.mean(y == random_predictions) * 100

    print("Random Guess Accuracy:")
    print(random_guess_accuracy)
    





