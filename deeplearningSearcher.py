#SOURCES
#https://keras.io/guides/sequential_model/
#https://keras.io/api/losses/
#https://keras.io/api/optimizers/

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    print("START")

    trainingData = pd.read_csv("train_CS.csv", header=None)

    sensor_data = np.split(trainingData, indices_or_sections=13, axis=1)
    #print(sensor_data.shape) is a list
    stacked_data = np.hstack(sensor_data)
    print(stacked_data.shape)#    25968,780
    reshaped_data = stacked_data.reshape((25968, 60, 13))
    print(reshaped_data.shape)#  (25968, 60, 13)
    #take out 2 and 8
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



    y = pd.read_csv("labels_CS.csv",header = None)
    y = y.values.reshape(-1)
    stacked_y=np.hstack(y)
    print(stacked_y.shape)# still 25968,...
    arrSensors=np.array([sensor0,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8,sensor9,sensor10,sensor11,sensor12])
    trainingData = np.column_stack((sensor2,sensor4))
    print(trainingData.shape)#    trainingData = np.column_stack((chosenSensors[0],chosenSensors[1],chosenSensors[2]))
    print("BREAK")
    listOfIndices=[]
    bestInd=0
    bestResult=0
    chosenSensors=[]
    for i in range(0):
        #print(i)
        for s in range(13):
            if s not in listOfIndices:
                aList=[]
                for n in range(i):
                    aList.append(chosenSensors[n])
                aList.append(arrSensors[s])
                aList=np.array(aList)
                #print(aList.shape)
                aList=aList.reshape((25968,60*aList.shape[0]))
                _model = keras.Sequential()
                #maybe test the best 3 sensors on their own and then add them together
                #or do it like before, find best sensor on its own, find another sensor that works best with that sensor, etc
                #print(aList.shape[1])
                _model.add(layers.Dense(100, activation='relu', input_dim=aList.shape[1]))
                _model.add(layers.Dense(50, activation='relu'))
                _model.add(layers.Dense(1, activation='sigmoid'))
                _model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

                #print("Sensor")
                #print(s)
                #print(aList.shape)# np.column_stack((chosenSensors[0],chosenSensors[1],chosenSensors[2]))
                history = _model.fit(aList, y, epochs=120, batch_size=40,verbose=0)
                tLoss,tAcc=_model.evaluate(aList, y)
                #print(tAcc)
                if tAcc>bestResult:
                    bestInd=s
                    bestResult=tAcc
        print("BEST SENSOR")
        print(bestInd)
        print(i)
        listOfIndices.append(bestInd)
        chosenSensors.append(arrSensors[bestInd])
    chosenSensors=np.array(chosenSensors)
    print("DONE?")
    print(chosenSensors.shape)
    chosenSensors=np.column_stack((arrSensors[4],arrSensors[8],arrSensors[0]))
    #best ones r 4, 8, 0?
    print(chosenSensors.shape)
    model = keras.Sequential()
    #maybe test the best 3 sensors on their own and then add them together
    #or do it like before, find best sensor on its own, find another sensor that works best with that sensor, etc

    model.add(layers.Dense(100, activation='relu', input_dim=120))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    trainingData =np.column_stack((sensor2,sensor4))#    testingData = np.column_stack((sensor2,sensor4,sensor8))
    print("START TRAINING")
    print(trainingData.shape)
    print(y.shape)
    history = model.fit(trainingData, y, epochs=500, batch_size=256, verbose=0)


    print("SHAPING TEST")
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

    testingData =np.column_stack((sensor2,sensor4))
                                          #0,4,8 best combination but lower acc
                                          #kaggle stuff
                                          #hyperparams are 10 epoch and 40 bsize
                                          #0,4,8
                                          #Score: 0.56197 Private score: 0.55572
                                          #2,4
                                          #Score: 0.69025 Private score: 0.68752
                                          #4,8
                                          #Score: 0.60613 Private score: 0.5829
                                          #4,0
                                          #Score: 0.58367 Private score: 0.56447
                                          #8,0
                                          #Score: 0.51466 Private score: 0.53205
                                          #2,0
                                          #Score: 0.62951 Private score: 0.6259
                                          #2,8
                                          #Score: 0.63348 Private score: 0.65946
                                          #2,4,8
                                          #Score: 0.6366 Private score: 0.63642
                                          #2,4,0
                                          #Score: 0.63197 Private score: 0.61856
                                          #2,8,0
                                          #Score: 0.57393 Private score: 0.58974


                                          #now hparams as 500 epoch and 256 bsize
                                          #2,8,4
                                          #Score: 0.63376 Private score: 0.6133
                                          #2,4
                                          #Score: 0.67917 Private score: 0.67243
                                          #0,4,8
                                          #Score: 0.54794 Private score: 0.54919
                                          #2, 8
                                          #Score: 0.60198 Private score: 0.60779
                                          #2,0
                                          #Score: 0.61813 Private score: 0.60945
                                          #2,4,0
                                          #Score: 0.60749 Private score: 0.6051
                                          #2,4
                                          #, epoch 400, 64 bsize
                                          #Score: 0.66934 Private score: 0.67351
                                          #epoch 500, 256
                                          #Score: 0.66821 Private score: 0.66988
                                          #epoch: 250, 32 bsize
                                          #Score: 0.68636 Private score: 0.67026
                                          #epoch: 350, 32 bsize
                                          #Score: 0.67596 Private score: 0.66581
                                          #epoch: 350, 64 bsize
                                          #Score: 0.6849 Private score: 0.67459
                                          #epoch: 200, 64 bsize
                                          #Score: 0.71239 Private score: 0.70596
                                          #epoch: 100, 64 bsize
                                          #Score: 0.69766 Private score: 0.6893
                                          #epoch: 200, 128 bsize
                                          #Score: 0.68253 Private score: 0.6748
                                          #epoch: 200, 64 bsize
                                          #Score: 0.69142 Private score: 0.69124
                                          #epoch: 500, 256 bsize
                                          #Score: 0.6969 Private score: 0.68511


    print("Test Stuff")
    test_loss, test_acc = model.evaluate(trainingData, y)
    print("Done with eval")
    y_test = model.predict(testingData)
    print(len(y_test))
    print("Done w/ pred")
    y_test= y_test.ravel()
    print("Ravel")
    y_test = (y_test >= 0.5).astype(int)

    output = pd.DataFrame({'sequence': range(25968, 25968+len(y_test)), 'state': y_test})
    print(len(y_test))#why is size 25968

    output.to_csv('output.csv', index=False)
    print('Test accuracy:', test_acc)
    #Evaluation Exception: Submission must have 12218 rows
main()