# Importing Libraries
import serial
import time
import struct
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler

#instantiate arduino object
arduino = serial.Serial(port='/dev/cu.usbmodem14101', baudrate=115200)
#Load Training set
xtrain_tacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt',header = None, delim_whitespace=True))
ytrain_tacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt',header = None, delim_whitespace=True))
ztrain_tacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt',header = None, delim_whitespace=True))
xtrain_bacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt',header = None, delim_whitespace=True))
ytrain_bacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt',header = None, delim_whitespace=True))
ztrain_bacc = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt',header = None, delim_whitespace=True))
xtrain_gyro = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt',header = None, delim_whitespace=True))
ytrain_gyro = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt',header = None, delim_whitespace=True))
ztrain_gyro = np.array(pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt',header = None, delim_whitespace=True))

y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt',header = None, delim_whitespace=True)


#Load test data

xtest_tacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt',header = None, delim_whitespace=True))
ytest_tacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt',header = None, delim_whitespace=True))
ztest_tacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt',header = None, delim_whitespace=True))
xtest_bacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt',header = None, delim_whitespace=True))
ytest_bacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt',header = None, delim_whitespace=True))
ztest_bacc = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt',header = None, delim_whitespace=True))
xtest_gyro = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt',header = None, delim_whitespace=True))
ytest_gyro = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt',header = None, delim_whitespace=True))

ztest_gyro = np.array(pd.read_csv('UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt',header = None, delim_whitespace=True))

#Test Labels
y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt',header = None, delim_whitespace=True)

def scaling(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train.reshape(-1,1))
    trainN = scaler.transform(train.reshape(-1,1)).reshape(train.shape)
    testN = scaler.transform(test.reshape(-1,1)).reshape(test.shape)
    return trainN, testN

#Scale signals
xtrain_taccN, xtest_taccN = scaling(xtrain_tacc, xtest_tacc)
xtrain_baccN, xtest_baccN = scaling(xtrain_bacc, xtest_bacc)
xtrain_gyroN, xtest_gyroN = scaling(xtrain_gyro, xtest_gyro)
ytrain_taccN, ytest_taccN = scaling(ytrain_tacc, ytest_tacc)
ytrain_baccN, ytest_baccN = scaling(ytrain_bacc, ytest_bacc)
ytrain_gyroN, ytest_gyroN = scaling(ytrain_gyro, ytest_gyro)
ztrain_taccN, ztest_taccN = scaling(ztrain_tacc, ztest_tacc)
ztrain_baccN, ztest_baccN = scaling(ztrain_bacc, ztest_bacc)
ztrain_gyroN, ztest_gyroN = scaling(ztrain_gyro, ztest_gyro)


#Combine 9 channels together 
x_train = [xtrain_taccN, ytrain_taccN, ztrain_taccN,
            xtrain_baccN, ytrain_baccN, ztrain_baccN,
           xtrain_gyroN, ytrain_gyroN, ztrain_gyroN]
x_test = [xtest_taccN, ytest_taccN, ztest_taccN,
          xtest_baccN, ytest_baccN, ztest_baccN,
          xtest_gyroN, ytest_gyroN, ztest_gyroN]

x_train = np.array(np.dstack(x_train),dtype=np.float32)
x_test = np.array(np.dstack(x_test),dtype = np.float32)



#make the label's index zero
y_train = y_train-1
y_test = y_test-1

x_test = [xtest_accN, ytest_accN, ztest_accN,
          xtest_baccN, ytest_baccN, ztest_baccN,
          xtest_gyroN, ytest_gyroN, ztest_gyroN]
x_test = np.array(np.dstack(x_test),dtype = np.float32)

infTime = 0
trueCount=0
confusedCount=0
totalCount= 100 #unmber of data entries to be evaluated
Activities = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
              'Sitting', 'Standing', 'Laying']

def plot_bar(currCount, totCount,stepTime,acc):
    bar = '\r  %2d%% [%s%s] (%d/%d) %.2f s/step real-time acc:%.2f'
    blockNum = 5
    blockSize = int(totCount/blockNum)
    blockDone = int(currCount/blockSize)
    a = '■'* blockDone
    b = '□'*(blockNum-blockDone)
    c = (currCount/totCount)*100
    print(bar % (c,a,b,currCount,totCount,stepTime,acc), end = ' ')


#for number in range (0,2947)
for i in range (0,totalCount):
    timestamp=time.time()
    for j in range (0,128):
        for k in range (0, 9):
            value = x_test[i][j][k]
            bin = struct.pack('f',value)
            arduino.write(bin)
            time.sleep(0.001)
    while arduino.inWaiting() == 0 :#keep reading byte untile empty queue
        continue

    a=int(arduino.readline().strip())#read inference result
    infTime = infTime + int(arduino.readline().strip())#read inference latency
    if a == y_test[i] :
        trueCount=trueCount+1
    acc = trueCount/(i+1)
    timestamp=time.time()-timestamp
    plot_bar(i+1,totalCount,timestamp,acc)

print(" ")
ave_acc=(trueCount/totalCount)*100
print("Accuracy is: %.2f" % ave_acc)
ave_time= infTime/totalCount
print("Average latency is: %.2f" % ave_time)



    

               

                      










 