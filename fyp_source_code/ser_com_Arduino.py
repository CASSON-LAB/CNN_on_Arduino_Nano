# Importing Libraries
import serial
import struct
import time
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler


arduino = serial.Serial(port='/dev/cu.usbmodem14101', baudrate=115200)

#Load Training Datasets
print("Loading and preprocessing datasets...")
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
y_test = np.array(pd.read_csv('UCI HAR Dataset/test/y_test.txt',header = None, delim_whitespace=True))

def scaling(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train.reshape(-1,1))
    scaled_train = scaler.transform(train.reshape(-1,1)).reshape(train.shape)
    scaled_test = scaler.transform(test.reshape(-1,1)).reshape(test.shape)
    return scaled_train, scaled_test
#Scale data into [-1,1] range
xtrain_taccS, xtest_taccS = scaling(xtrain_tacc, xtest_tacc)
xtrain_baccS, xtest_baccS = scaling(xtrain_bacc, xtest_bacc)
xtrain_gyroS, xtest_gyroS = scaling(xtrain_gyro, xtest_gyro)
ytrain_taccS, ytest_taccS = scaling(ytrain_tacc, ytest_tacc)
ytrain_baccS, ytest_baccS = scaling(ytrain_bacc, ytest_bacc)
ytrain_gyroS, ytest_gyroS = scaling(ytrain_gyro, ytest_gyro)
ztrain_taccS, ztest_taccS = scaling(ztrain_tacc, ztest_tacc)
ztrain_baccS, ztest_baccS = scaling(ztrain_bacc, ztest_bacc)
ztrain_gyroS, ztest_gyroS = scaling(ztrain_gyro, ztest_gyro)

#Combine 9 channels together 
x_train = [xtrain_taccS, ytrain_taccS, ztrain_taccS,
           xtrain_baccS, ytrain_baccS, ztrain_baccS,
           xtrain_gyroS, ytrain_gyroS, ztrain_gyroS]
x_train = np.array(np.dstack(x_train),dtype=np.float32)
x_test = [xtest_taccS, ytest_taccS, ztest_taccS,
          xtest_baccS, ytest_baccS, ztest_baccS,
          xtest_gyroS, ytest_gyroS, ztest_gyroS]
x_test = np.array(np.dstack(x_test),dtype = np.float32)
#make the label's index zero
y_train = y_train-1
y_test = y_test-1



infTime = 0 #store total inference time
latency=0 #store average latency
trueCount=0 #totao number of correct prediction

#prompt user to input an integer number
totalCount=int(input("How many test sample to run inferenece? Enter an integer greater than 10 : "))

Activities = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
              'Sitting', 'Standing', 'Laying']

def plot_bar(currCount, totCount,acc, latency):
    bar = '\r  %2d%% [%s%s] (%d/%d) real-time acc:%.2f and latency:%.2f ms'
    blockNum = 10
    blockSize = int(totCount/blockNum)
    blockDone = int(currCount/blockSize)
    a = '■'* blockDone
    b = '□'*(blockNum-blockDone)
    c = (currCount/totCount)*100
    print(bar % (c,a,b,currCount,totCount,acc,latency), end = ' ')


#Run inference over the test samples between 0 and the user-input number
for i in range (0,totalCount):
    for j in range (0,128):
        for k in range (0, 9):
            value = x_test[i][j][k]
            bin = struct.pack('f',value)
            arduino.write(bin)
            time.sleep(0.001)
    while arduino.inWaiting() == 0 :
        continue

    a=int(arduino.readline().strip())
    infTime = infTime + int(arduino.readline().strip())
    if a == y_test[i] :
        trueCount=trueCount+1

    acc = trueCount/(i+1)
    latency = infTime/(i+1)
    plot_bar(i+1,totalCount,acc,latency)

print(" ")
ave_acc=(trueCount/totalCount)*100
print("Accuracy is: %.2f" % ave_acc)
ave_time= infTime/totalCount
print("Average latency is: %.2f ms" % ave_time)
