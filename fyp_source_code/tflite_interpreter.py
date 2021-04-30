import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


##Load Training set
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

#TFlite Interpreter for Inference
modelname = input("model name?: ")
modelpath=modelname+'_model.tflite'
interpreter = tf.lite.Interpreter(model_path=modelpath)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(input_details)
x_test=x_test.reshape(2947,128,9,1)


#Run Interpreter in for loop for every test data entry 

accurate_count = 0

for i in range(0,2947):
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1,:,:])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predict_label = np.argmax(output_data) #predicted label from interpreter  
    #check if prediction is correct
    accurate_count += (predict_label == y_test.iloc[i][0]) 

#Overall accuracy for entire test 
accuracy = accurate_count * 1.0 / 2947 

print('TensorFlow Lite Float32 model accuracy = %.4f' % accuracy)

