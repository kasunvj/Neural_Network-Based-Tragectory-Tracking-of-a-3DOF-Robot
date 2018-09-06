import random
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard


linkLength = 2

Q1 = []
Q2 = []
Q3 = []
posX = []
posY = []
titaEnd = []
samples = 5000

def Xe (a,b,c):                 # return the X,Y,Tita for a given 2 joint angles
    return linkLength*math.cos(a)+linkLength*math.cos(a+b)+linkLength*math.cos(a+b+c)
def Ye (e,f,g):
    return linkLength*math.sin(e) + linkLength*math.sin(e + f) + linkLength*math.sin(e + f + g)
def tita (h,i,j):
    return math.degrees(h)+math.degrees(i)+math.degrees(j)

def build_model():              # NN Model
    model = keras.Sequential()
    #model.add(keras.layers.Dense(3))
    model.add(keras.layers.Dense(100, activation='tanh'))
    model.add(keras.layers.Dense(100, activation='tanh'))
    model.add(keras.layers.Dense(3, activation='linear'))
    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='mse', metrics=['accuracy'])  # mean squared error
    return model
def plot_history(history):      # Track the history
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label ='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='validation Lost')
    plt.legend()
    plt.ylim([0,10])

class printDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


file = open ("traing_data.csv","w")                             # Data Set Creation
for i in range (0,samples):
    q1= round(random.uniform(0,math.pi),2)
    q2= round(random.uniform(-math.pi,0),2)
    q3= round(random.uniform(-math.pi/2, math.pi/2), 2)

    Q1.append(q1)
    file.write(str(q1))
    file.write(",")

    Q2.append(q2)
    file.write(str(q2))
    file.write(",")

    Q3.append(q3)
    file.write(str(q3))
    file.write(",")

    X = Xe(q1,q2,q3)
    posX.append(X)
    file.write(str(round(X, 2)))
    file.write(",")

    Y =Ye(q1,q2,q3)
    posY.append(Y)
    file.write(str(round(Y, 2)))
    file.write(",")

    T = tita(q1,q2,q3)
    titaEnd.append(T)
    file.write(str(round(T, 2)))
    file.write("\n")

file.close()

for i in range(0,len(posX)):
    plt.plot([posX[i],posX[i]+0.2*math.cos(math.radians(titaEnd[i]))],[posY[i],posY[i]+0.2*math.sin(math.radians(titaEnd[i]))],'k-')

plt.scatter(posX,posY)                              #Plotting the data set
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Fig 1: data set of 1000 possible endeffector positions and orientations")
plt.show()


dataMat =np.c_[Q1,Q2,Q3,posX,posY,titaEnd]          # Augmenting to the data marix

for i in range (0,samples):                         # Removing duplicated end effeector positions
    check1 = dataMat[i,3]
    check2 = dataMat[i,4]
    check3 = dataMat[i,5]
    for j in range (0,samples):
        if i != j:
            #print(i,j,'checking..')
            if (dataMat[j,3] == check1 and dataMat[j,4] == check2 and dataMat[j,5] == check3):
                print(i,j,dataMat[j,3],dataMat[j,4],dataMat[j,5])

#MinMaxNorm = preprocessing.minmax_scale(dataMat,(-1,1))              #Preprocessing - Normalizing data between -1 and +1



#plt.scatter(MinMaxNorm[:,3],MinMaxNorm[:,4])
#plt.title("Preprocessed data")
#plt.show()

#print(MinMaxNorm)

data =dataMat[:,[3,4,5]] #X,Y,Tita
output =dataMat[:,[0,1,2]]  # Q1,Q2,Q3

train_input = data[0:int(0.7*samples),:]                             #Separate data set in to Train, Test And Validation
train_output = output[0:int(0.7*samples),:]

test_input = data[int(0.7*samples):int(0.85*samples),:]
test_output = output[int(0.7*samples):int(0.85*samples),:]

validate_input = data[int(0.85*samples):int(samples),:]
validate_output = output[int(0.85*samples):int(samples),:]

print("Train INPUT---------------------")
print(np.shape(train_input))
print(train_input)
print("OUTPUT--------------------")
print(np.shape(output))
print(output)

x_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_eva = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_scaler_eva = preprocessing.MinMaxScaler(feature_range=(-1,1))

dataX = x_scaler.fit_transform(train_input)
dataY = y_scaler.fit_transform(train_output)
dataX_test = x_scaler_test.fit_transform(test_input)
dataY_test = y_scaler_test.fit_transform(test_output)
dataX_eva = x_scaler_eva.fit_transform(validate_input)
dataY_eva = y_scaler_eva.fit_transform(validate_output)

NAME = "Trajectry Tracking"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))           # Create    callbacks for tensorboard visualizations

model =  build_model()                                              # Building the model


history = model.fit(dataX,dataY, nb_epoch=100, callbacks=[tensorboard])    #train the model

[loss,mae] =model.evaluate(dataX_test,dataY_test,verbose=0)        #evaluation

print("Testing set Mean Abs Error: ${:7.2f}".format(mae))


dataX_input = x_scaler.transform(validate_input)
test_prediction = model.predict(dataX_input) #predict
real_prediction =y_scaler.inverse_transform(test_prediction)

plt.scatter(validate_output[:,0],real_prediction[:,0],c='b')              # Plotting Actual angles( x: desired output(Joint angles used to genarate Xe,Ye nd Titae,y: output from prediction )
plt.scatter(validate_output[:,1],real_prediction[:,1],c='g')
plt.scatter(validate_output[:,2],real_prediction[:,2],c='r')
plt.xlabel('True Values angles in rad')
plt.ylabel('Predictions  angles in rad')
plt.title("Fig 3: True Value Vs Prediction")
plt.legend("If all predicted values equal to the desired(true) value, this will be lie on 45 degree line")
plt.show()
#_Tensor Board
#tensorboard --logdir=logs/
print("*********************************")
print(validate_input[100,0]," ",validate_input[100,1])
print(Xe(real_prediction[100,0],real_prediction[100,1],real_prediction[100,2])," ",Ye(real_prediction[100,0],real_prediction[100,1],real_prediction[100,2]) )
print("*********************************")

single_data_1 = np.array([[5, 2, 60]])
single_data = x_scaler.transform(single_data_1)
single_prediction = model.predict(single_data)
single_real_prediction = y_scaler.inverse_transform(single_prediction)

print(single_data_1[0,0]," ",single_data_1[0,1])
print(Xe(single_real_prediction[0,0],single_real_prediction[0,1],single_real_prediction[0,2])," ",Ye(single_real_prediction[0,0],single_real_prediction[0,1],single_real_prediction[0,2]) )
print("*********************************")

                                                                                #inputs to the network - End efector positions(X and Y) and orientation
                                                                                #Output - Joint angles (Q1,Q2,Q3)
Xc = 3
Yc = 2
r = 1
data_points =100

Input_Circle = np.zeros((data_points,3),float)
Output_Circle = np.zeros((data_points,3),float)
Single_input = np.zeros((1,3),float)

titaz = np.linspace(5,360,num =data_points)


tagectory =[]

for i in range (0,len(titaz)):
     Input_Circle[i][0]=Xc + r*math.cos(np.radians(titaz[i]))
     Input_Circle[i][1]=Yc + r*math.sin(np.radians(titaz[i]))
     Input_Circle[i][2]= 0



plt.scatter(Input_Circle[:,0],Input_Circle[:,1],c ='b')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Figure 3: Desired Tragectory Cordinates and  Predicted ")


inin = np.zeros((1,3),float)
inin[0][0]=Input_Circle[0,0]
inin[0][1]=Input_Circle[0,1]
inin[0][2]=Input_Circle[0,2]

# print(inin)

# inin = x_scaler.transform(np.array([[3,3,180]]))
# single_prediction = model.predict(inin)
# Real_Prediction = y_scaler.inverse_transform(single_prediction)
#
#
Predicted_cordinates =np.zeros((data_points,3),float)
print(np.shape(Input_Circle))

single_data_1 = np.array(Input_Circle)
single_data = x_scaler.transform(single_data_1)
single_prediction = model.predict(single_data)
single_real_prediction = y_scaler.inverse_transform(single_prediction)

print(single_data_1[0,0]," ",single_data_1[0,1])
print(Xe(single_real_prediction[0,0],single_real_prediction[0,1],single_real_prediction[0,2])," ",Ye(single_real_prediction[0,0],single_real_prediction[0,1],single_real_prediction[0,2]) )
print("*****************!***************")


for q in range(0,data_points):
    Predicted_cordinates[q][0]= Xe(single_real_prediction[q,0],single_real_prediction[q,1],single_real_prediction[q,2])
    Predicted_cordinates[q][1]= Ye(single_real_prediction[q,0],single_real_prediction[q,1],single_real_prediction[q,2])

plt.scatter(Predicted_cordinates[:,0],Predicted_cordinates[:,1],c='r')
plt.show()

#print(Xe(Real_Prediction[0,0],Real_Prediction[0,1],Real_Prediction[0,2])," ",Ye(Real_Prediction[0,0],Real_Prediction[0,1],Real_Prediction[0,2]))


































# for i in range (0,100):
#     Single_input[0][0] = Input_Circle[i,0]
#     Single_input[0][1] = Input_Circle[i, 1]
#     Single_input[0][2] = Input_Circle[i, 2]
#
#     print("Single Input")
#     print(Single_input)
#
#     Single_input = x_scaler_eva.transform(Single_input)
#     Single_Cor_prediction = model.predict(Single_input)
#     Real_prediction = y_scaler_eva.inverse_transform(Single_Cor_prediction)
#
#
#
#     print("Prediction result")
#     Xe_hat= Xe(Real_prediction[0,0],Real_prediction[0,1],Real_prediction[0,2])
#     Ye_hat= Ye(Real_prediction[0,0],Real_prediction[0,1],Real_prediction[0,2])
#     Tita_hat = tita(Real_prediction[0,0],Real_prediction[0,1],Real_prediction[0,2])
#     print(Xe_hat," ",Ye_hat," ",Tita_hat)
#     print("--------------*----------------")
#     plt.scatter(Xe_hat,Ye_hat,c='r')
#
# plt.show()
