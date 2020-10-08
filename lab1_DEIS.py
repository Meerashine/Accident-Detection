# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:47:39 2020

@author: Meerashine Joe
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os, os.path
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,LSTM,Reshape,GlobalAveragePooling1D,InputLayer
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential
import os, os.path
import math
import tensorflow
#import cv2
from tensorflow.keras.callbacks import TensorBoard
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



# At first we import the data that is stored and we are accessing all the 
#categories and subcategories with in the directory from both testinga and 
#training
train_categories = []
train_samples = []
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train"):
    train_categories.append(i)
    #print(type(i))
    train_samples.append(str(len(os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train" )))+ str(i))
    #print(train_samples)

test_categories = []
test_samples = []
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test"):
    test_categories.append(i)
    #test_samples.append(str(len(os.listdir(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Test"))) + str(i))

#print(type(train_samples))
print("No. of Training Samples:", len(train_samples))
print("No. of Training Categories:", len(train_categories))
print("we are good till  now..")
#print("No. of Test Samples:", len(test_samples))
#print("No. of Testing Categories:", len(test_categories))


#import all the images in to training and testing from all the categories with indices.
train = []
test = []

print("here I goes into the crap...")
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train"):
    one_hot = np.zeros(shape=[len(train_categories)])
    #print("length",len(train_categories))
    actual_index = train_categories.index(i)
    one_hot[actual_index] = 1
    #print(one_hot)
    for files in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train" + "\\" + i):
        #print(files)
        #for images in os.list(os.path.join(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Training" +list(i), files))):
        #img_array = mpimg.imread(os.path.join(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Training"  + "\\" , files))
        img_array = mpimg.imread(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train" + "\\" + i +"\\" + files)
        train.append([img_array, one_hot])
        #print("Train Category Status: {}/{}".format(actual_index+1, len(train_categories)))
        #print(train)
print("we are free from train")
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test"):
    one_hot = np.zeros(shape=[len(test_categories)])
    actual_index = test_categories.index(i)
    one_hot[actual_index] = 1
    for files in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test" + "\\" +i):
        img_array = mpimg.imread(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test" + "\\" + i + "\\" + files)
        test.append([img_array, one_hot])
        #print("test Category Status: {}/{}".format(actual_index+1, len(test_categories)))

print("fuck I cam out successfully...")      
train_x =[]
train_y= []
test_x =[]
test_y =[]

for i in range(len(train)):
     train_x.append(train[i][0])
     train_y.append(train[i][1])

for i in range(len(test)):
    test_x.append(test[i][0])
    test_y.append(test[i][1])

#generating vaidation set from the test set and split is used to split the data.
#we take here 20 percentage.
split = np.random.choice(len(train), size=math.floor(len(train)*0.2))
print(split)


validation_x =[]
validation_y =[]

print("goes again...")
for i in range(int(len(split))):
    validation_x.append(test[i][0])
    validation_y.append(test[i][1])
 
final_train_x =np.asarray(train_x)

final_train_y =np.asarray(train_y)
#print(final_train_y.shape)
final_test_x =np.asarray(test_x)
final_test_y =np.asarray(test_y)
final_validation_x =np.asarray(validation_x)
final_validation_y =np.asarray(validation_y)


print("fasten your belts")
for i in range(len(final_train_x)):
    final_train_x[i] = (final_train_x[i] /255)
#print(final_train_x.shape)
for i in range(len(final_test_x)):
    final_test_x[i] = (final_test_x[i] /255)
    
for i in range(len(final_validation_x)):
    final_validation_x[i] = (final_validation_x[i] /255)
#print(final_validation_x[7].shape)

print("check ends")


#making all images of same size
for i in range(len(final_train_x)):
    final_train_x[i] =tf.image.resize(final_train_x[i], [255,255])

#resizing validation
for i in range(len(final_validation_x)):
    final_validation_x[i] =tf.image.resize(final_validation_x[i], [255,255])
    
    
#resizing test
for i in range(len(final_test_x)):
    final_test_x[i] =tf.image.resize(final_test_x[i], [255,255])
    
    
new_x = np.empty((len(final_train_x),255,255,3))
for i in range(len(final_train_x)):
    new_x[i] = final_train_x[i]
  

new_x = new_x.reshape(-1, 255, 255,3) 


#validation set
new_val_x = np.empty((len(final_validation_x),255,255,3))
for i in range(len(final_validation_x)):
    new_val_x[i] = final_validation_x[i]
    #print(new_val_x[i].shape)
    

new_val_x = new_val_x.reshape(-1, 255, 255,3) 
#for i in range(len(new_val_x)):
    #print(new_val_x[i].shape)
    
#test data
new_test_x = np.empty((len(final_test_x),255,255,3))
for i in range(len(final_test_x)):
    new_test_x[i] = final_test_x[i]
    #print(new_test_x[i].shape)
    

new_test_x = new_test_x.reshape(-1, 255, 255,3) 
#for i in range(len(new_test_x)):
    #print(new_test_x[i].shape)
#@print(new_x.shape)
#print(final_train_y.shape)
#print(new_val_x.shape)
#print(final_validation_y.shape)


print("Resizing done")
cnn3 = Sequential()
cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(255,255,3)))
cnn3.add(MaxPooling2D((2, 2)))
cnn3.add(Dropout(0.25))

cnn3.add(Flatten())

cnn3.add(Dense(64, activation='relu'))
cnn3.add(Dropout(0.3))
cnn3.add(Dense(3, activation='softmax'))

cnn3.compile(loss="categorical_crossentropy",
              optimizer="Adam",
              metrics=['accuracy'])
#model1.summary()

print("Recognising------------------")
cnn3.fit(x = new_x,y = final_train_y,batch_size=32,validation_data=(new_val_x,final_validation_y),epochs=10)    
cnn3.save("my model.h5")

prediction = cnn3.predict(new_test_x)

pred_max=[]
for i in prediction:
    pred_max.append(np.argmax(i))
#print(pred_max)

test_y_max=[]
for i in test_y:
    test_y_max.append(np.argmax(i))
#print(test_y_max)



accuracy=0

print("Accuracy score without PCA",accuracy_score(test_y_max,pred_max))


