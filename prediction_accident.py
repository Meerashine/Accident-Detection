# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 22:22:11 2020

@author: Meerashine Joe
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from keras.models import Model
import tensorflow as tf
from keras.models import load_model

model = load_model('my model.h5')

import os, os.path

train_categories = []
train_samples = []
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train"):
    train_categories.append(i)

# models.load_weights("finalmodel.hdf5")
img = Image.open(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test\accident\images_011.jpg")
original_img = np.array(img, dtype=np.uint8)
plt.imshow(original_img)

if img.size[0] > img.size[1]:
    scale = 100 / img.size[1]
    new_h = int(img.size[1] * scale)
    new_w = int(img.size[0] * scale)
    new_size = (new_w, new_h)
else:
    scale = 100 / img.size[0]
    new_h = int(img.size[1] * scale)
    new_w = int(img.size[0] * scale)
    new_size = (new_w, new_h)

resized = img.resize(new_size)
resized_img = np.array(resized, dtype=np.uint8)
plt.imshow(resized_img)
#plt.show()


left = 0
right = left + 100
up = 0
down = up + 100

cropped = resized.crop((left, up, right, down))
cropped_img = np.array(cropped, dtype=np.uint8)
#plt.imshow(cropped_img)
#plt.show()

cropped_img = cropped_img / 255.0
print(cropped_img.shape)


cropped_img =tf.image.resize(cropped_img, [255,255])
print(cropped_img.shape)
X = np.reshape(cropped_img, newshape=(-1, 255,255,3))
#print(X)
#print(X.shape)
prediction_multi = model.predict(x=X)
store = np.argmax(prediction_multi)
print(np.argmax(prediction_multi))

print("Predicted image is : ", train_categories[store])
plt.show()