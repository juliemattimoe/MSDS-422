#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 01:49:02 2018

@author: juliemattimoe
"""

from keras.models import model_from_json
import cv2
import numpy as np
from PIL import Image

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

import os
os.path.isfile('1.jpg')

img = cv2.imread('1.jpg')
img = cv2.resize(img, (50,50))
print(img.shape)
img = img.reshape(1, 50, 50, 3)

print(img.shape)
#print(np.argmax(loaded_model.predict(img)))
print(loaded_model.predict(img))

image1 = Image.open('1.jpg')
image1

img = cv2.imread('11.jpg')
img = cv2.resize(img, (50,50))
print(img.shape)
img = img.reshape(1, 50, 50, 3)

print(img.shape)
#print(np.argmax(loaded_model.predict(img)))
print(loaded_model.predict(img))

image2 = Image.open('11.jpg')
image2