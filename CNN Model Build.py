#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 00:29:26 2018

@author: juliemattimoe
"""

import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

cat_im = Image.open('cat.0.jpg')
dog_im = Image.open('dog.0.jpg')

cat_im
cat_im.size
cat_array = np.array(cat_im.getdata())
cat_array.shape

dog_im
dog_im.size
dog_array = np.array(cat_im.getdata())
dog_array.shape

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_data',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 1600,
                         epochs = 10,
                         validation_steps = 400)

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.h5")
print("saved model..! ready to go.")
