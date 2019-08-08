import csv
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Activation, Lambda, Cropping2D

# from PIL import Image

from scipy import ndimage

# adjusting steering measurements
correction = 0.2

images = []
steerings = []

# for csvfile in ['./driving_log1.csv', './driving_log.csv']:
for csv_file in ['./driving_log.csv']:
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # jump header
        next(reader)
        for row in reader:
            center_path = row[0]
            left_path = row[1]
            right_path = row[2]
            
#             center_imgname = center_path.split('/')[-1]
#             left_imgname = left_path.split('/')[-1]
#             right_imgname = right_path.split('/')[-1]
            
            steering_center = float(row[3])
            steering_left = float(row[3])+ correction
            steering_right = float(row[3]) - correction
            fl_steering_center = -steering_center
            fl_steering_left = -steering_left
            fl_steering_right = -steering_right
            
            img_center = ndimage.imread(center_path)
            img_left = ndimage.imread(left_path)
            img_right = ndimage.imread(right_path)
            fl_img_center = np.fliplr(img_center)
            fl_img_left = np.fliplr(img_left)
            fl_img_right = np.fliplr(img_right)
            
#             plt.imsave('test.jpeg', img_center)
            images.append(img_center)
            images.append(img_left)
            images.append(img_right)
            images.append(fl_img_center)
            images.append(fl_img_left)
            images.append(fl_img_right)
          
            steerings.append(steering_center)
            steerings.append(steering_left)
            steerings.append(steering_right)
            steerings.append(fl_steering_center)
            steerings.append(fl_img_left)
            steerings.append(fl_img_right)
# print(np.array(images).shape)
# print(len(images))

X_train = np.array(images)
y_train = np.array(steerings)

print(X_train.shape)
print(y_train[0])

# model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=( (70, 20), (0,0) )))
model.add( Conv2D( 24, (5, 5)) ) 
# model.add(Activation('relu'))
model.add( Conv2D( 36, (5, 5)) ) 
# model.add(Activation('relu'))
model.add( Conv2D( 48, (5, 5)) ) 
# model.add(Activation('relu'))
model.add( Conv2D( 64, (3, 3)) ) 
# model.add(Activation('relu'))
model.add( Conv2D( 64, (3, 3)) ) 
# model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 
model.add(Dense(1)) # unit... output dimension

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3,shuffle=True, epochs = 2)

model.save('model.h5')
