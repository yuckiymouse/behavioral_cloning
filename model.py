import csv
import cv2
import numpy as np
import os

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Activation, Lambda, Dropout, MaxPooling2D, Cropping2D

lines = []
for csvfile in ['./driving_log1.csv', './driving_log.csv']:
# for csvfile in ['./driving_log.csv']:
    with open(csvfile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
# delete csv header
lines = lines[1:]

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        #print(measurement)
        measurements.append(measurement)
    
aurgumented_images = []
aurgumented_measurements = []

for image, measurement in zip(images, measurements):
    aurgumented_images.append(image)
    aurgumented_measurements.append(measurement)
    aurgumented_images.append(cv2.flip(image, 1)) # flipping around y-axis
    aurgumented_measurements.append(measurement * (-1.0))

X_train = np.array(aurgumented_images)
y_train = np.array(aurgumented_measurements)

# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates
#         sklearn.utils.shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
            
#             images = []
#             measurements = []
#             for batch_sample in batch_samples:
#                 for i in range(3):
#                     source_path = batch_sample[i]
#                     filename = source_path.split('/')[-1]
#                     current_path = '../data/IMG/' + filename
#                     image = cv2.imread(current_path)
#                     images.append(image)
#                     measurement = float(line[3])
#                     measurements.append(measurement)

#             aurgumented_images = []
#             aurgumented_measurements = []

#             for image, measurement in zip(images, measurements):
#                 aurgumented_images.append(image)
#                 aurgumented_measurements.append(measurement)
#                 aurgumented_images.append(cv2.flip(image, 1)) # flipping around y-axis
#                 aurgumented_measurements.append(measurement * (-1.0))

#             X_train = np.array(images)
#             y_train = np.array(measurement)
#             yield sklearn.utils.shuffle(X_train, y_train)
############# end of def
            

# print(train_generator)
# # model
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
model.fit(X_train, y_train, validation_split=0.3,shuffle=True, nb_epoch = 2)

model.save('model.h5')

