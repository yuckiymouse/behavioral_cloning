import csv
import cv2
import numpy as np
import os
import sys

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Activation, Lambda, Cropping2D

images = []
# images_right = []
# images_left = []
steerings = []
# steerings_right = []
# steerings_left = []

for csvfile in ['./driving_log1.csv', './driving_log.csv']:
    with open(csvfile, 'r') as f:
            reader = csv.reader(f)
            # jump header
            next(reader)
            for row in reader:
                steering_center = float(row[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path = "./" # fill in the path to your training IMG directory
                img_center = cv2.imread(path + row[0])
                # delete spaces 
                row[1] = row[1].replace(' ', '')
                row[2] = row[2].replace(' ', '')
                img_left = cv2.imread(path + row[1])
                img_right = cv2.imread(path + row[2])

                # add images and angles to data set
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                
                steerings.append(steering_center)
                steerings.append(steering_left)
                steerings.append(steering_right)

                
# print(np.array(steerings).shape)

print(np.array(images).shape)
print(len(images))

# sys.exit()
                
#                 print(images_left)
# sys.exit()                
# for csvfile in ['./driving_log1.csv', './driving_log.csv']:
# # for csvfile in ['./driving_log.csv']:
#     with open(csvfile) as csvfile:
#         reader = csv.reader(csvfile)
#         for line in reader:
#             lines.append(line)
        
# # delete csv header
# lines = lines[1:]

# images = []
# measurements = []
# for line in lines:
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('/')[-1]
#         current_path = './IMG/' + filename
#         image = cv2.imread(current_path)
#         images.append(image)
#         measurement = float(line[3])
#         #print(measurement)
#         measurements.append(measurement)
    
aurgumented_images = []
aurgumented_steerings = []

for image, steering in zip(images, steerings):
#     print(image)
#     print(steering)
#     sys.exit()
    aurgumented_images.extend(image)
#     print(aurgumented_images)
    
#     sys.exit()
    aurgumented_images.append(cv2.flip(image, 1)) # flipping around y-axis
    
    aurgumented_steerings.extend(steering)
    aurgumented_steerings.append(steering * (-1.0))

# for measurement in zip(steerings_center, steerings_rightr, steerings_left):
#     aurgumented_measurements.append(measurement)
#     aurgumented_measurements.append(measurement * (-1.0))
print(np.array(aurgumented_images).shape)


X_train = np.array(aurgumented_images)
y_train = np.array(aurgumented_steerings)

X_train = X_train.reshape((-1,160,320,3))
print(X_train)
sys.exit()
# X_train = aurgumented_images
# y_train = aurgumented_steerings

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
model.fit(X_train, y_train, validation_split=0.3,shuffle=True, epochs = 2)

model.save('model.h5')

