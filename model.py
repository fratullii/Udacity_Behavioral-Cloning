import os
import csv
import cv2
import numpy as np
import sklearn

from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from math import ceil
from random import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Dropout, Dense, Flatten

## MODEL ARCHITECTURE
model = Sequential()

# Preprocessing layers 
model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Convolutional layers
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu")) 

# Flatten
model.add(Flatten())

# Fully connected NN
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

## IMPORT DATA AND TRAIN

data_path = "/opt/carnd_p3/data/" # available only in GPU mode

samples = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
header = samples.pop(0)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)        
       
def generator(samples, batch_size=32, correction_factor=.2):
    # The actual batch_size is batch_size * 6
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                center_name = data_path + 'IMG/' + batch_sample[0].split('/')[-1]
                left_name = data_path + 'IMG/' + batch_sample[1].split('/')[-1]
                right_name = data_path + 'IMG/' + batch_sample[2].split('/')[-1]
                
                center_image = imread(center_name)
                left_image = imread(left_name)
                right_image = imread(right_name)
                
                # Steering angles correction for lateral cameras images
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction_factor
                right_angle = center_angle - correction_factor
                
                images.append(center_image)
                images.append(right_image)
                images.append(left_image)
                
                angles.append(center_angle)
                angles.append(right_angle)
                angles.append(left_angle)
                
                # Augment data by flipping it
                for i in range(batch_size):
                    images.append(cv2.flip(images[i], 1))
                    angles.append(-angles[i])
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Hyperparameters
batch_size = 32
n_epochs = 8

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch = ceil(len(train_samples)/batch_size),
            validation_data = validation_generator,
            validation_steps = ceil(len(validation_samples)/batch_size),
            epochs=n_epochs, verbose=1)

# save model
model.save('model.h5')