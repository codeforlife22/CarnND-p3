import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input, Activation, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D, Conv2D, Dropout
from keras.layers import Cropping2D
from random import shuffle
from keras.callbacks import ModelCheckpoint


images = []
batch_size = 32
base_dir = 'data_copy/'
img_dir = base_dir + 'IMG/'

def add_images_from_csv(images, csv_path):
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0].strip() == 'center':
                continue ## skip .csv header
            images.append(line)
    return images

images = add_images_from_csv(images, base_dir + 'driving_log_udacity.csv')
images = add_images_from_csv(images, base_dir + 'driving_log_recovery.csv')
images = add_images_from_csv(images, base_dir + 'driving_log.csv')
train_images, validation_images = train_test_split(images, test_size=0.2)
print('Total number of images are %d'%len(images))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = img_dir + batch_sample[0].split('/')[-1]
                
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                
                #images.append(center_image)
                #angles.append(center_angle)
                
                left_name = img_dir + batch_sample[1].split('/')[-1]
                right_name = img_dir + batch_sample[2].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                left_angle = center_angle + 0.2
                right_angle = center_angle - 0.2
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
               
                
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)
            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    
train_generator = generator(train_images, batch_size) 
validation_generator = generator(validation_images, batch_size)

model = Sequential()

#cropping image
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))

#nomarlize data

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65, 320, 3)))


##DNN Architecture, all the parameters are adopted from the lecture ####

model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())

## try to add another layer of convnet, but leads to worse performance
'''
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
'''

model.add(Flatten())

model.add(Dense(120))

model.add(Dense(84))

model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')

#save the best model trained so far
filepath = 'tmp/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history_object = model.fit_generator(train_generator, steps_per_epoch =
	len(train_images) / batch_size, validation_data = 
	validation_generator,
	validation_steps = len(validation_images) / batch_size, 
	epochs=10, verbose=1, callbacks=[model_checkpoint])

model.save('model.h5')
