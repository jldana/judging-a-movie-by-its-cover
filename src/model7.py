import os
from model_aide import *
import numpy as np
np.random.seed(1337)  # for reproducibility


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# batch_size = 50 # number of training samples used at a time to update the weights
# nb_classes = 10  # number of output possibilites: [0 - 9] (don't change)
# nb_epoch = 3    # number of passes through the entire train dataset before weights "final"

# input image dimensions

# number of convolutional filters to use
nb_filters = 4
# size of pooling area for max pooling
pool_size = (2, 2) # decreases image size, and helps to avoid overfitting
# convolution kernel size
kernel_size = (5, 5) # slides over image to learn features

img_width, img_height = 224, 224

train_data_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/data/train'
validation_data_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/data/validation'
predict_data_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/unknown'
nb_train_samples = 2000
nb_validation_samples = 600
epochs = 50
batch_size = 50
num_classes = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model = Sequential()
'''
model = VGG16(weights=None, include_top=True, classes=16)

img_path = '../predict/unknown/rambo_iii.jpg'
img = image.load_img(img_path, target_size=(224, 224))
im2 = image.img_to_array(img)
#img2 = im2.reshape(1, 206, 206, 3)
x = np.expand_dims(im2, axis=0)
print('expanded dims:{}'.format(x))
x = preprocess_input(x)
'''





#features = model.predict(x)

model.add(Conv2D(32,  kernel_size = kernel_size, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))
#
#model.add(VGG16(include_top=False, weights='imagenet', input_tensor=model, input_shape=input_shape, pooling=None, classes=1000))

model.add(Conv2D(128,  kernel_size = kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Conv2D(256, kernel_size = kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(4, 4)))

model.add(Conv2D(64, kernel_size = (7, 7)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(4, 4)))

# model.add(Conv2D(32, kernel_size = (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size =(4, 4)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())

#impred = io.imread('../predict/unknown/sarfarosh.jpg')
#img = image.load_img(img_path, target_size=(206, 206))



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')



model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save_weights('first_go.h5')



# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
