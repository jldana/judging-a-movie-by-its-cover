import os
import numpy as np
np.random.seed(1337)  # for reproducibility


import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping
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
nb_filters = 8
# size of pooling area for max pooling
pool_size = (2, 2) # decreases image size, and helps to avoid overfitting
# convolution kernel size
kernel_size = (3, 3) # slides over image to learn features

img_width, img_height = 130, 130

train_data_dir = '../alternate_data/data/train'
validation_data_dir = '../alternate_data/data/validation'
nb_train_samples = 4872
nb_validation_samples = 1612
epochs = 500
batch_size = 15
num_classes = 8

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
          metrics=['accuracy'])
'''
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#
#
# #features = model.predict(x)
#
# model.add(Conv2D(64,  kernel_size = kernel_size, input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size =(2, 2)))
# #
# #model.add(VGG16(include_top=False, weights='imagenet', input_tensor=model, input_shape=input_shape, pooling=None, classes=1000))
#
# model.add(Conv2D(128,  kernel_size = kernel_size))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = pool_size))
#
# model.add(Conv2D(512, kernel_size = kernel_size))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size =pool_size))
#
# model.add(Conv2D(256, kernel_size = kernel_size))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size =(2, 2)))
#
# model.add(Conv2D(64, kernel_size = kernel_size))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size =pool_size))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.00001, decay=0.000001)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())
#model.load_weights('first_go.h5')

#impred = io.imread('../predict/unknown/sarfarosh.jpg')
#img = image.load_img(img_path, target_size=(206, 206))



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.0,
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


csvlog = CSVLogger('../log/model_log.csv', separator=',', append=True)
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0, patience=30, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='../checkpoints/the_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks = [csvlog, checkpointer, earlystop])


mod_plot = plot_model(model, to_file='model3.png')
yaml_str = model.to_yaml()
mod_file = 'mod3.yaml'
mod_file2 = 'model_11.json'
config_file = 'config_mod.h5'
with open(mod_file, 'w') as f:
    f.write(yaml_str)
print('Model Saved!')
model.save_weights('mod3_w_good.h5')
print('Weights Saved!')


#model11 = Model.from_config(config)


model.save('dont_stop.hdf5')
print('Model Saved!')
