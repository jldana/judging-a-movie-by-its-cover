import os
from model_aide import *
import numpy as np
np.random.seed(1337)  # for reproducibility


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
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
print('=======Now Running Model 9!=======')
# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
pool_size = (2, 2) # decreases image size, and helps to avoid overfitting
# convolution kernel size
kernel_size = (5, 5) # slides over image to learn features

img_width, img_height = 224, 224

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
predict_data_dir = '../predict/unknown'
nb_train_samples = 2000
nb_validation_samples = 600
epochs = 50
batch_size = 40
num_classes = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

VGG = VGG16(include_top=False, weights='imagenet', input_tensor=Input((224, 224, 3)))
for l in VGG.layers:
    l.trainable = False

x = Flatten(input_shape=VGG.output.shape)(VGG.output)
x = Dense(4096, activation='relu', name='ft_fc1')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(16, activation = 'softmax')(x)

model = Model(inputs=VGG.input, outputs=predictions)

#model = Sequential()
'''
model = VGG16(weights=None, include_top=True, classes=16)

img_path = '../predict/unknown/rambo_iii.jpg'
img = image.load_img(img_path, target_size=(224, 224))
im2 = image.img_to_array(img)
#img2 = im2.reshape(1, 206, 206, 3)
x = np.expand_dims(im2, axis=0)
print('expanded dims:{}'.format(x))
x = preprocess_input(x)

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])
'''





#features = model.predict(x)

# model.add(Conv2D(32,  kernel_size = kernel_size, input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size =(2, 2)))
# #
# #model.add(VGG16(include_top=False, weights='imagenet', input_tensor=model, input_shape=input_shape, pooling=None, classes=1000))
#
# model.add(Conv2D(128,  kernel_size = kernel_size))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = pool_size))
#
# model.add(Conv2D(256, kernel_size = kernel_size))
# model.add(Activation('tanh'))
# model.add(MaxPooling2D(pool_size =(4, 4)))
#
# model.add(Conv2D(64, kernel_size = kernel_size))
# model.add(Activation('sigmoid'))
# model.add(MaxPooling2D(pool_size =(4, 4)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())
#model.load_weights('first_go.h5')

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

'''
tully = model.predict(img2, verbose=0)
gross = model.predict_classes(img2, verbose=1)
y_classes = tully.argmax(axis=1)
print(tully)
print(y_classes)
'''

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#predict_generator = train_datagen.flow_from_directory(
    # predict_data_dir,
    # target_size=(img_width, img_height),
    # batch_size = batch_size,
    # class_mode='categorical')



model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50)


# model.save_weights('first_go.h5')



# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])



# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
# img = load_img('/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/movie_posters/day_at_the_races.jpg')  # this is a PIL image
# x = img_to_array(img)
#  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# df_mov = movie_db()
# base_class, names = base_class(df_mov)
# classes = class_handler(base_class)
# class_df = reframe(classes, names)
# uniques = class_df.genres.unique()
# cleaned_uni = [genre.lower().replace(' ', '_').replace('-', '_') for genre in uniques]
# cleaned_uni.remove('unknown')
# cleaned_uni.remove('fantasy')
# cleaned_uni.remove('mystery')
#
# print(cleaned_uni)
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# for genre in cleaned_uni:
#     sv_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/
#     i = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_to_dir='/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/src/prewiew/', save_prefix='movie', save_format='jpg'):
#         i += 1
#         if i > 20:
#             break  # otherwise the generator would loop indefinitely

#if __name__ == '__main__':








    #
    # path1 = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/movie_posters'
    # for i in os.scandir(path=path1):
    #     search_name = str(i).split(' ')[1].rstrip(".jpg'>").lstrip("'")
    #     print(search_name)
