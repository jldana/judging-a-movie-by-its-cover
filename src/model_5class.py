import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, PReLU
from keras.optimizers import SGD, Adam, Adamax
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io
from keras.preprocessing import image
from keras import regularizers

def model_builder(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(32, (3, 3)))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(Dense(128))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(Dense(64))
    model.add(PReLU(alpha_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())
    return model

def model_train(batch_size, img_width, img_height, epochs):
    nb_train_samples = 3699
    nb_validation_samples = 1113
    train_data_dir = '../alt_data_2/data/train'
    validation_data_dir = '../alt_data_2/data/validation'
    # this is the augmentation configuration I will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.0,
        zoom_range=0.2,
        horizontal_flip=False)

    # this is the augmentation configuration I will use for testing:
    #       only rescaling
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

    csvlog = CSVLogger('../log/model_log_5class.csv', separator=',', append=True)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0, patience=130, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath='../checkpoints/the_weights_5class.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks = [csvlog, checkpointer, earlystop])

    mod_plot = plot_model(model, to_file='model_5class.png')
    model.save_weights('mod3_w_good_5class.h5')
    print('Weights Saved!')
    model.save('final_5class.hdf5')
    print('Model Saved!')
pass

def predict_class():
    model = load_model('final_5class.hdf5')
    imag = input('File to predict on: ')
    img_path = '../predict/unknown/{}.jpg'.format(imag)
    img = image.load_img(img_path, target_size=(224, 224))
    im2 = image.img_to_array(img)
    im2 = np.expand_dims(im2, axis=0)
    predict = model.predict(x, batch_size = 32, verbose = 1)
    return predict

if __name__ == '__main__':
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (2, 2) # decreases image size, and helps to avoid overfitting
    # convolution kernel size
    kernel_size = (3, 3) # slides over image to learn features
    # image dims
    img_width, img_height = 130, 130
    # number of proccess runs
    epochs = 500
    #number of samples per run
    batch_size = 30
    #number of classes for final layer
    num_classes = 5

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = model_builder(input_shape, num_classes)
    model_train(batch_size, img_width, img_height, epochs)
