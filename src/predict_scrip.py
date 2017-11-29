import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils, plot_model
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input





img_path = '../predict/unknown/5_children_and_it.jpg'
img = image.load_img(img_path, target_size=(224, 224))
im2 = image.img_to_array(img)
#img2 = im2.reshape(1, 206, 206, 3)
x = np.expand_dims(im2, axis=0)
#print('expanded dims:{}'.format(x))
x = preprocess_input(x)

model_path = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/src/weights/new_weights_1125.hdf5'
model = load_model(model_path)
predict = model.predict(x, batch_size = 32, verbose = 1)
