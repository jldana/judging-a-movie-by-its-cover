from keras.models import load_model
from keras import backend as K
from skimage.transform import resize
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def active_getter(model, layer, im_path):
    img = image.load_img(im_path, target_size=(130, 130))
    im_put = image.img_to_array(img)
    im_put = np.expand_dims(im_put, axis=0)
    activ1 = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    activations = activ1((im_put, False))
    return activations

def layer_select(layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    return layer_dict[layer_name]

def layer_view(model, layer_name, movie_name):
    im_path = '../predict/unknown/{}.jpg'.format(movie_name)
    layer = layer_select(layer_name)
    show = active_getter(model, layer, im_path)
    show = show[0][0]
    n = 5
    fig = plt.figure(figsize=(36, 36))
    for i in range(n):
        for j in range(n):
            idx = (n*i) + j
            ax = fig.add_subplot(n, n, idx+1)
            ax.imshow(show[:, :, idx])
            plt.axis('off')
    plt.savefig('{}_display.pdf'.format(layer_name))
    plt.clf()
    plt.close()
    pass

if __name__ == '__main__':

    model_path = '3_class_model.hdf5'
    model = load_model(model_path)
    movie_name = input('Which movie are we looking at? ')
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print(layer_dict.keys())
    layer_name = input('Which layer, from those listed above, would you like to view? ')
    if layer_name == 'all':
        for layr in layer_dict.keys():
            if layr == 'flatten_1':
                break
            else:
                layer_view(model, layr, movie_name)
    else:
        layer_view(model, layer_name, movie_name)
