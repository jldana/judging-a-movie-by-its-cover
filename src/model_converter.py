from coremltools import converters


def keras_to_mlmodel(mod_path):
    coreml_model = converters.keras.convert(mod_path, input_names='image',
                                                image_input_names='image',
                                                class_labels=['Action/Thriller', 'Comedy', 'Drama'])
    coreml_model.save('3_class_DAC.mlmodel')
    pass

if __name__ == '__main__':
    mod_path = '3_class_model.hdf5'
    keras_to_mlmodel(mod_path)
