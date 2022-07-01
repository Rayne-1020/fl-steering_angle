from keras.applications import vgg16
from keras.models import Model, Sequential
#from keras.applications.inception_v3 import inceptionV3
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
#from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras import backend as K 

#cg23
def build_cnn1(image_size = None, weights_path = None):
    image_size = (320,480) or (128, 128)
    if K.image_data_format() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3,)
    
    img_input = Input(input_shape)

    x = Convolution2D(32, 3, 3, activation = 'relu', padding = 'same')(img_input)
    x = MaxPooling2D((2, 2), strides = (2, 2))(x)
    x = Dropout(0.5)(x)#0.25

    x = Convolution2D(64, 3, 3, activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2 ,2), strides = (2, 2))(x)
    x = Dropout(0.5)(x)#0.25

    x = Convolution2D(128, 3, 3, activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2 ,2), strides = (2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(1024, activation = 'relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1)(y)

    model = Model(img_input, y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')

    if weights_path:
        model.load_weights(weights_path)
    
    return model

#dolaameng
def build_cnn2(image_size = None, weights_path = None):
    image_size = (320,480) or (60, 80)
    if K.image_data_format() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3,)
    img_input = Input(input_shape)
    #print(img_input)

    x = Convolution2D(64, 3, 3, activation = 'relu', padding = 'same')(img_input)
    x = Dropout(0.5)(x)
    x = Convolution2D(64, 3, 3, activation = 'relu', padding = 'same')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2 ,2), strides = (2, 2))(x)

    x = Convolution2D(128, 3, 3, activation = 'relu', padding = 'same')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2 ,2), strides = (2, 2))(x)


    y = Flatten()(x)
    y = Dense(1024, activation = 'relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1024, activation = 'relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1)(y)

    model = Model(img_input, y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')

    if weights_path:
        model.load_weights(weights_path)
    
    return model


