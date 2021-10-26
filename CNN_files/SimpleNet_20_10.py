from CNNs import blocks
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import  BatchNormalization, Activation, Dropout
from keras.models import Model


def get_model(input_img, n_filters=8, kernel_size=3):
    c1 = Conv2D(n_filters, kernel_size, padding = 'same')(input_img)
    b1 = BatchNormalization()(c1)
    ac1 = Activation('relu')(b1)
    p1 = AveragePooling2D((2,2), 2)(ac1)

    c2 = Conv2D(n_filters*2, kernel_size, padding='same')(p1)
    b2 = BatchNormalization()(c2)
    ac2 = Activation('relu')(b2)
    p2 = AveragePooling2D((2, 2), 2)(ac2)

    c3 = Conv2D(n_filters*4, kernel_size, padding='same')(p2)
    b3 = BatchNormalization()(c3)
    ac3 = Activation('relu')(b3)
    p3 = AveragePooling2D((2, 2), 2)(ac3)

    c4 = Conv2DTranspose(n_filters * 4, kernel_size, strides = (2,2), padding='same')(p3)
    b4 = BatchNormalization()(c4)
    ac4 = Activation('relu')(b4)

    c5 = Conv2DTranspose(n_filters*2, kernel_size, strides = (2,2),padding='same')(ac4)
    b5 = BatchNormalization()(c5)
    ac5 = Activation('relu')(b5)

    c6 = Conv2DTranspose(n_filters * 2, kernel_size, strides=(2, 2), padding='same')(ac5)
    b6 = BatchNormalization()(c6)
    ac6 = Activation('relu')(b6)

    d4 = Dropout(0.2, trainable=True)(ac6)
    outputs = Conv2D(1, (1, 1), activation='relu')(d4)
    model = Model(inputs=[input_img], outputs=[outputs])

    return model
