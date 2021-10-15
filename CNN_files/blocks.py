from keras.layers import  BatchNormalization, Activation, add
from keras.layers.convolutional import Conv2D



def conv2d_block(input_tensor, n_filters, kernel_size, activation):
    """Funkcja definiująca pojedynczą operację konwolucji. Została ona utworzona w celu ułatwienia
    czytelności kodu."""

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)



    return x

def residual_block(input_tensor, n_filters,activation, kernel_size = 5):
    """Funkcja definiująca blok resztkowy jako dwa bloki konwolucyjne i połącznenie skrótowe."""

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)


    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = add([input_tensor, x])
    return x