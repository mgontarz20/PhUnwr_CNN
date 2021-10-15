from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from CNNs.blocks import conv2d_block, residual_block




def get_unet(input_img, n_filters=16, kernel_size=5):
    """Funkcja definiująca model o architekturze U-Net z blokami resztkowymi."""
    #Ścieżka kurcząca

    c1 = conv2d_block(input_img, n_filters * 1, kernel_size)
    r1 = residual_block(c1, n_filters * 1, kernel_size)
    cc1 = conv2d_block(r1, n_filters * 1, kernel_size)
    p1 = MaxPooling2D((2, 2), padding='same')(cc1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size)
    r2 = residual_block(c2, n_filters * 2, kernel_size)
    cc2 = conv2d_block(r2, n_filters * 2, kernel_size)

    u3 = Conv2DTranspose(n_filters * 1, (5, 5), strides=(2, 2), padding='same')(cc2)
    u3 = concatenate([u3, cc1])
    c4 = conv2d_block(u3, n_filters * 1, kernel_size)
    r4 = residual_block(c4, n_filters * 1, kernel_size)
    cc4 = conv2d_block(r4, n_filters * 1, kernel_size)

    outputs = Conv2D(1, (1, 1), activation='relu')(cc4)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model