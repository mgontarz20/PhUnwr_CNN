from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from CNNs.blocks import conv2d_block, residual_block



def get_unet(input_img, n_filters=8, kernel_size=3):
    """Funkcja definiująca model o architekturze U-Net z blokami resztkowymi."""
    #Ścieżka kurcząca

    c1 = conv2d_block(input_img, n_filters * 1, kernel_size)
    r1 = residual_block(c1, n_filters * 1, kernel_size)
    cc1 = conv2d_block(r1, n_filters * 1, kernel_size)
    p1 = MaxPooling2D((2, 2), padding='same')(cc1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size)
    r2 = residual_block(c2, n_filters * 2, kernel_size)
    cc2 = conv2d_block(r2, n_filters * 2, kernel_size)
    p2 = MaxPooling2D((2, 2), padding='same')(cc2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size)
    r3 = residual_block(c3, n_filters * 4, kernel_size)
    cc3 = conv2d_block(r3, n_filters * 4, kernel_size)
    p3 = MaxPooling2D((2, 2), padding='same')(cc3)

    #bridge

    c4 = conv2d_block(p3, n_filters * 8, kernel_size)
    r4 = residual_block(c4, n_filters * 8, kernel_size)
    cc4 = conv2d_block(r4, n_filters * 8, kernel_size)

    #expanding
    u5 = Conv2DTranspose(n_filters * 4, (5, 5), strides=(2, 2), padding='same')(cc4)
    u5 = concatenate([u5, cc3])
    c6 = conv2d_block(u5, n_filters * 4, kernel_size)
    r6 = residual_block(c6, n_filters * 4, kernel_size)
    cc6 = conv2d_block(r6, n_filters * 4, kernel_size)

    u6 = Conv2DTranspose(n_filters * 2, (5, 5), strides=(2, 2), padding='same')(cc6)
    u6 = concatenate([u6, cc2])
    c7 = conv2d_block(u6, n_filters * 2, kernel_size)
    r7 = residual_block(c7, n_filters * 2, kernel_size)
    cc7 = conv2d_block(r7, n_filters * 2, kernel_size)

    u8 = Conv2DTranspose(n_filters, (5, 5), strides=(2, 2), padding='same')(cc7)
    u8 = concatenate([u8, cc1])
    c8 = conv2d_block(u8, n_filters, kernel_size)
    r8 = residual_block(c8, n_filters, kernel_size)
    cc8 = conv2d_block(r8, n_filters, kernel_size)

    outputs = Conv2D(1, (1, 1), activation='relu')(cc8)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model