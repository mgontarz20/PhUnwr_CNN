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

    c4 = conv2d_block(p3, n_filters * 8, kernel_size)
    r4 = residual_block(c4, n_filters * 8, kernel_size)
    cc4 = conv2d_block(r4, n_filters * 8, kernel_size)
    p4 = MaxPooling2D((2, 2), padding='same')(cc4)

    #bridge

    c5 = conv2d_block(p4, n_filters * 16, kernel_size)
    r5 = residual_block(c5, n_filters * 16, kernel_size)
    cc5 = conv2d_block(r5, n_filters * 16, kernel_size)

    #expanding
    u6 = Conv2DTranspose(n_filters * 8, (5, 5), strides=(2, 2), padding='same')(cc5)
    u6 = concatenate([u6, cc4])
    c7 = conv2d_block(u6, n_filters * 8, kernel_size)
    r7 = residual_block(c7, n_filters *8, kernel_size)
    cc7 = conv2d_block(r7, n_filters * 8, kernel_size)

    u7 = Conv2DTranspose(n_filters * 4, (5, 5), strides=(2, 2), padding='same')(cc7)
    u7 = concatenate([u7, cc3])
    c8 = conv2d_block(u7, n_filters * 4, kernel_size)
    r8 = residual_block(c8, n_filters * 4, kernel_size)
    cc8 = conv2d_block(r8, n_filters * 4, kernel_size)

    u8 = Conv2DTranspose(n_filters * 2, (5, 5), strides=(2, 2), padding='same')(cc8)
    u8 = concatenate([u8, cc2])
    c9 = conv2d_block(u8, n_filters * 2, kernel_size)
    r9 = residual_block(c9, n_filters * 2, kernel_size)
    cc9 = conv2d_block(r9, n_filters * 2, kernel_size)

    u9 = Conv2DTranspose(n_filters, (5, 5), strides=(2, 2), padding='same')(cc9)
    u9 = concatenate([u9, cc1])
    c10 = conv2d_block(u9, n_filters, kernel_size)
    r10 = residual_block(c10, n_filters, kernel_size)
    cc10 = conv2d_block(r10, n_filters, kernel_size)

    outputs = Conv2D(1, (1, 1), activation='relu')(cc10)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model