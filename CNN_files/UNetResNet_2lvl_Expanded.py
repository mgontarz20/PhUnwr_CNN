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
    c_1 = conv2d_block(cc1, n_filters * 1, kernel_size)
    r_1 = residual_block(c_1, n_filters * 1, kernel_size)
    cc_1 = conv2d_block(r_1, n_filters * 1, kernel_size)
    p1 = MaxPooling2D((2, 2), padding='same')(cc_1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size)
    r2 = residual_block(c2, n_filters * 2, kernel_size)
    cc2 = conv2d_block(r2, n_filters * 2, kernel_size)
    c_2 = conv2d_block(cc2, n_filters * 2, kernel_size)
    r_2 = residual_block(c_2, n_filters * 2, kernel_size)
    cc_2 = conv2d_block(r_2, n_filters * 2, kernel_size)
    p2 = MaxPooling2D((2, 2), padding='same')(cc_2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size)
    r3 = residual_block(c3, n_filters * 4, kernel_size)
    cc3 = conv2d_block(r3, n_filters * 4, kernel_size)
    c_3 = conv2d_block(cc3, n_filters * 4, kernel_size)
    r_3 = residual_block(c_3, n_filters * 4, kernel_size)
    cc_3 = conv2d_block(r_3, n_filters * 4, kernel_size)

    u4 = Conv2DTranspose(n_filters * 2, (5, 5), strides=(2, 2), padding='same')(cc_3)
    u4 = concatenate([u4, cc_2])
    c5 = conv2d_block(u4, n_filters * 2, kernel_size)
    r5 = residual_block(c5, n_filters * 2, kernel_size)
    cc5 = conv2d_block(r5, n_filters * 2, kernel_size)
    c_5 = conv2d_block(cc5, n_filters * 2, kernel_size)
    r_5 = residual_block(c_5, n_filters * 2, kernel_size)
    cc_5 = conv2d_block(r_5, n_filters * 2, kernel_size)

    u5 = Conv2DTranspose(n_filters, (5, 5), strides=(2, 2), padding='same')(cc_5)
    u5 = concatenate([u5, cc_1])
    c6 = conv2d_block(u5, n_filters, kernel_size)
    r6 = residual_block(c6, n_filters, kernel_size)
    cc6 = conv2d_block(r6, n_filters, kernel_size)
    c_6 = conv2d_block(cc6, n_filters, kernel_size)
    r_6 = residual_block(c_6, n_filters, kernel_size)
    cc_6 = conv2d_block(r_6, n_filters, kernel_size)

    outputs = Conv2D(1, (1, 1), activation='relu')(cc_6)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model