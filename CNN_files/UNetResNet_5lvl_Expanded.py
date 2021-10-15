from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from CNNs.blocks import conv2d_block, residual_block


def get_unet(input_img, n_filters=8, kernel_size = 3):
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
    p3 = MaxPooling2D((2, 2), padding='same')(cc_3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size)
    r4 = residual_block(c4, n_filters * 8, kernel_size)
    cc4 = conv2d_block(r4, n_filters * 8, kernel_size)
    c_4 = conv2d_block(cc4, n_filters * 8, kernel_size)
    r_4 = residual_block(c_4, n_filters * 8, kernel_size)
    cc_4 = conv2d_block(r_4, n_filters * 8, kernel_size)
    p4 = MaxPooling2D((2, 2), padding='same')(cc_4)

    c5 = conv2d_block(p4, n_filters * 16, kernel_size)
    r5 = residual_block(c5, n_filters * 16, kernel_size)
    cc5 = conv2d_block(r5, n_filters * 16, kernel_size)
    c_5 = conv2d_block(cc5, n_filters * 16, kernel_size)
    r_5 = residual_block(c_5, n_filters * 16, kernel_size)
    cc_5 = conv2d_block(r_5, n_filters * 16, kernel_size)
    p5 = MaxPooling2D((2, 2), padding='same')(cc_5)

    c6 = conv2d_block(p5, n_filters*32, kernel_size)
    r6 = residual_block(c6, n_filters * 32, kernel_size)
    cc6 = conv2d_block(r6, n_filters * 32, kernel_size)
    c_6 = conv2d_block(cc6, n_filters * 32, kernel_size)
    r_6 = residual_block(c_6, n_filters * 32, kernel_size)
    cc_6 = conv2d_block(r_6, n_filters * 32, kernel_size)

    # Ścieżka ekspansywna
    u6 = Conv2DTranspose(n_filters * 16, (3, 3), strides=(2, 2), padding='same')(cc_6)
    u6 = concatenate([u6, cc_5])
    c7 = conv2d_block(u6, n_filters * 8, kernel_size)
    r7 = residual_block(c7, n_filters * 8, kernel_size)
    cc7 = conv2d_block(r7, n_filters * 8, kernel_size)
    c_7 = conv2d_block(cc7, n_filters * 8, kernel_size)
    r_7 = residual_block(c_7, n_filters * 8, kernel_size)
    cc_7 = conv2d_block(r_7, n_filters * 8, kernel_size)

    u7 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(cc_7)
    u7 = concatenate([u7, cc_4])
    c8 = conv2d_block(u7, n_filters * 8, kernel_size)
    r8 = residual_block(c8, n_filters * 8, kernel_size)
    cc8 = conv2d_block(r8, n_filters * 8, kernel_size)
    c_8 = conv2d_block(cc8, n_filters * 8, kernel_size)
    r_8 = residual_block(c_8, n_filters * 8, kernel_size)
    cc_8 = conv2d_block(r_8, n_filters * 8, kernel_size)

    u8 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(cc_8)
    u8 = concatenate([u8, cc_3])
    c9 = conv2d_block(u8, n_filters * 4, kernel_size)
    r9 = residual_block(c9, n_filters * 4, kernel_size)
    cc9 = conv2d_block(r9, n_filters * 4, kernel_size)
    c_9 = conv2d_block(cc9, n_filters * 4, kernel_size)
    r_9 = residual_block(c_9, n_filters * 4, kernel_size)
    cc_9 = conv2d_block(r_9, n_filters * 4, kernel_size)

    u9 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(cc_9)
    u9 = concatenate([u9, cc_2])
    c10= conv2d_block(u9, n_filters * 2, kernel_size)
    r10 = residual_block(c10, n_filters * 2, kernel_size)
    cc10 = conv2d_block(r10, n_filters * 2, kernel_size)
    c_10 = conv2d_block(cc10, n_filters * 2, kernel_size)
    r_10 = residual_block(c_10, n_filters * 2, kernel_size)
    cc_10 = conv2d_block(r_10, n_filters * 2, kernel_size)

    u10 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(cc_10)
    u10 = concatenate([u10, cc_1])
    c11 = conv2d_block(u10, n_filters * 1, kernel_size)
    r11 = residual_block(c11, n_filters * 1, kernel_size)
    cc11 = conv2d_block(r11, n_filters * 1, kernel_size)
    c_11 = conv2d_block(cc11, n_filters * 1, kernel_size)
    r_11 = residual_block(c_11, n_filters * 1, kernel_size)
    cc_11 = conv2d_block(r_11, n_filters * 1, kernel_size)

    outputs = Conv2D(1, (1, 1), activation='relu')(cc_11)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model