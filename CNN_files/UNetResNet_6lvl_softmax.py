from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from CNNs.blocks import conv2d_block, residual_block


def get_unet(input_img,num_classes, n_filters=8, kernel_size = 3,):
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

    c5 = conv2d_block(p4, n_filters * 16, kernel_size)
    r5 = residual_block(c5, n_filters * 16, kernel_size)
    cc5 = conv2d_block(r5, n_filters * 16, kernel_size)
    p5 = MaxPooling2D((2, 2), padding='same')(cc5)

    c5_new = conv2d_block(p5, n_filters * 32, kernel_size)
    r5_new = residual_block(c5_new, n_filters * 32, kernel_size)
    cc5_new = conv2d_block(r5_new, n_filters * 32, kernel_size)
    p5_new = MaxPooling2D((2, 2), padding='same')(cc5_new)


    c6_new = conv2d_block(p5_new, n_filters*64, kernel_size)
    r6_new = residual_block(c6_new, n_filters * 64, kernel_size)
    cc6_new = conv2d_block(r6_new, n_filters * 64, kernel_size)

    # Ścieżka ekspansywna
    u6_new = Conv2DTranspose(n_filters * 32, (3, 3), strides=(2, 2), padding='same')(cc6_new)
    u6_new = concatenate([u6_new, cc5_new])
    c7_new = conv2d_block(u6_new, n_filters * 16, kernel_size)
    r7_new = residual_block(c7_new, n_filters * 16, kernel_size)
    cc7_new = conv2d_block(r7_new, n_filters * 16, kernel_size)


    u6 = Conv2DTranspose(n_filters * 16, (3, 3), strides=(2, 2), padding='same')(cc7_new)
    u6 = concatenate([u6, cc5])
    c7 = conv2d_block(u6, n_filters * 8, kernel_size)
    r7 = residual_block(c7, n_filters * 8, kernel_size)
    cc7 = conv2d_block(r7, n_filters * 8, kernel_size)

    u7 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(cc7)
    u7 = concatenate([u7, cc4])
    c8 = conv2d_block(u7, n_filters * 8, kernel_size)
    r8 = residual_block(c8, n_filters * 8, kernel_size)
    cc8 = conv2d_block(r8, n_filters * 8, kernel_size)

    u8 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(cc8)
    u8 = concatenate([u8, cc3])
    c9 = conv2d_block(u8, n_filters * 4, kernel_size)
    r9 = residual_block(c9, n_filters * 4, kernel_size)
    cc9 = conv2d_block(r9, n_filters * 4, kernel_size)

    u9 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(cc9)
    u9 = concatenate([u9, cc2])
    c10= conv2d_block(u9, n_filters * 2, kernel_size)
    r10 = residual_block(c10, n_filters * 2, kernel_size)
    cc10 = conv2d_block(r10, n_filters * 2, kernel_size)

    u10 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(cc10)
    u10 = concatenate([u10, cc1])
    c11 = conv2d_block(u10, n_filters * 1, kernel_size)
    r11 = residual_block(c11, n_filters * 1, kernel_size)
    cc11 = conv2d_block(r11, n_filters * 1, kernel_size)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(cc11)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model