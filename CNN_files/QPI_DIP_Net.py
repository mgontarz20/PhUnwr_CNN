from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.merge import concatenate
from keras.layers import PReLU, BatchNormalization, UpSampling2D
from CNNs.blocks import initial_conv, conv2D_3x3_s2,conv2D_1x1_s1,conv2D_3x3_s1,bilinear_interp_Upsampling_2x2, skip_connection_1x1




def get_model(input_img, n_filters = 128):

    #DOWNSAMPLING - ENCODER
    #Level 1
    in1 = initial_conv(input_img, n_filters)
    print(in1.shape)
    #Level 2
    c1 = conv2D_3x3_s2(in1, n_filters)
    c2 = conv2D_3x3_s1(c1, n_filters)
    print(c2.shape)
    #Level 3
    c3 = conv2D_3x3_s2(c2, n_filters)
    c4 = conv2D_3x3_s1(c3, n_filters)
    print(c4.shape)
    #Level 4
    c5 = conv2D_3x3_s2(c4, n_filters)
    c6 = conv2D_3x3_s1(c5, n_filters)
    print(c6.shape)
    #Level 5
    c7 = conv2D_3x3_s2(c6, n_filters)
    c8 = conv2D_3x3_s1(c7, n_filters)
    print(c8.shape)
    #Level 6 - bridge
    c9 = conv2D_3x3_s2(c8, n_filters)
    c10 = conv2D_3x3_s1(c9, n_filters)
    print(c10.shape)
    #UPSAMPLING - DECODER

    #Level 5
    u1 = bilinear_interp_Upsampling_2x2(c10)
    con1 = skip_connection_1x1(c8, u1)
    c10 = conv2D_3x3_s1(con1, n_filters)
    c11 = conv2D_1x1_s1(c10, n_filters)

    # Level 4
    u2 = bilinear_interp_Upsampling_2x2(c11)
    con2 = skip_connection_1x1(c6, u2)
    c12 = conv2D_3x3_s1(con2, n_filters)
    c13 = conv2D_1x1_s1(c12, n_filters)

    # Level 3
    u3 = bilinear_interp_Upsampling_2x2(c13)
    con3 = skip_connection_1x1(c4, u3)
    c14 = conv2D_3x3_s1(con3, n_filters)
    c15 = conv2D_1x1_s1(c14, n_filters)

    # Level 2
    u4 = bilinear_interp_Upsampling_2x2(c15)
    con4 = skip_connection_1x1(c2, u4)
    c16 = conv2D_3x3_s1(con4, n_filters)
    c17 = conv2D_1x1_s1(c16, n_filters)

    # Level 1
    u5 = bilinear_interp_Upsampling_2x2(c17)
    con5 = skip_connection_1x1(in1, u5)
    c18 = conv2D_3x3_s1(con5, n_filters)
    c19 = conv2D_1x1_s1(c18, n_filters)

    output = Conv2D(filters = 1, kernel_size=(1,1), strides=(1,1))(c19)

    model = Model(inputs=[input_img], outputs=[output])
    return model



