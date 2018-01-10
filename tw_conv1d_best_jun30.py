from keras.layers import Input


def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    
    if e > end:
        return lr_end
    
    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))
    
    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end



def residual_block(input_tensor, n_conv, elu=1, relu=0, first=0):
    
    # Input convolution
    if first:
        x1 = Conv1D(kernel_size=1, filters=n_conv, strides=1, padding='same')(input_tensor)
    else:
        x1 = Conv1D(n_conv, (1), padding='same')(input_tensor)   

    # Define the inner_residual_block
    def inner_residual_block(input_tensor, n_conv, elu=1, relu=0):
        # First batchnorm
        x2 = BatchNormalization(axis=1)(input_tensor)
        # First activation
        if elu:
            x2 = ELU()(x2)
        elif relu:
            x2 = LeakyReLU()(x2)
        # First convolution
        x2 = Conv1D(n_conv, (3), padding='same')(x2)  

        # Second batchnorm
        x2 = BatchNormalization(axis=1)(input_tensor)
        # Second activation
        if elu:
            x2 = ELU()(x2)
        elif relu:
            x2 = LeakyReLU()(x2)
        # Second convolution
        x2 = Conv1D(n_conv, (3), padding='same')(x2)
        return x2  

    x = inner_residual_block(input_tensor, n_conv, elu=elu, relu=relu)
    x = Dropout(0.4)(x)
    x = inner_residual_block(x, n_conv, elu=elu, relu=relu)   
    x = add([x1, x])

    return x  

elu = 1
relu = 0

input_layer = Input(shape=input_shape)
a = Conv1D(32, (3), padding='same', strides=2)(input_layer)
b = residual_block(a, 64, elu=elu, relu=relu, first=1)
b = residual_block(a, 128, elu=elu, relu=relu)
b = residual_block(a, 256, elu=elu, relu=relu)
# b = residual_block(a, 32, elu=elu, relu=relu)

# c = residual_block(b, 64, elu=elu, relu=relu, first=1)
# c = residual_block(c, 64, elu=elu, relu=relu)
# c = residual_block(c, 64, elu=elu, relu=relu)
# c = residual_block(c, 64, elu=elu, relu=relu)

# d = residual_block(c, 128, elu=elu, relu=relu, first=1)
# d = residual_block(d, 128, elu=elu, relu=relu)
# d = residual_block(d, 128, elu=elu, relu=relu)
# d = residual_block(d, 128, elu=elu, relu=relu)


e = MaxPooling1D(2)(b)
e = Flatten()(e)
#d = Dense(256)(d)
#d = LeakyReLU(alpha=0.02)(d)
#d = Dropout(0.4)(d)
#e = Dense(output_shape)(e)
#e = Activation('softmax')(e)

x = Dense(128)(e)
x = LeakyReLU(alpha=(0.02))(x)
x = Dense(128)(x)
x = LeakyReLU(alpha=(0.02))(x)
x = Dense(128)(x)
x = LeakyReLU(alpha=(0.02))(x)
x = Dropout(0.5, name='dropout_70')(x)
x = Dense(output_shape)(x)
x = LeakyReLU(alpha=(0.02))(x)
x = Activation('softmax')(x)

model = Model(input=input_layer, output=x)