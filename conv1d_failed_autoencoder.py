

    
def autoencode_and_stack(X, y=None, test_size=0.1, valid_size=0.1, encoding_dim=5):

    while X.shape[1] % 8 != 0:
        X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
 

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create validation split from train split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size)

    ### Train the autoencoder

    # SCALE EACH FEATURE INTO [0, 1] RANGE
    sX_train = minmax_scale(X_train, axis = 0)
    # SCALE EACH FEATURE INTO [0, 1] RANGE
    sX_valid = minmax_scale(X_valid, axis = 0)
    # SCALE EACH FEATURE INTO [0, 1] RANGE
    sX_test = minmax_scale(X_test, axis = 0)
    n_original_features = len(list(X))

    sX_train_ae = sX_train


    # Create split for autoencoder training
    sX_train_ae, sX_test_ae, dummy_y_train, dummy_y_test = train_test_split(sX_train, X_train, train_size = 0.5)

    # Stack minmax scaled to 3dim for autoencoder training
    sX_train_ae_training_stack = X_to_Conv1D_arrays(sX_train_ae)
    #sX_valid_stack = X_to_Conv1D_arrays(sX_valid)
    sX_test_ae_training_stack = X_to_Conv1D_arrays(sX_test_ae)

    # Stack minmax scaled to 3dim for autoencoder dimensional reduction later 
    sX_train_stack = X_to_Conv1D_arrays(sX_train)
    sX_valid_stack = X_to_Conv1D_arrays(sX_valid)
    sX_test_stack = X_to_Conv1D_arrays(sX_test)

    input_shape = sX_train_ae_training_stack.shape[1:3]

    ### AN EXAMPLE OF SIMPLE AUTOENCODER ###

    input_dim = Input(shape = input_shape, name ='input_layer')
    print("Input_shape is:", input_shape)

    # DEFINE THE DIMENSION OF ENCODER
    encoding_dim = encoding_dim

    # DEFINE THE ENCODER LAYERS
    encoded1 = Conv1D(8, kernel_size=12, padding='same', activation = 'relu', activity_regularizer=regularizers.l1(10e-5), name='first_conv1d_encoder_layer')(input_dim)
    encoded2 = MaxPooling1D(2)(encoded1)
    encoded3 = Conv1D(4, 6, padding='same', activation = 'relu', name='second_conv1d_encoder_layer')(encoded2)
    encoded4 = Conv1D(2, 3, padding='same', activation = 'relu', )(encoded3)
    encoded5 = MaxPooling1D(1)(encoded4)

    # DEFINE THE DECODER LAYERS
    #decoded1 = Conv1D(int(n_original_features*0.25), padding='same', activation = 'relu', )(encoded4)
    decoded1 = Conv1D(1, kernel_size=1, padding='same', activation = 'relu', name='first_conv1d_decoder_layer' )(encoded5)
    decoded2 = UpSampling1D(2)(decoded1)
    decoded3 = Conv1D(4,  kernel_size=6,  padding='same', activation = 'relu', name='second_conv1d_decoder_layer' )(decoded2)
    decoded4 = UpSampling1D(2)(decoded3)
    outputs = Conv1D(1, kernel_size=6, padding='same', activation='sigmoid')(decoded4)

    # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
    autoencoder = Model(input = input_dim, output = outputs)

    # CONFIGURE AND TRAIN THE AUTOENCODER
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy')
    autoencoder.fit(sX_train_ae_training_stack, sX_train_ae_training_stack, epochs = 1, batch_size = 256, shuffle = True, validation_split = 0.1)

    # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
    encoder = Model(input = input_dim, output = encoded5)
    encoded_input = Input(shape = (encoding_dim, ))
    #encoded_out = encoder.predict(sX_test_ae_training_stack)

    #Add a Flatten layer to the end
    x = Flatten()(encoder.layers[-1].output)
    encoder_model_input = encoder.input
    encoder = Model(input=encoder_model_input, output=[x])
    print(encoder.summary())


    # Dimensionality reduction
    X_train_ae = encoder.predict(sX_train_stack)
    X_test_ae = encoder.predict(sX_test_stack)
    X_valid_ae = encoder.predict(sX_valid_stack)


    # Stack original X_train to 3dim for conv net
    X_train = X_to_Conv1D_arrays(X_train)
    X_valid = X_to_Conv1D_arrays(X_valid)
    X_test = X_to_Conv1D_arrays(X_test)

    # Stack auto-encoded X_train to 3dim for conv net
    X_train_ae = X_to_Conv1D_arrays(X_train_ae)
    X_valid_ae = X_to_Conv1D_arrays(X_valid_ae)
    X_test_ae = X_to_Conv1D_arrays(X_test_ae)

    # Merge all features together
    X_train = np.concatenate((X_train, X_train_ae), axis=1)
    X_valid = np.concatenate((X_valid, X_valid_ae), axis=1)
    X_test = np.concatenate((X_test, X_test_ae), axis=1)


    n_original_features = len(list(X))
    print( "Original n of features:", n_original_features)

    autoencoded_features = X_train_ae.shape[1]
    print( "autoencoded_features:", autoencoded_features)

    return X_train, X_valid, X_test, y_train, y_valid, y_test
    #return X_train_ae



