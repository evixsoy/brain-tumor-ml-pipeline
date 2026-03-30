import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras import regularizers

def attention_block(x, gating, inter_channels, name):

    # convert encode skip connection na stejny conv2d 1x1 jak decode skip connection
    theta_x = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', name=f'{name}_theta')(x)
    
    # Encoder signal projection
    phi_g = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', name=f'{name}_phi')(gating)
    
    # kombinace encoder + decoder
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    
    # Attention score
    psi_f = layers.Conv2D(1, kernel_size=1, strides=1, padding='same', name=f'{name}_psi')(f)
    
    # attention mapa 0 = ignore , 1 = use
    rate = layers.Activation('sigmoid')(psi_f)
    
    # Apply attention
    return layers.multiply([x, rate], name=f'{name}_output')

def create_unet_efficientnet(shape_input=(512, 512, 3), classes=4, encoder_freeze=False, decoder_sizes=[512, 256, 128, 64, 32], dropout_rate=0.5, backbone='B0'):
    inputs = keras.Input(shape=shape_input, name='image_input')

    if backbone == 'B3':
        encoder_model = EfficientNetB3
        skip_layers = ['block1b_add', 'block2c_add', 'block3c_add', 'block5e_add']
    else: # Default B0
        encoder_model = EfficientNetB0
        skip_layers = ['block1a_project_bn', 'block2b_add', 'block3b_add', 'block4c_add']

    encoder = encoder_model(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None
    )

    if not encoder_freeze:
        #training whole model
        for layer in encoder.layers:
            layer.trainable = True
    else:
        # decoder + bottleneck - fine tuning
        for layer in encoder.layers:
            layer.trainable = False
    
    skips = []
    for layer_name in skip_layers:
        skips.append(encoder.get_layer(layer_name).output)

    bridge = encoder.output
    bridge = layers.Conv2D(1024,kernel_size=3, padding='same',name ='bottleneck_conv1', kernel_regularizer=regularizers.l2(1e-4),)(bridge)
    bridge = layers.BatchNormalization(name='bottleneck_bn1', momentum=0.9)(bridge)
    bridge = layers.Activation('relu', name='bottleneck_relu1')(bridge)

    bridge = layers.Conv2D(1024,kernel_size=3, padding='same',name ='bottleneck_conv2', kernel_regularizer=regularizers.l2(1e-4),)(bridge)
    bridge = layers.BatchNormalization(name='bottleneck_bn2', momentum=0.9)(bridge)
    bridge = layers.Activation('relu', name='bottleneck_relu2')(bridge)
    bridge = layers.Dropout(dropout_rate, name='bottleneck_dropout')(bridge)

    skip_connections_list = skips[::-1]
    upsample = bridge

    #create decoder blocks
    for i in range(len(skip_connections_list)):
        upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=f'decoder_upsample_{i+1}')(upsample)
        
        # pouziti attention gate na skip connection
        att = attention_block(x=skip_connections_list[i], gating=upsample, inter_channels=decoder_sizes[i], name=f'attention_{i+1}')
        
        upsample = layers.Concatenate(name=f'decoder_concat_{i+1}')([upsample, att])
        
        for g in range(2):
            upsample = layers.Conv2D(decoder_sizes[i], kernel_size=3, padding='same', name=f'decoder_conv{i+1}_{g+1}', kernel_regularizer=regularizers.l2(1e-4))(upsample)
            upsample = layers.BatchNormalization(name=f'decoder_bn{i+1}_{g+1}', momentum=0.9)(upsample)
            upsample = layers.Activation('relu', name=f'decoder_relu{i+1}_{g+1}')(upsample)
        
        upsample = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i+1}')(upsample)
    
    upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='decoder_upsample_final')(upsample)
    upsample = layers.Conv2D(64, kernel_size=3, padding='same', name='decoder_conv_final_1', kernel_regularizer=regularizers.l2(1e-4))(upsample)
    upsample = layers.BatchNormalization(name='decoder_bn_final', momentum=0.9)(upsample)
    upsample = layers.Activation('relu', name='decoder_relu_final')(upsample)
    
    #output layer - multiclass
    outputs = layers.Conv2D(classes, kernel_size=1, activation = 'softmax', name='output', dtype='float32', kernel_regularizer=regularizers.l2(1e-4),)(upsample)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f'unet-efficientnet{backbone.lower()}')
    return model
