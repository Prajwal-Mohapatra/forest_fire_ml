import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2

def conv_block(inputs, num_filters, kernel_size=3, dilation_rate=1, weight_decay=1e-5):
    """Convolution block with batch normalization, activation, and L2 regularization"""
    x = Conv2D(num_filters, kernel_size, padding='same', dilation_rate=dilation_rate, 
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def residual_block(inputs, num_filters, strides=1, weight_decay=1e-5):
    """Residual block with skip connection and L2 regularization"""
    x = conv_block(inputs, num_filters, weight_decay=weight_decay)
    x = Conv2D(num_filters, 3, padding='same', strides=strides, 
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    if strides != 1 or inputs.shape[-1] != num_filters:
        skip = Conv2D(num_filters, 1, strides=strides, padding='same', 
                     kernel_regularizer=l2(weight_decay))(inputs)
        skip = BatchNormalization()(skip)
    else:
        skip = inputs
    
    x = Add()([x, skip])
    x = Activation('relu')(x)
    return x

def atrous_spatial_pyramid_pooling(inputs, filters=256, weight_decay=1e-5):
    """ASPP module for multi-scale feature extraction with L2 regularization"""
    shape = inputs.shape
    
    # Image pooling
    pool = GlobalAveragePooling2D()(inputs)
    pool = Reshape((1, 1, shape[-1]))(pool)
    pool = Conv2D(filters, 1, padding='same', kernel_regularizer=l2(weight_decay))(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    pool = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(pool)
    
    # 1x1 convolution
    conv1 = Conv2D(filters, 1, padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # 3x3 convolutions with different dilation rates
    conv3_1 = Conv2D(filters, 3, padding='same', dilation_rate=6, 
                     kernel_regularizer=l2(weight_decay))(inputs)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    
    conv3_2 = Conv2D(filters, 3, padding='same', dilation_rate=12, 
                     kernel_regularizer=l2(weight_decay))(inputs)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    
    conv3_3 = Conv2D(filters, 3, padding='same', dilation_rate=18, 
                     kernel_regularizer=l2(weight_decay))(inputs)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    
    # Concatenate all features
    concat = Concatenate()([pool, conv1, conv3_1, conv3_2, conv3_3])
    
    # Final convolution
    output = Conv2D(filters, 1, padding='same', kernel_regularizer=l2(weight_decay))(concat)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    
    return output

def build_resunet_a(input_shape=(256, 256, 12), num_classes=1, dropout_rate=0.2, weight_decay=1e-5):
    """
    Build ResUNet-A architecture for fire prediction with L2 regularization
    Args:
        input_shape: Input tensor shape (updated to 12 channels for LULC encoding)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization (increased from 0.1 to 0.2)
        weight_decay: L2 regularization strength (added for better generalization)
    """
    inputs = Input(input_shape)
    
    # Encoder
    # Stage 1
    conv1 = conv_block(inputs, 64, weight_decay=weight_decay)
    conv1 = residual_block(conv1, 64, weight_decay=weight_decay)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Stage 2
    conv2 = residual_block(pool1, 128, weight_decay=weight_decay)
    conv2 = residual_block(conv2, 128, weight_decay=weight_decay)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Stage 3
    conv3 = residual_block(pool2, 256, weight_decay=weight_decay)
    conv3 = residual_block(conv3, 256, weight_decay=weight_decay)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Stage 4
    conv4 = residual_block(pool3, 512, weight_decay=weight_decay)
    conv4 = residual_block(conv4, 512, weight_decay=weight_decay)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    # Bridge with ASPP
    bridge = atrous_spatial_pyramid_pooling(pool4, 512, weight_decay=weight_decay)
    
    # Add dropout after bridge if specified
    if dropout_rate > 0:
        bridge = Dropout(dropout_rate)(bridge)
    
    # Decoder
    # Stage 4
    up4 = UpSampling2D((2, 2), interpolation='bilinear')(bridge)
    up4 = Concatenate()([up4, conv4])
    up4 = residual_block(up4, 512, weight_decay=weight_decay)
    up4 = residual_block(up4, 512, weight_decay=weight_decay)
    
    # Stage 3
    up3 = UpSampling2D((2, 2), interpolation='bilinear')(up4)
    up3 = Concatenate()([up3, conv3])
    up3 = residual_block(up3, 256, weight_decay=weight_decay)
    up3 = residual_block(up3, 256, weight_decay=weight_decay)
    
    # Stage 2
    up2 = UpSampling2D((2, 2), interpolation='bilinear')(up3)
    up2 = Concatenate()([up2, conv2])
    up2 = residual_block(up2, 128, weight_decay=weight_decay)
    up2 = residual_block(up2, 128, weight_decay=weight_decay)
    
    # Stage 1
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(up2)
    up1 = Concatenate()([up1, conv1])
    up1 = residual_block(up1, 64, weight_decay=weight_decay)
    up1 = residual_block(up1, 64, weight_decay=weight_decay)
    
    # Add dropout before final layer if specified
    if dropout_rate > 0:
        up1 = Dropout(dropout_rate)(up1)
    
    # Output layer with L2 regularization
    outputs = Conv2D(num_classes, 1, activation='sigmoid', padding='same', 
                    kernel_regularizer=l2(weight_decay))(up1)
    
    model = Model(inputs, outputs, name='ResUNet-A')
    return model
