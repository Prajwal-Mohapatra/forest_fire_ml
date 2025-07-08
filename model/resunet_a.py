# 4. model/resunet_a.py
# ====================
import tensorflow as tf
from keras.layers import *
from keras.models import Model

def conv_block(inputs, num_filters, kernel_size=3, dilation_rate=1):
    """Convolution block with batch normalization and activation"""
    x = Conv2D(num_filters, kernel_size, padding='same', dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    """Residual block with skip connection"""
    x = conv_block(inputs, num_filters)
    x = Conv2D(num_filters, 3, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    if strides != 1 or inputs.shape[-1] != num_filters:
        skip = Conv2D(num_filters, 1, strides=strides, padding='same')(inputs)
        skip = BatchNormalization()(skip)
    else:
        skip = inputs
    
    x = Add()([x, skip])
    x = Activation('relu')(x)
    return x

def atrous_spatial_pyramid_pooling(inputs, filters=256):
    """ASPP module for multi-scale feature extraction"""
    shape = inputs.shape
    
    # Image pooling
    pool = GlobalAveragePooling2D()(inputs)
    pool = Reshape((1, 1, shape[-1]))(pool)
    pool = Conv2D(filters, 1, padding='same')(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    pool = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(pool)
    
    # 1x1 convolution
    conv1 = Conv2D(filters, 1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # 3x3 convolutions with different dilation rates
    conv3_1 = Conv2D(filters, 3, padding='same', dilation_rate=6)(inputs)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    
    conv3_2 = Conv2D(filters, 3, padding='same', dilation_rate=12)(inputs)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    
    conv3_3 = Conv2D(filters, 3, padding='same', dilation_rate=18)(inputs)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    
    # Concatenate all features
    concat = Concatenate()([pool, conv1, conv3_1, conv3_2, conv3_3])
    
    # Final convolution
    output = Conv2D(filters, 1, padding='same')(concat)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    
    return output

def build_resunet_a(input_shape=(256, 256, 9), num_classes=1):
    """
    Build ResUNet-A architecture for fire prediction
    """
    inputs = Input(input_shape)
    
    # Encoder
    # Stage 1
    conv1 = conv_block(inputs, 64)
    conv1 = residual_block(conv1, 64)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Stage 2
    conv2 = residual_block(pool1, 128)
    conv2 = residual_block(conv2, 128)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Stage 3
    conv3 = residual_block(pool2, 256)
    conv3 = residual_block(conv3, 256)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Stage 4
    conv4 = residual_block(pool3, 512)
    conv4 = residual_block(conv4, 512)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    # Bridge with ASPP
    bridge = atrous_spatial_pyramid_pooling(pool4, 512)
    
    # Decoder
    # Stage 4
    up4 = UpSampling2D((2, 2), interpolation='bilinear')(bridge)
    up4 = Concatenate()([up4, conv4])
    up4 = residual_block(up4, 512)
    up4 = residual_block(up4, 512)
    
    # Stage 3
    up3 = UpSampling2D((2, 2), interpolation='bilinear')(up4)
    up3 = Concatenate()([up3, conv3])
    up3 = residual_block(up3, 256)
    up3 = residual_block(up3, 256)
    
    # Stage 2
    up2 = UpSampling2D((2, 2), interpolation='bilinear')(up3)
    up2 = Concatenate()([up2, conv2])
    up2 = residual_block(up2, 128)
    up2 = residual_block(up2, 128)
    
    # Stage 1
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(up2)
    up1 = Concatenate()([up1, conv1])
    up1 = residual_block(up1, 64)
    up1 = residual_block(up1, 64)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid', padding='same')(up1)
    
    model = Model(inputs, outputs, name='ResUNet-A')
    return model
