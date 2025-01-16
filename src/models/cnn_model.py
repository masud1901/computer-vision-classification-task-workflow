"""Advanced CNN model architecture for plant disease classification."""

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, Add, BatchNormalization, ReLU,
    GlobalAveragePooling2D, Reshape, Dense, Multiply, MaxPooling2D,
    Dropout, Concatenate, Layer, Lambda, LeakyReLU, AveragePooling2D
)
from tensorflow.keras.initializers import he_normal, glorot_uniform
from keras.saving import register_keras_serializable


@register_keras_serializable()
class ChannelAttention(Layer):
    """Channel Attention module to focus on important feature channels."""
    
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_1 = Dense(channel // self.ratio,
                                  activation='relu',
                                  kernel_initializer='he_normal')
        self.shared_dense_2 = Dense(channel,
                                  kernel_initializer='he_normal')
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = GlobalAveragePooling2D()(inputs)
        
        avg_pool = self.shared_dense_1(avg_pool)
        max_pool = self.shared_dense_1(max_pool)
        
        avg_pool = self.shared_dense_2(avg_pool)
        max_pool = self.shared_dense_2(max_pool)
        
        attention = Add()([avg_pool, max_pool])
        attention = Lambda(lambda x: tf.nn.sigmoid(x))(attention)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config


@register_keras_serializable()
class SpatialAttention(Layer):
    """Enhanced Spatial Attention module with refined attention mechanism."""
    
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = Conv2D(
            1, 
            self.kernel_size,
            padding='same',
            kernel_initializer=he_normal(),
            use_bias=False
        )
        self.bn = BatchNormalization()
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
        concat = Concatenate()([avg_pool, max_pool])
        
        attention = self.conv(concat)
        attention = self.bn(attention)
        attention = Lambda(lambda x: tf.nn.sigmoid(x))(attention)
        
        return Multiply()([inputs, attention])
    
    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class PlantDiseaseModel:
    """Advanced Plant Disease Classification Model with CBAM attention."""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_reg = regularizers.l2(1e-4)

    def _conv_block(self, x, filters, kernel_size=3, strides=1):
        """Basic convolution block with batch normalization and activation."""
        x = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=he_normal(),
            kernel_regularizer=self.l2_reg
        )(x)
        x = BatchNormalization()(x)
        return LeakyReLU(alpha=0.1)(x)

    def _attention_block(self, x):
        """Convolutional Block Attention Module (CBAM)."""
        x = ChannelAttention()(x)
        x = SpatialAttention()(x)
        return x

    def _dense_block(self, x, num_layers, growth_rate):
        """Dense block for feature reuse."""
        for _ in range(num_layers):
            shortcut = x
            x = self._conv_block(x, growth_rate, 1)
            x = self._conv_block(x, growth_rate, 3)
            x = Concatenate()([shortcut, x])
        return x

    def _transition_block(self, x, reduction):
        """Transition block to reduce feature map size."""
        filters = int(x.shape[-1] * reduction)
        x = self._conv_block(x, filters, 1)
        return AveragePooling2D(2, strides=2)(x)

    def _inception_module(self, x, filters):
        """Enhanced Inception module with varied receptive fields."""
        branch1x1 = self._conv_block(x, filters, 1)
        
        branch3x3 = self._conv_block(x, filters, 1)
        branch3x3 = self._conv_block(branch3x3, filters, 3)
        
        branch5x5 = self._conv_block(x, filters, 1)
        branch5x5 = self._conv_block(branch5x5, filters, 5)
        
        branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
        branch_pool = self._conv_block(branch_pool, filters, 1)
        
        return Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])

    def build_model(self):
        """Build and return the complete model architecture."""
        inputs = Input(shape=self.input_shape)
        
        # Initial processing
        x = self._conv_block(inputs, 64)
        x = self._attention_block(x)
        
        # First dense block and transition
        x = self._dense_block(x, num_layers=4, growth_rate=32)
        x = self._transition_block(x, reduction=0.5)
        
        # Inception module
        x = self._inception_module(x, filters=128)
        x = self._attention_block(x)
        
        # Second dense block and transition
        x = self._dense_block(x, num_layers=4, growth_rate=32)
        x = self._transition_block(x, reduction=0.5)
        
        # Final processing
        x = GlobalAveragePooling2D()(x)
        x = Dense(
            256,
            activation='relu',
            kernel_regularizer=self.l2_reg
        )(x)
        x = Dropout(0.5)(x)
        x = Dense(
            128,
            activation='relu',
            kernel_regularizer=self.l2_reg
        )(x)
        x = Dropout(0.3)(x)
        
        # Classification head
        outputs = Dense(
            self.num_classes,
            activation='softmax',
            kernel_initializer=glorot_uniform()
        )(x)
        
        return Model(inputs=inputs, outputs=outputs) 