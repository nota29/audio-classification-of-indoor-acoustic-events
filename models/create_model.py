from keras.layers import Dense, add, BatchNormalization, Dropout, Conv2D, Flatten,\
    MultiHeadAttention, LayerNormalization, Input, concatenate, Activation, average,\
    Conv1D, MaxPooling2D, Rescaling, GlobalMaxPool2D
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, add, BatchNormalization, Dropout, Conv2D, Flatten,\
    AveragePooling2D, GlobalMaxPooling2D, Input, concatenate, Activation, average,\
    Conv1D, MaxPooling2D, Permute, Reshape, Rescaling, MaxPooling1D, Resizing, DepthwiseConv2D, Add, GlobalAveragePooling2D,\
    Multiply, ReLU, Lambda
import tensorflow_addons as tfa


#This function creates a 1D Convolution model
def conv1d(x_size, y_size, n_channels, grouping_num, initializer, regularizer, name, activation):
    """
    :param x_size: Dimension of x_axis
    :param y_size: Dimension of y_axis
    :param n_channels: Whether the images to be used are in grayscale or RGB format
    :param grouping_num: Grouping parameter for group normalization layers
    :param initializer: Weight initializer function
    :param regularizer: Regularizer function
    :param name: Model name
    :param activation: Activation function to be used in every layer, except the output one
    :return: A model object
    """
    groups_a = grouping_num
    # Preprocessing layers
    # Reshape data to be in the following format:(x_size,y_size) from (x_size, y_size,1)
    # Permute the data from (n_mfcc,time) to (time,n_mfcc), in order to have 1D Convolution
    model_input = Input(shape=(x_size, y_size, n_channels), name='input_layer')
    reshape_layer = Reshape(target_shape=(x_size, y_size), name='reshape_layer')(model_input)
    # to rescale between [-1,1] use rescale=1./127.5, offset=-1
    # to rescale between [0,1] use rescale=1./255, offset=0
    rescale_layer = Rescaling(scale=1. / 255, offset=0, name='Rescale_layer')(reshape_layer)
    permute_layer = Permute((2, 1), name='permute_layer')(rescale_layer)
    # First layers of each side, a denotes one model, and b the other
    group1_a = tfa.layers.GroupNormalization(groups=groups_a, name='group1_a')(permute_layer)
    conv1d_1a = tfa.layers.WeightNormalization(
        Conv1D(32, kernel_size=3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer,
               name='conv1d_1a'))(group1_a)
    # maxpool_a = MaxPooling1D(name='maxpool_a')(conv1d_1a)
    # Second layers
    group2_a = tfa.layers.GroupNormalization(groups=groups_a, name='group2_a')(conv1d_1a)
    conv1d_2a = tfa.layers.WeightNormalization(
        Conv1D(64, kernel_size=3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer,
               name='conv1d_2a'))(group2_a)
    dropout1_a = Dropout(0.5, name='dropout1_a')(conv1d_2a)
    maxpool1_a = MaxPooling1D(name='maxpool1_a')(dropout1_a)

    # Third layers
    group3_a = tfa.layers.GroupNormalization(groups=groups_a, name='group3_a')(maxpool1_a)
    conv1d_3a = tfa.layers.WeightNormalization(
        Conv1D(128, kernel_size=3, activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1d_3a'))(group3_a)
    dropout2_a = Dropout(0.5, name='dropout2_a')(conv1d_3a)
    maxpool2_a = MaxPooling1D(name='maxpool2_a')(dropout2_a)

    # Fourth Layers
    group4_a = tfa.layers.GroupNormalization(groups=groups_a, name='group4_a')(maxpool2_a)
    conv1d_4a = tfa.layers.WeightNormalization(
        Conv1D(256, kernel_size=3, activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1d_4a'))(group4_a)
    dropout3_a = Dropout(0.5, name='dropout3_a')(conv1d_4a)
    maxpool3_a = MaxPooling1D(name='maxpool3_a')(dropout3_a)

    # Fifth layers
    group5_a = tfa.layers.GroupNormalization(groups=groups_a, name='group5_a')(maxpool3_a)
    conv1d_5a = tfa.layers.WeightNormalization(
        Conv1D(512, kernel_size=3, activation=activation, kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1d_5a'))(group5_a)
    dropout4_a = Dropout(0.5, name='dropout4_a')(conv1d_5a)
    maxpool4_a = MaxPooling1D(name='maxpool4_a')(dropout4_a)

    # Last group normalization and flattening
    group8_a = tfa.layers.GroupNormalization(groups=groups_a, name='group8_a')(maxpool4_a)
    flatten_a = Flatten(name='flatten_a')(group8_a)
    # Dense layers
    dropout6_a = Dropout(0.5, name='dropout6_a')(flatten_a)
    dense2_a = tfa.layers.WeightNormalization(
        Dense(128, activation=activation, kernel_initializer=initializer,
              kernel_regularizer=regularizer, name='dense2_a'))(dropout6_a)
    dropout7_a = Dropout(0.5, name='dropout7_a')(dense2_a)
    out_a = tfa.layers.WeightNormalization(
        Dense(9, activation='softmax', kernel_initializer=initializer, kernel_regularizer=regularizer,
              name='out_a'))(dropout7_a)
    model = tf.keras.models.Model(inputs=model_input, outputs=out_a, name=name)
    return model


# The following function creates a 2D Convolutional model with symmetrical kernels
def conv2d_same_kernels(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):
    """ 
    :param name: model name
    :param x_size: image dimension
    :param y_size: image dimension
    :param n_channels: Whether images are in grayscale or rgb format
    :param initializer: kernel initializing function to use
    :param reg: Kernel regularizer function to use for generalization puprposes
    :param batchsize: Batchsize to be used
    :param logits: Define whether the model returns the logits or the probabilities
    :returns model: A keras model object
    """
    model = Sequential(name=name)
    # Input layer
    model.add(Rescaling(1 / 255, offset=0, name='rescaling', input_shape=(x_size, y_size, n_channels)))
    model.add(BatchNormalization(name='batchnorm1'))
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer=initializer, activation='gelu',
                     input_shape=(x_size, y_size, n_channels), padding='same', kernel_regularizer=reg, name='conv2ds_1'))
    model.add(MaxPooling2D(name='maxpool1'))
    model.add(Dropout(0.4))
    # First Hidden Layer
    model.add(BatchNormalization(name='batchnorm2'))
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu',
                     name='conv2ds_2'))
    model.add(MaxPooling2D(name='maxpool2'))
    model.add(Dropout(0.4))
    # Second Hidden Layer
    model.add(BatchNormalization(name='batchnorm3'))
    model.add(Conv2D(128, kernel_size=(3, 3), kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu',
                     name='conv2ds_3'))
    model.add(MaxPooling2D(name='maxpool3'))
    model.add(Dropout(0.5, name='dropout2'))
    # Third Hidden Layer
    model.add(BatchNormalization(name='batchnorm4'))
    model.add(Conv2D(256, kernel_size=(5, 5), kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu',
                     name='conv2ds_4'))
    model.add(MaxPooling2D(name='maxpool4'))

    # Output Layer
    model.add(GlobalMaxPooling2D(name='GlobalMaxPool'))
    model.add(BatchNormalization(name='batchnorm5'))
    model.add(Dense(64, kernel_regularizer=reg, kernel_initializer=initializer, activation='gelu', name='dense1'))
    model.add(Dropout(0.5, name='dropout3'))
    if logits:
        model.add(Dense(9, kernel_regularizer=reg, kernel_initializer=initializer, name='output'))
    else:
        model.add(Dense(9, kernel_regularizer=reg, kernel_initializer=initializer, activation='softmax', name='output'))
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))

    return model


# This function creates a Conv2D model with asymmetrical kernels
def conv2d_diff_kernels(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):
    """ 
    :param name: model name
    :param x_size: image dimension
    :param y_size: image dimension
    :param n_channels: Whether images are in grayscale or rgb format
    :param initializer: kernel initializing function to use
    :param reg: Kernel regularizer function to use for generalization purposes
    :param batchsize: Batchsize to be used
    :param logits: Define whether the model returns the logits or the probabilities
    :returns model: A keras model object
    """
    # Input layers
    model_input = Input(shape=(x_size, y_size, n_channels), name='input')
    rescale = Rescaling(1 / 255, offset=0, name='rescale')(model_input)

    # First layers
    batch_norm1 = BatchNormalization(name='batch1')(rescale)
    conv2d_1 = Conv2D(64, kernel_size=(5, 1), kernel_initializer=initializer, activation='gelu',
                      kernel_regularizer=reg, name='conv2d_1')(batch_norm1)
    dropout = Dropout(0.2, name='dropout')(conv2d_1)
    max_pool = MaxPooling2D(pool_size=(3, 1), name='maxpool')(dropout)

    # Second layers
    batch_norm2 = BatchNormalization(name='batch2')(max_pool)
    conv2d_2 = Conv2D(128, kernel_size=(10, 1), kernel_initializer=initializer, activation='gelu',
                      kernel_regularizer=reg, name='conv2d_2')(batch_norm2)

    # Third layers
    batch_norm3 = BatchNormalization(name='batch3')(conv2d_2)
    dropout1 = Dropout(0.3, name='dropout1')(batch_norm3)
    conv2d_3 = Conv2D(256, kernel_size=(1, 6), kernel_initializer=initializer, activation='gelu',
                      kernel_regularizer=reg, name='conv2d_3')(dropout1)
    max_pool2 = MaxPooling2D(pool_size=(1, 4), name='maxpool2')(conv2d_3)
    batch_norm4 = BatchNormalization(name='batch4')(max_pool2)

    Gap = GlobalMaxPool2D(name='gap')(batch_norm4)
    # Dense layers
    dense = Dense(128, activation='gelu', kernel_initializer=initializer, name='dense', kernel_regularizer=reg)(Gap)
    dropout = Dropout(0.5, name='dropout2')(dense)
    if logits:
        model_output = Dense(9, activation=None, kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)
    else:
        model_output = Dense(9, activation='softmax', kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)
    model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=name)
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))
    return model


#The following two functions create a Conv2D MobileNetV1-based model
def dw_separable(x, pointwise_filters, stride, block_id):
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False,
                        name=f'dw_{block_id}')(x)
    x = BatchNormalization(name=f'bn_dw_{block_id}')(x)
    x = ReLU(6., name=f'relu_dw_{block_id}')(x)

    # Pointwise
    x = Conv2D(pointwise_filters, kernel_size=1, strides=1, padding='same', use_bias=False,
               name=f'pw_{block_id}')(x)
    x = BatchNormalization(name=f'bn_pw_{block_id}')(x)
    x = ReLU(6., name=f'relu_pw_{block_id}')(x)

    return x

def mobile_net_v1(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):

    #input layers
    model_input = Input(shape=(x_size, y_size, n_channels), name='input') 
    rescale = Rescaling(1 / 255, offset=0, name='rescale')(model_input)

    #stem
    conv2d_1 = Conv2D(32, kernel_size=(3, 3),strides=2, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_1')(rescale)
    batch_norm1 = BatchNormalization(name='batch1')(conv2d_1)
    act1 = ReLU(6., name='relu1')(batch_norm1)

    #body
    stage1 = dw_separable(act1, 64, 1, 1)
    stage2 = dw_separable(stage1, 128, 2, 2)
    stage3 = dw_separable(stage2, 128, 1, 3)
    stage4 = dw_separable(stage3, 256, 2, 4) 
    stage5 = dw_separable(stage4, 256, 1, 5)
    stage6 = dw_separable(stage5, 512, 2, 6)
    stage7_1 = dw_separable(stage6, 512, 1, 7_1)
    stage7_2 = dw_separable(stage7_1, 512, 1, 7_2)
    stage7_3 = dw_separable(stage7_2, 512, 1, 7_3)
    stage7_4 = dw_separable(stage7_3, 512, 1, 7_4)
    stage7_5 = dw_separable(stage7_4, 512, 1, 7_5)
    stage8 = dw_separable(stage7_5, 1024, 2, 8)
    stage9 = dw_separable(stage8, 1024, 1, 9)

    #head
    pooling = GlobalAveragePooling2D(name='pooling')(stage9)


    dense = Dense(128, activation=tf.nn.relu6, kernel_initializer=initializer, name='dense', kernel_regularizer=reg)(pooling)
    dropout = Dropout(0.5, name='dropout')(dense)
    

    if logits:
        model_output = Dense(9, activation=None, kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)
    else:
        model_output = Dense(9, activation='softmax', kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)

    model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=name)
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))
    return model


#mobile_v3 models

#The following 3 functions are used for the MobileNetV3-based models
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6


def se_block(inputs, se_ratio=0.25):
    filters = inputs.shape[-1]
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, 1, filters))(se)
    se = Conv2D(int(filters * se_ratio), 1, activation='relu')(se)
    se = Conv2D(filters, 1, activation='hard_sigmoid')(se)
    return Multiply()([inputs, se])



def bottleneck(x, out_channels, kernel_size, expansion_size, stride, se, activation='relu'):
    inputs = x
    in_channels = x.shape[-1]
    if expansion_size != in_channels:
        x = Conv2D(expansion_size, 1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation(hard_swish if activation == 'hard_swish' else activation)(x)

    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(hard_swish if activation == 'hard_swish' else activation)(x)

    if se:
        x = se_block(x)

    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    if stride == 1 and in_channels == out_channels:
        x = Add()([inputs, x])
    return x


def mobile_net_v3_small(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):

    #input layer
    model_input = Input(shape=(x_size, y_size, n_channels), name='input')
    rescale = Rescaling(1 / 255, offset=0, name='rescale')(model_input)

    #stem
    conv2d_a = Conv2D(16, kernel_size=(3, 3),strides=2, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_a')(rescale)
    batch_norm_a = BatchNormalization(name='batch_a')(conv2d_a)
    act1 = Lambda(hard_swish)(batch_norm_a)

    #body
    stage1 = bottleneck(act1, 16, 3, 16, 2, se=True, activation='relu')
    stage2 = bottleneck(stage1, 24, 3, 72, 2, se=False, activation='relu')
    stage3 = bottleneck(stage2, 24, 3, 88, 1, se=False, activation='relu')
    stage4 = bottleneck(stage3, 40, 5, 96, 2, se=True, activation='hard_swish')
    stage5 = bottleneck(stage4, 40, 5, 240, 1, se=True, activation='hard_swish')
    stage6 = bottleneck(stage5, 40, 5, 240, 1, se=True, activation='hard_swish')
    stage7 = bottleneck(stage6, 48, 5, 120, 1, se=True, activation='hard_swish')
    stage8 = bottleneck(stage7, 48, 5, 144, 1, se=True, activation='hard_swish')
    stage9 = bottleneck(stage8, 96, 5, 288, 2, se=True, activation='hard_swish')
    stage10 = bottleneck(stage9, 96, 5, 576, 1, se=True, activation='hard_swish')
    stage11 = bottleneck(stage10, 96, 5, 576, 1, se=True, activation='hard_swish')

    #head
    conv2d_b = Conv2D(576, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_b')(stage11)
    batch_norm_b = BatchNormalization(name='batch_b')(conv2d_b)
    act_b = Lambda(hard_swish)(batch_norm_b)
    pooling = GlobalAveragePooling2D(name='pooling')(act_b)
    reshape = Reshape((1, 1, 576))(pooling)
    conv2d_c = Conv2D(1024, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_c')(reshape)
    act_c = Lambda(hard_swish)(conv2d_c) 
    conv2d_e = Conv2D(256, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_e')(act_c)
    act_e = Lambda(hard_swish)(conv2d_e)
    conv2d_d = Conv2D(9, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_d')(act_e)
    pooling2 = GlobalAveragePooling2D(name='pooling2')(conv2d_d)
    if logits:
        model_output = Dense(9, activation=None, kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(pooling2)
    else:
        model_output = Dense(9, activation='softmax', kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(pooling2)

    model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=name)
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))
    return model


#mobile_v3_large
def mobile_net_v3_large(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):

    #input layer
    model_input = Input(shape=(x_size, y_size, n_channels), name='input')
    rescale = Rescaling(1 / 255, offset=0, name='rescale')(model_input)

    #stem
    conv2d_a = Conv2D(16, kernel_size=(3, 3),strides=2, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_a')(rescale)
    batch_norm_a = BatchNormalization(name='batch_a')(conv2d_a)
    act_a = Lambda(hard_swish)(batch_norm_a)

    #body
    stage1 = bottleneck(act_a, 16, 3, 16, 1, se=False, activation='relu')
    stage2 = bottleneck(stage1, 24, 3, 64, 2, se=False, activation='relu')
    stage3 = bottleneck(stage2, 24, 3, 72, 1, se=False, activation='relu')
    stage4 = bottleneck(stage3, 40, 5, 72, 2, se=True, activation='relu')
    stage5 = bottleneck(stage4, 40, 5, 120, 1, se=True, activation='relu')
    stage6 = bottleneck(stage5, 40, 5, 120, 1, se=True, activation='relu')
    stage7 = bottleneck(stage6, 80, 3, 240, 2, se=False, activation='hard_swish')
    stage8 = bottleneck(stage7, 80, 3, 200, 1, se=False, activation='hard_swish')
    stage9 = bottleneck(stage8, 80, 3, 184, 1, se=False, activation='hard_swish')
    stage10 = bottleneck(stage9, 80, 3, 184, 1, se=False, activation='hard_swish')
    stage11 = bottleneck(stage10, 112, 3, 480, 1, se=True, activation='hard_swish')
    stage12 = bottleneck(stage11, 112, 3, 672, 1, se=True, activation='hard_swish')
    stage13 = bottleneck(stage12, 160, 5, 672, 2, se=True, activation='hard_swish')
    stage14 = bottleneck(stage13, 160, 5, 960, 1, se=True, activation='hard_swish')
    stage15 = bottleneck(stage14, 160, 5, 960, 1, se=True, activation='hard_swish')

    #head
    conv2d_b = Conv2D(960, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_b')(stage15)
    batch_norm_b = BatchNormalization(name='batch_b')(conv2d_b)
    act_b = Lambda(hard_swish)(batch_norm_b)
    pooling = GlobalAveragePooling2D(name='pooling')(act_b)
    reshape = Reshape((1, 1, 960))(pooling)
    conv2d_c = Conv2D(1280, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_c')(reshape)
    act_c = Lambda(hard_swish)(conv2d_c)
    conv2d_e = Conv2D(128, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_e')(act_c)
    act_e = Lambda(hard_swish)(conv2d_e)
    conv2d_d = Conv2D(9, kernel_size=(1, 1),strides=1, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_d')(act_e)
    
    pooling2 = GlobalAveragePooling2D(name='pooling2')(conv2d_d)
    if logits:
        model_output = Dense(9, activation=None, kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(pooling2)
    else:
        model_output = Dense(9, activation='softmax', kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(pooling2)
    
    model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=name)
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))
    return model


#The following 2 functions create a EfficientNet-B0-based model
def MBConvBlock(x, input_filters, output_filters, kernel_size, stride, expansion_factor, name):
    shortcut = x

    # Expand
    expanded_filters = input_filters * expansion_factor
    if expansion_factor != 1:
        x = Conv2D(expanded_filters, 1, padding='same', use_bias=False, name=name+'_expand')(x)
        x = BatchNormalization(name=name+'_bn1')(x)
        x = Activation(tf.nn.swish, name=name+'_relu1')(x)

    # Depthwise conv
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False, name=name+'_dw')(x)
    x = BatchNormalization(name=name+'_bn2')(x)
    x = Activation(tf.nn.swish, name=name+'_relu2')(x)

    #Squeeze and Excitation
    se_ratio=0.25
    in_channels = x.shape[-1]
    reduced_channels = max(1, int(in_channels * se_ratio))

    se = GlobalAveragePooling2D(name=name+'_se'+'_squeeze')(x)
    se = Reshape((1, 1, in_channels))(se)
    se = Conv2D(reduced_channels, 1, activation=tf.nn.swish, name=name+'_se'+'_reduce')(se)
    se = Conv2D(in_channels, 1, activation='sigmoid', name=name+'_se'+'_expand')(se)
    x = Multiply(name=name+'_se'+'_scale')([x, se])

    # Projection
    x = Conv2D(output_filters, 1, padding='same', use_bias=False, name=name+'_project')(x)
    x = BatchNormalization(name=name+'_bn3')(x)

    # Skip connection (only if input/output shapes match and stride == 1)
    if stride == 1 and input_filters == output_filters:
        x = Add(name=name+'_skip')([shortcut, x])
    
    return x



def eff_net(name, x_size, y_size, n_channels, initializer, reg, batchsize, logits=False):

    #input layers
    model_input = Input(shape=(x_size, y_size, n_channels), name='input')
    rescale = Rescaling(1 / 255, offset=0, name='rescale')(model_input)

    #stem
    conv2d_1 = Conv2D(32, kernel_size=(3, 3),strides=2, padding='same',use_bias=False, kernel_initializer=initializer,
                      kernel_regularizer=reg, name='conv2d_1')(rescale)
    batch_norm1 = BatchNormalization(name='batch1')(conv2d_1)
    act1 = Activation(tf.nn.swish, name='act1')(batch_norm1)

    #stage2
    stage2 = MBConvBlock(act1, 32, 16, kernel_size=3, stride=1, expansion_factor=1, name='stage2_block1')

    #stage3
    stage3_1 = MBConvBlock(stage2, 16, 24, kernel_size=3, stride=1, expansion_factor=6, name='stage3_block1')
    stage3_2 = MBConvBlock(stage3_1, 24, 24, kernel_size=3, stride=2, expansion_factor=6, name='stage3_block2')

    #stage4
    stage4_1 = MBConvBlock(stage3_2, 24, 40, kernel_size=5, stride=1, expansion_factor=6, name='stage4_block1')
    stage4_2 = MBConvBlock(stage4_1, 40, 40, kernel_size=5, stride=2, expansion_factor=6, name='stage4_block2')

    #stage5
    stage5_1 = MBConvBlock(stage4_2, 40, 80, kernel_size=3, stride=1, expansion_factor=6, name='stage5_block1')
    stage5_2 = MBConvBlock(stage5_1, 80, 80, kernel_size=3, stride=1, expansion_factor=6, name='stage5_block2')
    stage5_3 = MBConvBlock(stage5_2, 80, 80, kernel_size=3, stride=2, expansion_factor=6, name='stage5_block3')

    #stage6
    stage6_1 = MBConvBlock(stage5_3, 80, 112, kernel_size=5, stride=1, expansion_factor=6, name='stage6_block1')
    stage6_2 = MBConvBlock(stage6_1, 112, 112, kernel_size=5, stride=1, expansion_factor=6, name='stage6_block2')
    stage6_3 = MBConvBlock(stage6_2, 112, 112, kernel_size=5, stride=1, expansion_factor=6, name='stage6_block3')

    #stage7
    stage7_1 = MBConvBlock(stage6_3, 112, 192, kernel_size=5, stride=1, expansion_factor=6, name='stage7_block1')
    stage7_2 = MBConvBlock(stage7_1, 192, 192, kernel_size=5, stride=1, expansion_factor=6, name='stage7_block2')
    stage7_3 = MBConvBlock(stage7_2, 192, 192, kernel_size=5, stride=1, expansion_factor=6, name='stage7_block3')
    stage7_4 = MBConvBlock(stage7_3, 192, 192, kernel_size=5, stride=2, expansion_factor=6, name='stage7_block4')

    #stage8
    stage8_1 = MBConvBlock(stage7_4, 192, 320, kernel_size=3, stride=1, expansion_factor=6, name='stage8_block1')

    #head
    conv2d_2 = Conv2D(1280, 1, padding='same', use_bias=False, name='conv2d_2')(stage8_1)
    batch_norm2 = BatchNormalization(name='batch2')(conv2d_2)
    act2 = Activation(tf.nn.swish, name='act2')(batch_norm2)

    pooling = GlobalAveragePooling2D(name='pooling')(act2)

    dropout = Dropout(0.2, name='dropoutb')(pooling)

    if logits:
        model_output = Dense(9, activation=None, kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)
    else:
        model_output = Dense(9, activation='softmax', kernel_initializer=initializer, name='output',
                             kernel_regularizer=reg)(dropout)

    model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=name)
    model.build(input_shape=(batchsize, x_size, y_size, n_channels))
    return model





