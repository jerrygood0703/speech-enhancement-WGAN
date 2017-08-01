import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np


def pad_numbers(in_width, filter_size, stride):
    if stride == 2:
        out_width = np.ceil(float(in_width) / float(stride))
    else:
        out_width = in_width
    p = int(max(stride*(out_width-1)-in_width+filter_size, 0))
    if p%2==0:
        return [p//2, p//2]
    else:
        return [(p//2)+1, p//2]

def pad_numbers_plus(in_width, filter_size, stride):
    if stride == 2:
        out_width = np.ceil(float(in_width) / float(stride))
    else:
        out_width = in_width-1
    p = int(max(stride*(out_width-1)-in_width+filter_size, 0))
    if p%2==0:
        return [p//2, p//2]
    else:
        return [(p//2)+1, p//2]

def conv2d(batch_input, out_channels, filter_shape, strides):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[1]
        in_height = batch_input.get_shape()[2]
        in_width = batch_input.get_shape()[3]
        kh, kw = filter_shape
        _, _, sh, sw = strides
        w = tf.get_variable(name="w",
                            shape=[kh, kw, in_channels, out_channels], 
                            dtype=tf.float32, 
                            initializer=tf.random_normal_initializer(0, 0.02))
        # b = tf.get_variable(name='b',
        #                     shape=[out_channels],
        #                     initializer=tf.constant_initializer(0.0))
        
        ph = pad_numbers(int(in_height), kh, sh)
        pw = pad_numbers(int(in_width), kw, sw)

        padded_input = tf.pad(batch_input, [[0, 0], [0, 0], ph, pw], mode="REFLECT")
        # conv = tf.nn.bias_add(tf.nn.conv2d(padded_input, w, strides, padding="VALID", data_format="NCHW"), b, data_format="NCHW")
        conv = tf.nn.conv2d(padded_input, w, strides, padding="VALID", data_format="NCHW")
        return conv

def deconv_up(batch_input, out_channels, filter_shape, strides):
    with tf.variable_scope("deconv"):
        in_channels = batch_input.get_shape()[1]
        kh, kw = filter_shape
        _, _, sh, sw = strides

        up_layer = keras.layers.UpSampling2D(size=(2,1), data_format="channels_first")(batch_input)
        up_height = up_layer.get_shape()[2]
        up_width = up_layer.get_shape()[3]

        w = tf.get_variable(name="w",
                            shape=[kh, kw, in_channels, out_channels], 
                            dtype=tf.float32, 
                            initializer=tf.random_normal_initializer(0, 0.02))
        # b = tf.get_variable(name='b',
        #                     shape=[out_channels],
        #                     initializer=tf.constant_initializer(0.0))
        
        ph = pad_numbers(int(up_height), kh, sh)
        pw = pad_numbers(int(up_width), kw, sw)

        padded_input = tf.pad(up_layer, [[0, 0], [0, 0], ph, pw], mode="REFLECT")
        # conv = tf.nn.bias_add(tf.nn.conv2d(padded_input, w, strides, padding="VALID", data_format="NCHW"), b, data_format="NCHW")
        conv = tf.nn.conv2d(padded_input, w, strides, padding="VALID", data_format="NCHW")

        return conv

def deconv2D_up(batch_input, out_channels, filter_shape, strides):
    with tf.variable_scope("deconv"):
        in_channels = batch_input.get_shape()[1]
        kh, kw = filter_shape
        _, _, sh, sw = strides

        up_layer = keras.layers.UpSampling2D(size=(2,2), data_format="channels_first")(batch_input)
        up_height = up_layer.get_shape()[2]
        up_width = up_layer.get_shape()[3]

        w = tf.get_variable(name="w",
                            shape=[kh, kw, in_channels, out_channels], 
                            dtype=tf.float32, 
                            initializer=tf.random_normal_initializer(0, 0.02))
        # b = tf.get_variable(name='b',
        #                     shape=[out_channels],
        #                     initializer=tf.constant_initializer(0.0))
        
        ph = pad_numbers_plus(int(up_height), kh, sh)
        pw = pad_numbers(int(up_width), kw, sw)

        padded_input = tf.pad(up_layer, [[0, 0], [0, 0], ph, pw], mode="REFLECT")
        # conv = tf.nn.bias_add(tf.nn.conv2d(padded_input, w, strides, padding="VALID", data_format="NCHW"), b, data_format="NCHW")
        conv = tf.nn.conv2d(padded_input, w, strides, padding="VALID", data_format="NCHW")

        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def selu(x, name="selu"):
    """ When using SELUs you have to keep the following in mind:
    # (1) scale inputs to zero mean and unit variance
    # (2) use SELUs
    # (3) initialize weights with stddev sqrt(1/n)
    # (4) use SELU dropout
    """
    with tf.name_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def prelu(x, name='prelu'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # make one alpha per feature
        alpha = tf.get_variable('alpha', in_shape[-1],
                                initializer=tf.constant_initializer(0.),
                                dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
    return pos + neg

def activation(x,name='relu'):
    if name == 'prelu':
        return prelu(x)
    elif name == 'lrelu':
        return lrelu(x,0.2)
    elif name == 'tanh':
        return tf.nn.tanh(x)
    elif name == 'relu' :
        return tf.nn.relu(x)
    else:
        return None

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, [0, 2, 3], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def layernorm(x, axis, name):
    '''
    Layer normalization (Ba, 2016)
    J: Z-normalization using all nodes of the layer on a per-sample basis.

    Input:
        `x`: channel_first/NCHW format! (or fully-connected)
        `axis`: list
        `name`: must be assigned
    
    Example:
        # axis = [1, 2, 3]
        # x = tf.random_normal([64, 3, 10, 10])
        # name = 'D_layernorm'
    
    Return:
        (x - u)/s * scale + offset

    Source: 
        https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/layernorm.py
    '''
    mean, var = tf.nn.moments(x, axis, keep_dims=True)
    n_neurons = x.get_shape().as_list()[axis[0]]
    offset = tf.get_variable(
        name+'.offset',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.zeros_initializer
    )
    scale = tf.get_variable(
        name+'.scale',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.ones_initializer
    )
    return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-5)

def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        c = np.log(2 * np.pi)
        var = tf.exp(log_var)
        x_mu2 = tf.square(x - mu)   # [Issue] not sure the dim works or not?
        x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, -1)   # keep_dims=True,
        return log_prob

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

