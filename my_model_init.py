from tensorflow import keras
import tensorflow as tf


def my_IoU(y_pred, y_true):
    """
        IoU metric used in semantic segmentation.
    """
    inter = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    uni = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - inter
    return tf.reduce_mean(inter / uni)


def my_recompiler(model, lr=0.0001):
    """
        Compiling a model if necessary.
    """
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy', my_IoU])
    model.summary()


def conv2d_batch_norm_leaky(x, filters, size=3, strides=1, use_bias=False):
    """
        A combination of 2D convulation, batch normalization and leaky relu layers.
    """
    x = keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=use_bias,
                            kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def conv2d_batch_norm_relu(x, filters, size=3, strides=1, use_bias=False):
    """
        A combination of 2D convulation, batch normalization and relu layers.
    """
    x = keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=use_bias,
                            kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def my_small_net(input_shape=224, channels=3, output_shape=224, classes=10):
    """
        My model implementation with tensorflow-keras.
    """
    inputs = keras.Input(shape=(input_shape, input_shape, channels), name='inputs')

    x = conv2d_batch_norm_leaky(inputs, 8, 3)

    x = conv2d_batch_norm_leaky(x, 8, 3)
    x1 = conv2d_batch_norm_leaky(x, 8, 1)
    x = keras.layers.MaxPooling2D(pool_size=2)(x1)

    x = conv2d_batch_norm_leaky(x, 26, 3)
    x2 = conv2d_batch_norm_leaky(x, 28, 1)
    x = keras.layers.MaxPooling2D(pool_size=2)(x2)

    x = conv2d_batch_norm_leaky(x, 64, 3)
    x = conv2d_batch_norm_leaky(x, 104, 1)

    x = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = keras.layers.Concatenate()([x2, x])
    x = conv2d_batch_norm_leaky(x, 24, 3)
    x = conv2d_batch_norm_leaky(x, 28, 1)

    x = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = keras.layers.Concatenate()([x1, x])
    x = conv2d_batch_norm_leaky(x, 8, 3)
    x = conv2d_batch_norm_leaky(x, 8, 1)

    output = keras.layers.Conv2D(classes+1, 3, name='output', padding='same')(x)

    output = keras.layers.Reshape((output_shape, output_shape, classes+1), name='output-0')(output)
    output = keras.layers.Softmax(axis=-1)(output)

    my_model = keras.Model(inputs, output, name='my_jet_net')
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    my_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy', my_IoU])
    my_model.summary()
    return my_model
