import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow_compression as tfc
import resnet

def resblock(input, IC, OC, name):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = tf.keras.layers.Conv2D(filters=np.minimum(IC, OC), kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name=name + 'l1')(l1)

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = tf.keras.layers.Conv2D(filters=OC, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name=name + 'l2')(l2)

    if IC != OC:
        input = tf.keras.layers.Conv2D(filters=OC, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(), name=name + 'map')(input)

    return input + l2


def MC(input):

    m1 = tf.compat.v1.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc1')

    m2 = resblock(m1, 64, 64, name='mc2')

    m3 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(m2)

    m4 = resblock(m3, 64, 64, name='mc4')

    m5 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(m4)

    m6 = resblock(m5, 64, 64, name='mc6')

    m7 = resblock(m6, 64, 64, name='mc7')

    m8 = tf.image.resize(m7, [2 * tf.shape(input=m7)[1], 2 * tf.shape(input=m7)[2]])

    m8 = m4 + m8

    m9 = resblock(m8, 64, 64, name='mc9')

    m10 = tf.image.resize(m9, [2 * tf.shape(input=m9)[1], 2 * tf.shape(input=m9)[2]])

    m10 = m2 + m10

    m11 = resblock(m10, 64, 64, name='mc11')

    m12 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc12')(m11)

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = tf.keras.layers.Conv2D(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc13')(m12)

    return m13

