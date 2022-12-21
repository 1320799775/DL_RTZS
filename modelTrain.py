import os

from tensorflow import ConfigProto, GPUOptions, Session
from keras.backend.tensorflow_backend import set_session


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    set_session(Session(config=ConfigProto(gpu_options=
                                           GPUOptions(allow_growth=True))))

import h5py
from keras import backend as K
K.set_image_dim_ordering('th')

import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv1D, concatenate
from keras.layers import BatchNormalization, Dropout, add
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
seed = 7
np.random.seed(seed)

# setting
num_classes = 4096
#----------------------------------------------
# H is signal matrix in the paper with values of 0 or 1, its size is (40000, 4096, 1)
# X_train is simulated real time ZS spectra for training, its size is (32000, 4096, 1)
# X_test is simulated real time ZS spectra for testing, its size is (8000, 4096, 1)
# Y_train is ideal pure shift spectra for training, its size is (32000, 4096, 1)
# Y_test is ideal pure shift spectra for testing, its size is (8000, 4096, 1)
#----------------------------------------------

def CONV_BLOCK(input_x, filters, kernel_size, strides=1, padding='same',shortcut=False):
    x = CONV_BN(input_x, filters, 1)
    x = CONV_BN(x, filters, kernel_size)
    x = CONV_BN(x, filters, 1)
    if shortcut:
        cut = CONV_BN(input_x, filters, kernel_size=kernel_size, strides=strides, padding=padding)
        x = add([x,cut])
        x = BatchNormalization(gamma_initializer='glorot_normal')(x)
        # x = concatenate([x, cut])
        return x
    else:
        x = add([x,input_x])
        x = BatchNormalization(gamma_initializer='glorot_normal')(x)
        # x = concatenate([x, input_x])
        return x

def CONV_BN(x, filters, kernel_size, strides=1, activation=None, kernel_initializer= 'glorot_normal',
            gamma_initializer='glorot_normal', bias_initializer='zeros', padding='same', name=None):
    if name is not None:
        bn_name = name+'_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
               kernel_initializer= kernel_initializer,bias_initializer=bias_initializer,
               padding= padding, name=conv_name)(x)
    x = BatchNormalization(gamma_initializer=gamma_initializer, name=bn_name)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    # x = Dropout(0.2)(x)
    return x

ij0=0
batch_num=0
batch_num1=0
def HADAMARD_YZX(y_true,y_pred):
    ones_tensor = tf.ones([4096, 1])
    ones_tensor = tf.divide(ones_tensor, 1)
    y_true1 = tf.add(y_true, ones_tensor)
    y_pred1 = tf.add(y_pred, ones_tensor)
    a = K.abs(tf.subtract(y_pred1, y_true1))
    a = tf.div(a, y_true1)
    a = K.mean(a)
    global ij0
    global batch_num
    global batch_num1
    if ij0==0:
        Hi=H[batch_num:(batch_num+9),:,:]
        Hi = K.constant(Hi)
        y_true=tf.multiply(y_true,Hi)
        y_pred=tf.multiply(y_pred,Hi)

        ones_tensor = tf.ones([4096, 1])
        ones_tensor = tf.divide(ones_tensor, 1)
        y_true1 = tf.add(y_true, ones_tensor)
        y_pred1 = tf.add(y_pred, ones_tensor)
        b = K.abs(tf.subtract(y_pred1, y_true1))
        b = tf.div(b, y_true1)
        b = K.mean(b)
    else:
        Hi = H[(32000+batch_num1):(32000+batch_num1 + 9), :, :]
        Hi = K.constant(Hi)
        y_true = tf.multiply(y_true, Hi)
        y_pred = tf.multiply(y_pred, Hi)
        ones_tensor = tf.ones([4096, 1])
        ones_tensor = tf.divide(ones_tensor, 1)
        y_true1 = tf.add(y_true, ones_tensor)
        y_pred1 = tf.add(y_pred, ones_tensor)
        b = K.abs(tf.subtract(y_pred1, y_true1))
        b = tf.div(b, y_true1)
        b = K.mean(b)
    return a+b/80

def ACResNet(time_steps, data_dims=1, num_output_channels=1):
    w_init = 'glorot_normal'
    g_init = 'glorot_normal'
    #     filters=128
    filters = 128
    kernel_size = 25

    img_input = Input(shape=(time_steps, data_dims), name='main_input')
    x0 = CONV_BN(img_input, filters, kernel_size=kernel_size)

    x = CONV_BLOCK(x0, filters, kernel_size=kernel_size, shortcut=True)
    # x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    # x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x1 = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x1_1 = add([x1, x0])
    x1_1 = BatchNormalization(gamma_initializer='glorot_normal')(x1_1)

    x = concatenate([x1_1, x0])
    # x=x1_1
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size, shortcut=True)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x2 = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x2_1 = add([x1, x0])
    x2_1 = add([x2_1, x2])
    x2_1 = BatchNormalization(gamma_initializer='glorot_normal')(x2_1)

    x = concatenate([x2_1, x1_1, x0])
    # x=x2_1
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size, shortcut=True)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x3 = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x3_1 = add([x1, x0])
    x3_1 = add([x3_1, x2])
    x3_1 = add([x3_1, x3])
    x3_1 = BatchNormalization(gamma_initializer='glorot_normal')(x3_1)

    x = concatenate([x3_1, x2_1, x1_1, x0])
    # x=x3_1
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size, shortcut=True)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x4 = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x4_1 = add([x1, x0])
    x4_1 = add([x4_1, x2])
    x4_1 = add([x4_1, x3])
    x4_1 = add([x4_1, x4])
    x4_1 = BatchNormalization(gamma_initializer='glorot_normal')(x4_1)

    x = concatenate([x4_1, x3_1, x2_1, x1_1, x0])
    # x=x4_1
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size, shortcut=True)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x5 = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x5_1 = add([x1, x0])
    x5_1 = add([x5_1, x2])
    x5_1 = add([x5_1, x3])
    x5_1 = add([x5_1, x4])
    x5_1 = add([x5_1, x5])
    x5_1 = BatchNormalization(gamma_initializer='glorot_normal')(x5_1)

    x = concatenate([x5_1, x4_1, x3_1, x2_1, x1_1, x0])
    # x=x5_1
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size, shortcut=True)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)
    x = CONV_BLOCK(x, filters, kernel_size=kernel_size)

    x = CONV_BN(x, filters, kernel_size=kernel_size)
    x = Conv1D(num_output_channels, kernel_size=1, activation=None, kernel_initializer=w_init,
                    bias_initializer='zeros', padding='same')(x)

    model = Model(input=img_input, output=x)
    model.compile(loss=HADAMARD_YZX, optimizer='adam', metrics=['accuracy'])
    return model

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, mode='auto')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00011,patience=4,verbose=0,mode='auto')
callback = reduce_lr
model = ACResNet(num_classes, data_dims=1, num_output_channels=1)
callback.set_model(model)

epoch=20
batch_size=10
size_train=X_train.shape[0]
size_test=X_test.shape[0]

for i in range(epoch):
    ij0=0
    for batch_num in range(0,size_train,batch_size):#0,4,8...
        x=X_train[batch_num:(batch_num+9),:,:]
        y=Y_train[batch_num:(batch_num+9),:,:]
        loss=model.train_on_batch(x,y)
    print("loss=",loss)
    ij0=1
    for batch_num1 in range(0, size_test, batch_size):  # 0,4,8...
        x1 = X_test[batch_num1:(batch_num1 + 9), :, :]
        y1 = Y_test[batch_num1:(batch_num1 + 9), :, :]
        val_loss = model.test_on_batch(x1, y1)
    print("val_loss=",val_loss)

model.save('model/model.h5')
