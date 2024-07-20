# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 13:59:14 2024


Keras 2.3.1
Tensorflow 2.0.0
Python 3.6
numpy 1.19.2


@author: fes_map
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Add, Activation, Conv3D, BatchNormalization, AveragePooling3D, MaxPooling3D
from tensorflow.keras.layers import Flatten, Dense, LocallyConnected1D, Input, Reshape, Concatenate, Lambda
import tensorflow.keras.backend as K
from numpy import ceil

from .Network import Network
from .layers.Subtract import Subtract


class EP3FEN(Network):
    def __init__(self, input_shape, n_category, n_filters=(4, 8, 16), kernel_size=3, pool_size=3, local_sensing=7, n_stack=3):
        super().__init__("EP3FEN", input_shape, n_category)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.local_sensing = local_sensing
        self.n_stack = n_stack
        
        self.build_model()
    
    
    def extract(self, x, pos=None, flag="center"):
        '''
        x: 5-d tensor (n_sp, 1, n_row, n_col, n_band)
        extract the spectrum on the position of "pos"
        
        if 'b_expand' is True,  the output will has the same shape as 'x'
        '''
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]
        
        if pos == None:
            # default is the center
            pos = (n_row // 2, n_col // 2)
        
        center = x[:, :, pos[0]:pos[0] + 1, pos[1]:pos[1] + 1, :]
        center = Reshape((n_band, 1))(center)
        
        return center
        
    
    def EP3C(self, x, n_filters, kernel_size):
        r = Conv3D(n_filters, (1, 1, 1), strides=1, padding="same", data_format="channels_first")(x)
        x = Conv3D(n_filters, (kernel_size, 1, 1), strides=1, padding="same", data_format="channels_first")(x)
        x = Conv3D(n_filters, (1, kernel_size, 1), strides=1, padding="same", data_format="channels_first")(x)
        x = Conv3D(n_filters, (1, 1, kernel_size), strides=1, padding="same", data_format="channels_first")(x)
        
        o = Add()([x, r])
        o = BatchNormalization(axis=1)(o)
        o = Activation("relu")(o)
        
        return o
        
    
    def SBS(self, x, size):
        n_channel, n_row, n_col, n_band = K.int_shape(x)[1:]
        # semantic salient
        a = AveragePooling3D(pool_size=(1, 1, size), strides=1, padding="same", data_format="channels_first")(x)
        def _subtract(_x):
            return _x[0] - _x[1]
        d = Lambda(_subtract, output_shape=(n_channel, n_row, n_col, n_band))([x, a])
        # d = Subtract(output_dim=(n_channel, n_row, n_col, n_band))([x, a])
        Ss = MaxPooling3D(pool_size=(1, 1, size), strides=(1, 1, size), padding="same", data_format="channels_first")(d)
        
        # absolute salient
        Sa = MaxPooling3D(pool_size=(1, 1, size), strides=(1, 1, size), padding="same", data_format="channels_first")(x)
        
        S = Add()([Ss, Sa])

        return S
    
    
    def SDC(self, c, local_sensing):
        input_shape = K.int_shape(c)[1:]
        sdc = Sequential()
        sdc.add(LocallyConnected1D(1, kernel_size=local_sensing, strides=local_sensing//2, activation="relu", input_shape=input_shape))
        
        return sdc(c)
        
    
    def EP3FEM(self, x, c, n_filters, kernel_size, pool_size, local_sensing):
        f = self.EP3C(x, n_filters, kernel_size)
        s = self.SBS(f, pool_size)
        if K.int_shape(s)[-2] >= 2:
            s = AveragePooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding="same", data_format="channels_first")(s)
        e = self.SDC(c, local_sensing)
        
        return s, e
        
        
    def build_model(self):
        I = Input(shape=self.input_shape)
        
        x = I
        c = self.extract(x)
        
        print(1)
        f, e = self.EP3FEM(x, c, self.n_filters[0], self.kernel_size, self.pool_size, self.local_sensing)
        for i in range(1, self.n_stack):
            print(i+1)
            f, e = self.EP3FEM(f, e, self.n_filters[i], self.kernel_size, self.pool_size, self.local_sensing)
        
        f = Flatten()(f)
        e = Flatten()(e)
        
        fe = Concatenate(axis=1)([f, e])
        y = Dense(self.n_category, activation="softmax")(fe)
        
        self.model = Model(inputs=I, outputs=y, name="EP3FEN")
        
        
        