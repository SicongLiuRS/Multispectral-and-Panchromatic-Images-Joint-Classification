# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:12:57 2020

@author: dell
"""

import keras as K
import keras.layers as L
from data_utils import *


class auto_encoder_net(object):
    def __init__(self,MS_shape,dim=1):
        filters = 128
        dilations = [1, 3, 5, 7]

        self.input_spat = L.Input(MS_shape)
        
        self.conv0_0 = L.Conv2D(filters, (dilations[1], dilations[1]), padding='same')(self.input_spat)
        self.conv0_0 = L.BatchNormalization(axis=-1)(self.conv0_0)
        self.conv0_0 = L.Activation('relu')(self.conv0_0)
        
        self.conv0_1 = L.Conv2D(filters, (dilations[1],dilations[1]), padding='same')(self.conv0_0)
        self.conv0_1 = L.BatchNormalization(axis=-1)(self.conv0_1)
        self.conv0_1 = L.Activation('relu')(self.conv0_1)

        self.convup0_1 = L.UpSampling2D((2,2))(self.conv0_1)
        
        self.conv0_2 = L.Conv2D(filters, (dilations[1],dilations[1]), padding='same')(self.convup0_1)
        self.conv0_2 = L.BatchNormalization(axis=-1)(self.conv0_2)
        self.conv0_2 = L.Activation('relu')(self.conv0_2)
        
        self.conv0_3 = L.Conv2D(filters, (dilations[1],dilations[1]), padding='same')(self.conv0_2)
        self.conv0_3 = L.BatchNormalization(axis=-1)(self.conv0_3)
        self.conv0_3 = L.Activation('relu')(self.conv0_3)        

        self.convup0_3 = L.UpSampling2D((2,2))(self.conv0_3 ) 
        
        self.conv0_4 = L.Conv2D(filters, (dilations[2],dilations[2]), padding='same')(self.convup0_3)
        self.conv0_4 = L.BatchNormalization(axis=-1)(self.conv0_4)
        self.conv0_4 = L.Activation('relu')(self.conv0_4)       
        
        self.conv0_5 = L.Conv2D(filters, (dilations[2],dilations[2]), padding='same')(self.conv0_4)
        self.conv0_5 = L.BatchNormalization(axis=-1)(self.conv0_5)
        self.conv0_5 = L.Activation('relu')(self.conv0_5)
        
        self.conv6=L.Conv2D(dim, (3,3), padding='same')(self.conv0_5)
   
        self.model = K.models.Model([self.input_spat], self.conv6)

        opti = K.optimizers.Adam(lr=0.0001)

        self.model.compile(optimizer=opti, loss='mean_squared_error',
                           metrics=[])
        
 
        
class hidden_layer_net(object):
    def __init__(self, input_shape, path=1,dim=1,auto_weights=None):
        """
        input:
            input_shape: input shape of MS
            path: hidden layer branch
        """
        
        P2P = auto_encoder_net(input_shape,dim)
        P2P.model.load_weights(auto_weights)
        P2P.model.trainable = False
        p2p_in = P2P.model.input

        
        conv0 = L.Conv2D(128, (3,3), padding='same')(P2P.conv0_0)
        conv0 = L.BatchNormalization(axis=-1)(conv0)
        conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)

        conv0 = L.Conv2D(256, (3,3), padding='same')(conv0)
        conv0 = L.BatchNormalization(axis=-1)(conv0)
        conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)

        conv0 = L.Conv2D(512, (3,3), padding='same')(conv0)
        conv0 = L.BatchNormalization(axis=-1)(conv0)
        conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
        conv0 = L.GlobalMaxPooling2D()(conv0) 
        conv0 = L.Dropout(0.4)(conv0)
        conv0 = L.Dense(512)(conv0)
        
        conv1 = L.Conv2D(128, (3,3), padding='same')(P2P.conv0_1)
        conv1 = L.BatchNormalization(axis=-1)(conv1)
        conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)

        conv1 = L.Conv2D(256, (3,3), padding='same')(conv1)
        conv1 = L.BatchNormalization(axis=-1)(conv1)
        conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)

        conv1 = L.Conv2D(512, (3,3), padding='same')(conv1)
        conv1 = L.BatchNormalization(axis=-1)(conv1)
        conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)
        conv1 = L.GlobalMaxPooling2D()(conv1) 
        conv1 = L.Dropout(0.4)(conv1)
        conv1 = L.Dense(512)(conv1)
        
        conv2 = L.Conv2D(128, (3,3), padding='same')(P2P.conv0_2)
        conv2 = L.BatchNormalization(axis=-1)(conv2)
        conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
        conv2 = L.MaxPool2D(pool_size=(2, 2),padding='same')(conv2)
        conv2 = L.Conv2D(256, (3,3), padding='same')(conv2)
        conv2 = L.BatchNormalization(axis=-1)(conv2)
        conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
        conv2 = L.Conv2D(512, (3,3), padding='same')(conv2)
        conv2 = L.BatchNormalization(axis=-1)(conv2)
        conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
        conv2 = L.GlobalMaxPooling2D()(conv2)
        conv2 = L.Dropout(0.4)(conv2)
        conv2 = L.Dense(512)(conv2)        
        
        conv3 = L.Conv2D(128, (3,3), padding='same')(P2P.conv0_3)
        conv3 = L.BatchNormalization(axis=-1)(conv3)
        conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
        conv3 = L.MaxPool2D(pool_size=(2, 2),padding='same')(conv3)
        conv3 = L.Conv2D(256, (3,3), padding='same')(conv3)
        conv3 = L.BatchNormalization(axis=-1)(conv3)
        conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
        conv3 = L.Conv2D(512, (3,3), padding='same')(conv3)
        conv3 = L.BatchNormalization(axis=-1)(conv3)
        conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
        conv3 = L.GlobalMaxPooling2D()(conv3)
        conv3 = L.Dropout(0.4)(conv3)
        conv3 = L.Dense(512)(conv3)
        
        conv4 = L.Conv2D(128, (3,3), padding='same')(P2P.conv0_4)
        conv4 = L.BatchNormalization(axis=-1)(conv4)
        conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
        conv4 = L.MaxPool2D(pool_size=(2, 2),padding='same')(conv4)
        conv4 = L.Conv2D(256, (3,3), padding='same')(conv4)
        conv4 = L.BatchNormalization(axis=-1)(conv4)
        conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
        conv4 = L.MaxPool2D(pool_size=(2, 2),padding='same')(conv4)
        conv4 = L.Conv2D(512, (3,3), padding='same')(conv4)
        conv4 = L.BatchNormalization(axis=-1)(conv4)
        conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
        conv4 = L.GlobalMaxPooling2D()(conv4) 
        conv4 = L.Dropout(0.4)(conv4)
        conv4 = L.Dense(512)(conv4)         
        
        conv5 = L.Conv2D(128, (3,3), padding='same')(P2P.conv0_5)
        conv5 = L.BatchNormalization(axis=-1)(conv5)
        conv5 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv5)
        conv5 = L.MaxPool2D(pool_size=(2, 2),padding='same')(conv5)
        conv5 = L.Conv2D(256, (3,3), padding='same')(conv5)
        conv5 = L.BatchNormalization(axis=-1)(conv5)
        conv5 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv5)
        conv5 = L.MaxPool2D(pool_size=(2, 2),padding='same')(conv5)
        conv5 = L.Conv2D(512, (3,3), padding='same')(conv5)
        conv5 = L.BatchNormalization(axis=-1)(conv5)
        conv5 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv5)
        conv5 = L.GlobalMaxPooling2D()(conv5)        
        conv5 = L.Dropout(0.4)(conv5)
        conv5 = L.Dense(512)(conv5)
        
        ##--------------------------------
        logits0 = L.Dense(NUM_CLASS, activation='softmax')(conv0)
        logits1 = L.Dense(NUM_CLASS, activation='softmax')(conv1)
        logits2 = L.Dense(NUM_CLASS, activation='softmax')(conv2)
        logits3 = L.Dense(NUM_CLASS, activation='softmax')(conv3)
        logits4 = L.Dense(NUM_CLASS, activation='softmax')(conv4)
        logits5 = L.Dense(NUM_CLASS, activation='softmax')(conv5)

        if path==0:
            logits=logits0
        if path==1:
            logits=logits1
        if path==2:
            logits=logits2
        if path==3:
            logits=logits3
        if path==4:
            logits=logits4
        if path==5:
            logits=logits5
            
        self.model = K.models.Model([p2p_in], [logits])
        # optm = K.optimizers.Adam(lr=0.001)
        optm=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
        self.model.compile(optimizer=optm,
                           loss='categorical_crossentropy', metrics=['acc'])
        

class merge_hidden_layer_net(object):
    def __init__(self, input_shape,dim=1,auto_weights = None):
        
        HL1=hidden_layer_net(input_shape, 1,dim, auto_weights)
        HL1.model.load_weights(SavePathWeight+'/branch_weigth1.h5')
        HL1.model._layers.pop()
        HL1.model._layers.pop()
        HL1.model._layers.pop()
        HL1.model.trainable = False
        # HL1.model.trainable = True
        p2p_out1 = HL1.model.layers[-1].output 
        p2p_in1=HL1.model.input


        HL3=hidden_layer_net(input_shape, 3,dim, auto_weights)
        HL3.model.load_weights(SavePathWeight+'/branch_weigth3.h5')
        HL3.model._layers.pop()
        HL3.model._layers.pop()
        HL3.model._layers.pop()
        HL3.model.trainable = False
        # HL3.model.trainable = True
        p2p_out3 = HL3.model.layers[-1].output 
        p2p_in3=HL3.model.input


        HL5=hidden_layer_net(input_shape, 5,dim, auto_weights)
        HL5.model.load_weights(SavePathWeight+'/branch_weigth5.h5')
        HL5.model._layers.pop()
        HL5.model._layers.pop()
        HL5.model._layers.pop()
        HL5.model.trainable = False
        # HL5.model.trainable = True
        p2p_out5 = HL5.model.layers[-1].output 
        p2p_in5=HL5.model.input

        merge = L.concatenate([p2p_out1,p2p_out3,p2p_out5],axis=-1)
        merge=L.Dropout(0.4)(merge)

        merge=L.Dense(512)(merge)
        merge=L.Dense(512)(merge)
        merge = L.advanced_activations.LeakyReLU(alpha=0.2)(merge)
        logits=L.Dense(NUM_CLASS,activation='softmax')(merge)

        self.model = K.models.Model([p2p_in1,p2p_in3,p2p_in5], [logits])
        self.model.summary()
        optm=K.optimizers.SGD(lr=0.0001,momentum=0.99,decay=1e-4)
        self.model.compile(optimizer=optm,loss='categorical_crossentropy', metrics=['acc'])
