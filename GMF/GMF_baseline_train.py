# -*- coding: utf-8 -*-
"""
Created on Sept 24 21:02:00 2017

@author: Dahua Gan
"""

import sys
import pickle as pkl
import pandas as pd
import numpy as np
import os
import pickle as pkl
import scipy.sparse as sp
from time import time
import multiprocessing as mp
import sys
import math
from time import time
import heapq
import setproctitle

from keras.initializers import VarianceScaling
from keras.layers.merge import concatenate, multiply
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras.backend.tensorflow_backend as tK
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

title = 'GMF_baseline_train@gandahua'
setproctitle.setproctitle(title)

# ------ load in data ------

train_buy = pkl.load(open('/data/stu/gandahua/dataset_new/train_buy.pkl', 'rb'))
train_cart = pkl.load(open('/data/stu/gandahua/dataset_new/train_cart.pkl', 'rb'))
train_ipv = pkl.load(open('/data/stu/gandahua/dataset_new/train_ipv.pkl', 'rb'))
data_info = pkl.load(open('/data/stu/gandahua/dataset_new/data_info.pkl', 'rb'))


# testRatings = pkl.load(open('/data/stu/gandahua/dataset/testRatings.pkl', 'rb'))
# testNegatives = np.unique(sp.find(train_buy)[1]).tolist()



# ------ definition of functions ------

def model_init():
    with tK.tf.device('/gpu:0'):
        config = tK.tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tK.tf.Session(config=config)
    model_ipv = get_GMF(num_users=num_users + 1, num_items=num_items + 1,
                        layer_em=num_embedding, layer_reg=layer_reg,
                        enable_dropout=enable_dropout, dropout_val=dropout_val)
    model_cart = get_GMF(num_users=num_users + 1, num_items=num_items + 1,
                         layer_em=num_embedding, layer_reg=layer_reg,
                         enable_dropout=enable_dropout, dropout_val=dropout_val)
    model_buy = get_GMF(num_users=num_users + 1, num_items=num_items + 1,
                        layer_em=num_embedding, layer_reg=layer_reg,
                        enable_dropout=enable_dropout, dropout_val=dropout_val)
    model_ipv.compile(optimizer=Adagrad(), loss='binary_crossentropy')
    model_cart.compile(optimizer=Adagrad(), loss='binary_crossentropy')
    model_buy.compile(optimizer=Adagrad(), loss='binary_crossentropy')
    return model_ipv, model_cart, model_buy


def get_GMF(num_users, num_items, layer_em=16, layer_reg=[0, 0],
            enable_dropout=False, dropout_val=0.5):
    assert len(layer_reg) == 2
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    # Embedding layer
    GMF_Embedding_User = Embedding(input_dim=num_users,
                                   output_dim=layer_em,
                                   input_length=1,
                                   name="GMF_embedding_user",
                                   embeddings_initializer=VarianceScaling(scale=0.01, distribution='normal'),
                                   embeddings_regularizer=l2(layer_reg[0]))
    GMF_Embedding_Item = Embedding(input_dim=num_items,
                                   output_dim=layer_em,
                                   input_length=1,
                                   name='GMF_embedding_item',
                                   embeddings_initializer=VarianceScaling(scale=0.01, distribution='normal'),
                                   embeddings_regularizer=l2(layer_reg[0]))
    # post-embedding manipulation
    GMF_user_latent = Flatten()(GMF_Embedding_User(user_input))
    GMF_item_latent = Flatten()(GMF_Embedding_Item(item_input))
    # element-wise product
    GMF_product = multiply([GMF_user_latent, GMF_item_latent], name='GMF_product')
    # Final prediction layer
    if enable_dropout:
        GMF_product = Dropout(dropout_val)(GMF_product)
    prediction = Dense(1, activation='sigmoid',
                       kernel_initializer='lecun_uniform',
                       kernel_regularizer=l2(layer_reg[1]),
                       name='prediction')(GMF_product)
    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)
    return model  # ------ parameter setting ------


# model parameters
num_users = train_buy.shape[0]
num_items = train_buy.shape[1]

print('start!')

# ------ initial evaluation ------

num_embedding = 16

print('Embedding %d starts' % (num_embedding))
layer_reg = [0, 0]
enable_dropout = False
dropout_val = 0

# training parameters
num_negatives = 4
num_epochs_ipv = 40
num_epochs_cart = 40
num_epochs_buy = 40
batch_size = [2048, 128, 64]
verbose = 1
evaluation_threads = 1

# ------ initial evaluation ------
model_ipv, model_cart, model_buy = model_init()
model_buy.save('/data/stu/gandahua/CL/GMF/model/GMF_init_em_%d_neg_%d.h5'
               % (num_embedding, num_negatives), overwrite=True)

# ------ training ------

loss_ipv = []
loss_cart = []
loss_buy = []
val_loss_ipv = []
val_loss_cart = []
val_loss_buy = []

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# (user, item, labels) = pkl.load(open('/data/stu/gandahua/dataset_new/training_set/train_ipv_neg_%d_epoch_%d.pkl'%(num_negatives, epoch+1), 'rb'))



print('------ training on buy ------')

for epoch in range(num_epochs_buy):
    print('[epoch %d] of buy starts' % (epoch + 1))
    t1 = time()
    # load training set     
    (user, item, labels) = pkl.load(open(
        '/data/stu/gandahua/dataset_new/training_set/train_buy_all_items_neg_%d_epoch_%d.pkl' % (
            num_negatives, epoch + 1), 'rb'))
    # training buy
    hist = model_buy.fit([user, item], labels, batch_size=batch_size[2], epochs=1, verbose=1, shuffle=True,
                         validation_split=0.1, callbacks=[early_stopping])
    loss = hist.history['loss'][0]
    val_loss = hist.history['val_loss'][0]
    t2 = time()
    print('Iteration %d [%.1f s], loss = %.4f' % (epoch + 1, t2 - t1, loss))
    loss_buy.append(loss)
    val_loss_buy.append(val_loss)
    model_buy.save('/data/stu/gandahua/CL/GMF/model/buy_model/GMF_buy_em_%d_epoch_%d_neg_%d.h5'
                   % (num_embedding, epoch + 1, num_negatives), overwrite=True)

pd.DataFrame(loss_buy).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_buy_em_%d_neg_%d_loss.csv'
                              % (num_embedding, num_negatives))
pd.DataFrame(val_loss_buy).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_buy_em_%d_neg_%d_val_loss.csv'
                                  % (num_embedding, num_negatives))

print('GMF baseline training finished!')

assert len(loss_buy) == num_epochs_buy
assert len(val_loss_buy) == num_epochs_buy

print('------ training on cart ------')

for epoch in range(num_epochs_cart):
    print('[epoch %d] of cart starts' % (epoch + 1))
    t1 = time()
    # load training set     
    (user, item, labels) = pkl.load(open(
        '/data/stu/gandahua/dataset_new/training_set/train_cart_all_items_neg_%d_epoch_%d.pkl' % (
            num_negatives, epoch + 1), 'rb'))
    # training cart     
    hist = model_cart.fit([user, item], labels, batch_size=batch_size[1], epochs=1, verbose=0, shuffle=True,
                          validation_split=0.1, callbacks=[early_stopping])
    loss = hist.history['loss'][0]
    val_loss = hist.history['val_loss'][0]
    t2 = time()
    print('Iteration %d [%.1f s], loss = %.4f, val_loss = %.4f' % (epoch + 1, t2 - t1, loss, val_loss))
    loss_cart.append(loss)
    val_loss_cart.append(val_loss)
    model_cart.save('/data/stu/gandahua/CL/GMF/model/cart_model/GMF_cart_em_%d_epoch_%d_neg_%d.h5'
                    % (num_embedding, epoch + 1, num_negatives), overwrite=True)

pd.DataFrame(loss_cart).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_cart_em_%d_neg_%d_loss.csv'
                               % (num_embedding, num_negatives))
pd.DataFrame(val_loss_cart).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_cart_em_%d_neg_%d_val_loss.csv'
                                   % (num_embedding, num_negatives))

assert len(loss_cart) == num_epochs_cart
assert len(val_loss_cart) == num_epochs_cart

print('------ training on ipv ------')

for epoch in range(num_epochs_ipv):
    print('[epoch %d] of ipv starts' % (epoch + 1))
    t1 = time()
    # load training set     
    (user, item, labels) = pkl.load(
        open('/data/stu/gandahua/dataset_new/training_set/train_ipv_neg_%d_epoch_%d.pkl' % (num_negatives, epoch + 1),
             'rb'))
    # training ipv  
    hist = model_ipv.fit([user, item], labels, batch_size=batch_size[0], epochs=1, verbose=1, shuffle=True,
                         validation_split=0.1, callbacks=[early_stopping])
    loss = hist.history['loss'][0]
    val_loss = hist.history['val_loss'][0]
    t2 = time()
    print('Iteration %d [%.1f s], loss = %.4f, val_loss = %.4f' % (epoch + 1, t2 - t1, loss, val_loss))
    loss_ipv.append(loss)
    val_loss_ipv.append(val_loss)
    model_ipv.save('/data/stu/gandahua/CL/GMF/model/ipv_model/GMF_ipv_em_%d_epoch_%d_neg_%d.h5'
                   % (num_embedding, epoch + 1, num_negatives), overwrite=True)

pd.DataFrame(loss_ipv).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_ipv_em_%d_neg_%d_loss.csv'
                              % (num_embedding, num_negatives))
pd.DataFrame(val_loss_ipv).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_ipv_em_%d_neg_%d_val_loss.csv'
                                  % (num_embedding, num_negatives))

assert len(loss_ipv) == num_epochs_ipv
assert len(val_loss_ipv) == num_epochs_ipv
