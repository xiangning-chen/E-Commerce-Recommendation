# -*- coding: utf-8 -*-
"""
Created on Jul 31 21:02:00 2017

@author: Dahua
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
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

setproctitle.setproctitle('GMF_test_buy@gandahua')

# ------ load in data ------

train_buy = pkl.load(open('/data/stu/gandahua/dataset_new/train_buy.pkl', 'rb'))
testRatings = pkl.load(open('/data/stu/gandahua/dataset_new/testRatings.pkl', 'rb'))
testNegatives = pkl.load(open('/data/stu/gandahua/dataset_new/testNegatives.pkl', 'rb'))
data_info = pkl.load(open('/data/stu/gandahua/dataset_new/data_info.pkl', 'rb'))


def evaluate_model(model, testRatings, testNeg, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _K
    _model = model
    _testRatings = testRatings
    _K = K
    hits, ndcgs = [], []
    for idx in range(len(_testRatings)):
        rating = _testRatings[idx]
        # items = testNeg
        u = rating[0]
        gtItem = rating[1]
        items = [gtItem] + np.arange(7977).tolist()
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u, dtype='int32')
        predictions = _model.predict([users, np.array(items)], batch_size=len(users))
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        # Evaluate top rank list
        # return the keys(item id) with K largest predict values for user(idx)
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)
        hits.append(hr)
        ndcgs.append(ndcg)
        # if (idx % 1000) == 0:
        #     print(idx)
    return (hits, ndcgs)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def get_MTL(num_users, num_items,
            enable_train=[True, True, True],
            EM_layers=16, EM_layers_reg=0,
            FC_layers_1=[8], FC_layers_2=[8],
            which='None'):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    # Embedding layer
    Embedding_User = Embedding(input_dim=num_users,
                               output_dim=int(EM_layers),
                               input_length=1,
                               name="embedding_user",
                               trainable=enable_train[0],
                               embeddings_initializer=VarianceScaling(scale=0.01, distribution='normal'),
                               embeddings_regularizer=l2(EM_layers_reg))
    Embedding_Item = Embedding(input_dim=num_items,
                               output_dim=int(EM_layers),
                               input_length=1,
                               name='embedding_item',
                               trainable=enable_train[0],
                               embeddings_initializer=VarianceScaling(scale=0.01, distribution='normal'),
                               embeddings_regularizer=l2(EM_layers_reg))
    # post-embedding manipulation
    user_latent = Flatten()(Embedding_User(user_input))
    item_latent = Flatten()(Embedding_Item(item_input))
    # FC layers
    FC_input = multiply([user_latent, item_latent], name='product_vec', trainable=enable_train[0])
    # ------ training on ipv data ------
    output_ipv = Dense(1, name="output_ipv",
                       trainable=enable_train[0])(FC_input)
    output_ipv = BatchNormalization()(output_ipv)
    output_ipv = Activation('sigmoid')(output_ipv)
    # ----------------------------------------------------- #
    for i in range(len(FC_layers_1) + 1):
        if i == len(FC_layers_1):
            break
        else:
            if i == 0:
                FC_1_layer = Dense(FC_layers_1[i],
                                   trainable=enable_train[1],
                                   name="FC_1_layer_%d" % (i + 1))(FC_input)
            else:
                FC_1_layer = Dense(FC_layers_1[i],
                                   trainable=enable_train[1],
                                   name="FC_1_layer_%d" % (i + 1))(FC_1_layer)
            FC_1_layer = BatchNormalization()(FC_1_layer)
            FC_1_layer = Activation('relu')(FC_1_layer)
    # ------ training on cart data ------
    output_cart = Dense(1, name="output_cart",
                        trainable=enable_train[1])(FC_1_layer)
    output_cart = BatchNormalization()(output_cart)
    output_cart = Activation('sigmoid')(output_cart)
    # ----------------------------------------------------- #
    for i in range(len(FC_layers_2) + 1):
        if i == len(FC_layers_2):
            break
        else:
            if i == 0:
                FC_2_layer = Dense(FC_layers_2[i],
                                   trainable=enable_train[2],
                                   name="FC_2_layer_%d" % (i + 1))(FC_1_layer)
            else:
                FC_2_layer = Dense(FC_layers_2[i],
                                   trainable=enable_train[2],
                                   name="FC_2_layer_%d" % (i + 1))(FC_2_layer)
            FC_2_layer = BatchNormalization()(FC_2_layer)
            FC_2_layer = Activation('relu')(FC_2_layer)
    # ------ training on buy data ------
    output_buy = Dense(1, name="output_buy",
                       trainable=enable_train[2])(FC_2_layer)
    output_buy = BatchNormalization()(output_buy)
    output_buy = Activation('sigmoid')(output_buy)
    if which == 'ipv':
        model = Model(inputs=[user_input, item_input],
                      outputs=[output_ipv])
    elif which == 'cart':
        model = Model(inputs=[user_input, item_input],
                      outputs=[output_cart])
    elif which == 'buy':
        model = Model(inputs=[user_input, item_input],
                      outputs=[output_buy])
    else:
        model = Model(inputs=[user_input, item_input],
                      outputs=[output_ipv, output_cart, output_buy])
    return model


# ------ evaluation starts ------

with tK.tf.device('/gpu:0'):
    config = tK.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    tK.tf.Session(config=config)

# --- parameters ---

num_negatives = 4
evaluation_threads = 1
num_epochs_ipv = 40
num_epochs_cart = 40
num_epochs_buy = 40
num_embedding = 16

num_users = train_buy.shape[0]
num_items = train_buy.shape[1]
enable_train_ipv = [True, False, False]
enable_train_cart = [False, True, False]
enable_train_buy = [False, False, True]
enable_train = [True, True, True]
embedding_reg = 0

FC_layers_1 = [16]
FC_layers_2 = [8]

topK = 100

hr_ipv_list = []
ndcg_ipv_list = []
hr_cart_list = []
ndcg_cart_list = []
hr_buy_list = []
ndcg_buy_list = []

print('Evaluation of GMF baseline with embedding %d starts' % (num_embedding))

# --- buy ---

for epoch in range(num_epochs_buy):
    print('[epoch %d] of buy starts' % (epoch + 1))
    model_buy = load_model('/data/stu/gandahua/CL/GMF/model/buy_model/GMF_buy_em_%d_epoch_%d_neg_%d.h5'
                           % (num_embedding, epoch + 1, num_negatives))
    t1 = time()
    (hits, ndcgs) = evaluate_model(model_buy,
                                   testRatings,
                                   testNegatives,
                                   topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    t2 = time()
    hr_buy_list.append(hr)
    ndcg_buy_list.append(ndcg)
    print('buy epoch %d: topK = %d, HR = %.4f, NDCG = %.4f, [%d s]' % (epoch + 1, topK, hr, ndcg, t2 - t1))

best_hr = np.array(hr_buy_list).max()
best_ndcg = np.array(ndcg_buy_list).max()

print('[GMF_buy with embedding %d] Best HR = %.4f, Best NDCG = %.4f' % (num_embedding, best_hr, best_ndcg))

pd.DataFrame(hr_buy_list).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_buy_em_%d_neg_%d_topK_%d_hr.csv'
                                 % (num_embedding, num_negatives, topK))
pd.DataFrame(ndcg_buy_list).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_buy_em_%d_neg_%d_topK_%d_ndcg.csv'
                                   % (num_embedding, num_negatives, topK))

# --- cart ---

for epoch in range(num_epochs_cart):
    print('[epoch %d] of cart starts' % (epoch + 1))
    model_cart = load_model('/data/stu/gandahua/CL/GMF/model/cart_model/GMF_cart_em_%d_epoch_%d_neg_%d.h5'
                            % (num_embedding, epoch + 1, num_negatives))
    t1 = time()
    (hits, ndcgs) = evaluate_model(model_cart,
                                   testRatings,
                                   testNegatives,
                                   topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    t2 = time()
    hr_cart_list.append(hr)
    ndcg_cart_list.append(ndcg)
    print('cart epoch %d: topK = %d, HR = %.4f, NDCG = %.4f, [%d s]' % (epoch + 1, topK, hr, ndcg, t2 - t1))

best_hr = np.array(hr_cart_list).max()
best_ndcg = np.array(ndcg_cart_list).max()

print('[GMF_cart with embedding %d] Best HR = %.4f, Best NDCG = %.4f' % (num_embedding, best_hr, best_ndcg))

pd.DataFrame(hr_cart_list).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_cart_em_%d_neg_%d_topK_%d_hr.csv'
                                  % (num_embedding, num_negatives, topK))
pd.DataFrame(ndcg_cart_list).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_cart_em_%d_neg_%d_topK_%d_ndcg.csv'
                                    % (num_embedding, num_negatives, topK))

# --- ipv ---

for epoch in range(num_epochs_ipv):
    print('[epoch %d] of ipv starts' % (epoch + 1))
    model_ipv = load_model('/data/stu/gandahua/CL/GMF/model/ipv_model/GMF_ipv_em_%d_epoch_%d_neg_%d.h5'
                           % (num_embedding, epoch + 1, num_negatives))
    t1 = time()
    (hits, ndcgs) = evaluate_model(model_ipv,
                                   testRatings,
                                   testNegatives,
                                   topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    t2 = time()
    hr_ipv_list.append(hr)
    ndcg_ipv_list.append(ndcg)
    print('ipv epoch %d: topK = %d, HR = %.4f, NDCG = %.4f, [%d s]' % (epoch + 1, topK, hr, ndcg, t2 - t1))

best_hr = np.array(hr_ipv_list).max()
best_ndcg = np.array(ndcg_ipv_list).max()

print('[GMF_ipv with embedding %d] Best HR = %.4f, Best NDCG = %.4f' % (num_embedding, best_hr, best_ndcg))

pd.DataFrame(hr_ipv_list).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_ipv_em_%d_neg_%d_topK_%d_hr.csv'
                                 % (num_embedding, num_negatives, topK))
pd.DataFrame(ndcg_ipv_list).to_csv('/home/stu/gandahua/CL/plot/GMF/GMF_ipv_em_%d_neg_%d_topK_%d_ndcg.csv'
                                   % (num_embedding, num_negatives, topK))
