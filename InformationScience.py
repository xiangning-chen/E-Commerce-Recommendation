# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from scipy.io import mmread
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata
import json
from scipy import io, sparse
import random
import math
import re
from zhon.hanzi import punctuation
import jieba
import time
from sklearn.preprocessing import normalize


class SENP():
    def __init__(self):
        self.user_num = 0
        self.nega_num = 0
        self.k = 100
        self.data_path = '../../data/SENP/{}'

    def gen_item_word_mtx(self):
        data = []
        users_list = []
        item_id_list = []
        punc_en = "[’!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~]+"
        punc_ch = "。 ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙" \
                  "〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        # stopwords = self.get_stopwords()
        # read stop words from file
        with open(self.data_path.format('stopwords'), 'r') as f:
            stopwords = [line.strip().decode('utf-8') for line in f]

        with open(self.data_path.format('expsea_item_info'), 'r') as f:
            for line in f:
                data.append(list(eval(line)))
            data_pd = pd.DataFrame(data, columns=['item_id', 'title', 'product_desc', 'brand_name', 'product_props',
                                                  'props_sex', 'props_age', 'brand_desc', 'cate1_name', 'cate2_name', 'cate3_name'])
            # drop duplicate items
            item_list = list(data_pd['item_id'])
            item_idx = []
            item_set = set()
            for item in item_list:
                if item not in item_set:
                    item_idx.append(item_list.index(item))
                    item_set.add(item)
            data_pd = data_pd.iloc[item_idx, :]

            # word split
            cols = [1, 2, 3, 5, 7, 8, 9, 10]
            for i in xrange(data_pd.shape[0]):
                user_temp = []
                for j in cols:
                    temp1 = re.sub(ur"[%s]+" % punc_ch.decode("utf-8"), '', data_pd.iloc[i][j])
                    temp2 = re.sub(ur"[%s]+" % punctuation, '', temp1)
                    temp3 = re.sub(punc_en, '', temp2)
                    age = re.findall(u'[\d-]*\u5c81', temp3) + re.findall(u'[\d-]*\u4e2a\u6708', temp3)
                    if age:
                        temp3 = re.sub(u'[\d-]*\u5c81', '', temp3)
                        temp3 = re.sub(u'[\d-]*\u4e2a\u6708', '', temp3)
                        user_temp += age
                    user_temp += list(jieba.cut(temp3))
                # product_props
                temp2 = []
                temp1 = [item.split(':')[-1] for item in data_pd.iloc[i][4].split(';')]
                for item in temp1:
                    if ',' in item:
                        temp2 += item.split(',')
                    elif item != u'':
                        temp2.append(item)
                user_temp += temp2
                # props_age
                user_temp += re.sub(u'\u5468', '', data_pd.iloc[i][6]).split(',')

                # delete stopwords
                user_temp = [item for item in user_temp if item not in stopwords]
                # aggregate
                users_list.append(user_temp)
                item_id_list.append(data_pd.iloc[i, 0])

        self.item_ids = item_id_list
        print 'item num: ' + str(len(self.item_ids))
        # union of words
        all_words = list(set([item for sublist in users_list for item in sublist]))
        print 'word num: ' + str(len(all_words))
        words_dict = dict(zip(all_words, range(len(all_words))))
        item_word_mtx = np.zeros((len(users_list), len(all_words)), int)
        for user_id, item in enumerate(users_list):
            for word in item:
                item_word_mtx[user_id, words_dict[word]] += 1

        self.ip = sparse.csr_matrix(item_word_mtx)
        print 'item_word matrix created'

    def gen_matrix(self):
        # read records
        data = []
        with open(self.data_path.format('expsea_search_buy'), 'r') as f:
            for line in f:
                data.append(list(eval(line)))
            data_pd = pd.DataFrame(data, columns=['user_id', 'item_id', 'search_word', 'action_type', 'action_count',
                                                  'timestamp', 'date'])
        # users
        user_ids = list(data_pd['user_id'].drop_duplicates())
        user_dict = dict(zip(user_ids, range(len(user_ids))))
        print 'user num: ' + str(len(user_ids))

        # items
        item_dict = dict(zip(self.item_ids, range(len(self.item_ids))))
        # print 'item num: ' + str(len(self.item_ids))

        # initial
        buy_user_item_mtx = np.zeros((len(user_ids), len(self.item_ids)), int)
        buy_train_mtx = np.zeros((len(user_ids), len(self.item_ids)), float)
        search_train_mtx = np.zeros((len(user_ids), len(self.item_ids)), float)

        for user_name, group in data_pd.groupby('user_id'):
            for action_type, subgroup in group.groupby('action_type'):
                item_list = list(subgroup['item_id'])
                if action_type == 'buy':
                    # test
                    buy_test_item = data_pd.iloc[subgroup['timestamp'].argmax(), 1]
                    # train
                    item_list.remove(buy_test_item)
                    for item in item_list:
                        buy_train_mtx[user_dict[user_name], item_dict[item]] += 1
                        # buy_user_item_mtx[user_dict[user_name], item_dict[item]] = 2
                    # test
                    buy_user_item_mtx[user_dict[user_name], item_dict[buy_test_item]] = 1
                else:
                    for item in item_list:
                        search_train_mtx[user_dict[user_name], item_dict[item]] += 1
        # # negative samples
        # for i in xrange(buy_user_item_mtx.shape[0]):
        #     buy_user_item_mtx[i, random.sample(list(np.where(buy_user_item_mtx[i, :] == 0)[0]), self.nega_num)] = -1
        #     buy_user_item_mtx[i, list(np.where(buy_user_item_mtx[i, :] == 2)[0])] = 0

        self.user_it = sparse.csr_matrix(buy_user_item_mtx)

        print 'calculating TF-IDF ...'
        # TF
        self.ip = normalize(self.ip, norm='l1', axis=1)

        # IDF
        idf = np.zeros((1, self.ip.shape[1]), float)
        for i in xrange(self.ip.shape[1]):
            idf[0, i] = math.log(float(len(self.item_ids)) / (len(self.ip[:, i].data) + 1) + 1)

        self.ip = sparse.csr_matrix(self.ip.toarray() * idf)

        # matrix multiplexing
        print 'buy matrix multiplication'
        self.user_bp = sparse.csr_matrix(buy_train_mtx).dot(self.ip)

        print 'search matrix multiplication'
        self.user_sp = sparse.csr_matrix(search_train_mtx).dot(self.ip)

    def evaluate_sample(self, use_search=True, combine_method='rank', combine_param=0.5, write_file=True):
        # user_item = sparse.csr_matrix([[0, 1, -1, -1], [1, 0, -1, -1], [0, -1, 1, -1]])
        # item_profile = sparse.random(4, 7, density=0.25)
        # user_s_profile = sparse.random(3, 7, density=0.25)
        # user_b_profile = sparse.random(3, 7, density=0.25)

        user_item_test = self.user_it.nonzero()
        idx = np.where(np.roll(user_item_test[0], 1) != user_item_test[0])[0]
        idx = np.append(idx, user_item_test[0].shape[0])

        result = dict()
        for k in range(len(idx) - 1):
            start = idx[k]
            end = idx[k + 1]
            item_idx = user_item_test[1][start:end]

            chosen_sample_index = self.user_it[user_item_test[0][start], item_idx]
            positive_item = chosen_sample_index.multiply(chosen_sample_index > 0).nonzero()[1]

            item_profile_chosen = self.ip.tocsr()[item_idx, :]
            user_s_profile_chosen = self.user_sp.tocsr()[user_item_test[0][start], :]
            user_b_profile_chosen = self.user_bp.tocsr()[user_item_test[0][start], :]
            b_score = cosine_similarity(item_profile_chosen, user_b_profile_chosen)
            b_rank = rankdata(-b_score, method='min')
            s_score = cosine_similarity(item_profile_chosen, user_s_profile_chosen)
            s_rank = rankdata(-s_score, method='min')
            #print(user_item_test[0][start])
            #print(positive_item)
            result[k] = dict()
            result[k]['right'] = int(positive_item[0])
            result[k]['b_rank'] = b_rank.tolist()
            result[k]['s_rank'] = s_rank.tolist()
            result[k]['b_score'] = b_score.tolist()
            result[k]['s_score'] = s_score.tolist()

            if self.user_num % 100 == 0:
                print(self.user_num)
            self.user_num += 1

        with open(self.data_path.format('result_' + str(self.nega_num) + '.json'), 'w') as fw:
            fw.write(json.dumps(result, indent=4))
        return result

    def evaluate_sample2(self):
        result = dict()
        for k in range(self.user_it.shape[0]):

            b_score = cosine_similarity(self.ip, self.user_bp.tocsr()[k, :])
            b_rank = rankdata(-b_score, method='min')
            s_score = cosine_similarity(self.ip, self.user_sp.tocsr()[k, :])
            s_rank = rankdata(-s_score, method='min')

            result[k] = dict()
            result[k]['right'] = int(self.user_it[k, :].nonzero()[1][0])
            result[k]['b_rank'] = b_rank.tolist()
            result[k]['s_rank'] = s_rank.tolist()
            result[k]['b_score'] = b_score.tolist()
            result[k]['s_score'] = s_score.tolist()

            if self.user_num % 100 == 0:
                print(self.user_num)
            self.user_num += 1

        with open(self.data_path.format('result_' + str(self.nega_num) + '.json'), 'w') as fw:
            fw.write(json.dumps(result, indent=4))
        return result

    def test(self):
        # res = self.evaluate()
        # to do
        r_ndcg = []
        s_ndcg = []
        b_r_ndcg = []
        with open(self.data_path.format('result_' + str(self.nega_num) + '.json'), 'r') as f_in:
            res = json.load(f_in)
        for user in res.items():
            r_rank = rankdata(np.array(user[1]['s_rank']) + np.array(user[1]['b_rank']), method='min')
            s_rank = rankdata(-(np.array(user[1]['s_score']) + np.array(user[1]['b_score'])), method='min')
            # buy
            b_r_rank = user[1]['b_rank']
            if b_r_rank[user[1]['right']] < self.k:
                b_r_ndcg.append(np.log2(2) / np.log(2 + b_r_rank[user[1]['right']]))
            else:
                b_r_ndcg.append(0)
            # buy & search
            if r_rank[user[1]['right']] < self.k:
                r_ndcg.append(np.log2(2) / np.log(2 + r_rank[user[1]['right']]))
            else:
                r_ndcg.append(0)

            if s_rank[user[1]['right']] < self.k:
                s_ndcg.append(np.log2(2) / np.log(2 + s_rank[user[1]['right']]))
            else:
                s_ndcg.append(0)

        return [np.array(b_r_ndcg).mean(), np.array(r_ndcg).mean(), np.array(s_ndcg).mean()]
        # print 'buy - ndcg: ' + str(np.array(b_r_ndcg).mean())
        # print 'buy & search -  rank ndcg: ' + str(np.array(r_ndcg).mean())
        # print 'buy & search - score ndcg: ' + str(np.array(s_ndcg).mean())

    def evaluate(self):
        r_ndcg = []
        r_hr = []
        s_ndcg = []
        s_hr = []
        b_r_ndcg = []
        b_r_hr = []
        user_num = float(self.user_it.shape[0])
        for k in range(self.user_it.shape[0]):
            if self.user_num % 100 == 0:
                print(self.user_num)
            self.user_num += 1

            b_score = cosine_similarity(self.ip, self.user_bp.tocsr()[k, :])
            b_rank = rankdata(-b_score, method='min')
            s_score = cosine_similarity(self.ip, self.user_sp.tocsr()[k, :])
            s_rank = rankdata(-s_score, method='min')
            right_id = int(self.user_it[k, :].nonzero()[1][0])

            r_rank_all = rankdata(s_rank + b_rank, method='min')
            s_rank_all = rankdata(-(s_score + b_score), method='min')

            # hr
            b_r_hr.append(1 if b_rank[right_id] < self.k else 0)
            r_hr.append(1 if r_rank_all[right_id] < self.k else 0)
            s_hr.append(1 if s_rank_all[right_id] < self.k else 0)

            # ndcg
            b_r_ndcg.append(np.log2(2) / np.log(2 + b_rank[right_id]) if b_rank[right_id] < self.k else 0)
            r_ndcg.append(np.log2(2) / np.log(2 + r_rank_all[right_id]) if r_rank_all[right_id] < self.k else 0)
            s_ndcg.append(np.log2(2) / np.log(2 + s_rank_all[right_id]) if s_rank_all[right_id] < self.k else 0)

        hr = [sum(b_r_hr)/user_num, sum(r_hr)/user_num, sum(s_hr)/user_num]
        ndcg = [np.array(b_r_ndcg).mean(), np.array(r_ndcg).mean(), np.array(s_ndcg).mean()]
        return [hr, ndcg]


if __name__ == '__main__':
    t_start = time.time()
    senp = SENP()
    senp.gen_item_word_mtx()
    t1 = time.time()
    senp.gen_matrix()
    t2 = time.time()
    result = senp.evaluate()
    t3 = time.time()
    print 'preprosess time: ' + str(t1 - t_start)
    print 'generate matrix time: ' + str(t2 - t1)
    print 'evaluate time: ' + str(t3 - t2)
    print 'HR: ' + str(result[0])
    print 'NDCG: ' + str(result[1])
