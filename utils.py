import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=40):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):  # groupby用法
        pos_list = hist['item_id'].tolist()

        def gen_neg():  # 生成负样本
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i - 1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:  # 最后一个最为测试数据
                # test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])    # pos_list是item_id列表，cate_list是按照物品id升序排列的cate列表
                # test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])    # pos_list[i]找到物品id，依据物品id找到cate_id
                test_data.append([user_id, hist_i, [pos_list[i], cate_list[pos_list[i]]],
                                  1])  # pos_list是item_id列表，cate_list是按照物品id升序排列的cate列表
                test_data.append(
                    [user_id, hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])  # pos_list[i]找到物品id，依据物品id找到cate_id
            # elif i == len(pos_list) - 2:    # 倒数第二个作为验证数据
            #     # val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
            #     # val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            #     val_data.append([user_id, hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
            #     val_data.append([user_id, hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            else:
                # train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                # train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                train_data.append([user_id, hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([user_id, hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    # feature columns，
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        sparseFeature('cate_id', cate_count, embed_dim),
                        sparseFeature('user_id', user_count, embed_dim * 2),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    # behavior_list = ['item_id']  # , 'cate_id'
    behavior_list = ['item_id', 'cate_id']

    # shuffle
    random.shuffle(train_data)
    # random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'target_item', 'label'])
    # val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    # [dense_inputs, sparse_inputs, seq_inputs, item_inputs]
    train_X = [np.array([0.] * len(train)), np.array(train['user_id'].tolist()),  # 是一个列表
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    # val_X = [np.array([0] * len(val)),np.array(val['user_id'].tolist()),
    #          pad_sequences(val['hist'], maxlen=maxlen),
    #          np.array(val['target_item'].tolist())]
    # val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array(test['user_id'].tolist()),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')

    # for i in range(len(train_X)):
    #     print(train_X[i].shape)
    return feature_columns, behavior_list, (train_X, train_y), (test_X, test_y)


# create_amazon_electronic_dataset('raw_data/remap.pkl')
class DataInput:
    def __init__(self, train_X, train_Y, batch_size):
        self.batch_size = batch_size
        self.train_X = train_X
        self.train_Y = train_Y
        self.epoch_size = len(self.train_X[0]) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.train_X[0]):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        start_index = self.i * self.batch_size
        end_index = min((self.i + 1) * self.batch_size, len(self.train_X[0]))
        self.i += 1

        x = [np.array(self.train_X[i][start_index: end_index].tolist()) for i in range(len(self.train_X))]
        x[0] = np.expand_dims(x[0], axis=-1)
        x[1] = np.expand_dims(x[1], axis=-1)
        y = self.train_Y[start_index: end_index]
        # for i in range(len(x)):
        #     print(x[i])

        return self.i, (x, y)


def buid_amazon_electronic_dataset(data_path, embed_dim=64, maxlen=40):
    with open(data_path, 'rb') as f:
        train_set = pickle.load(f)
        # print('train_set shape', train_set[0])      # (114002, [23550, 23428], 9480, 0)
        test_set = pickle.load(f)
        # print('test_set shape', test_set[0])
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

        train_set = pd.DataFrame(train_set, columns=['user_id', 'hist', 'target_item', 'label'])
        test_set = pd.DataFrame(test_set, columns=['user_id', 'hist', 'target_item', 'label'])

        train_set['target_item'] = train_set['target_item'].apply(lambda x: [x, cate_list[x]])
        test_set['target_item'] = test_set['target_item'].apply(lambda x: [x, cate_list[x]])

        train_set['hist'] = train_set['hist'].apply(lambda x: [[x[i], cate_list[x[i]]] for i in range(len(x))])
        test_set['hist'] = test_set['hist'].apply(lambda x: [[x[i], cate_list[x[i]]] for i in range(len(x))])

        train_X = [np.array([0.] * len(train_set)),
                   np.array(train_set['user_id'].tolist()),
                   pad_sequences(train_set['hist'], maxlen=maxlen, padding='post'),
                   np.array(train_set['target_item'].to_list())]
        train_Y = train_set['label'].values
        test_x = [np.array([0.] * len(test_set)),
                  np.array(test_set['user_id'].tolist()),
                  pad_sequences(test_set['hist'], maxlen=maxlen, padding='post'),
                  np.array(test_set['target_item'].to_list())]
        test_Y = test_set['label'].values

        feature_columns = [[],
                           [
                               sparseFeature('item_id', item_count, embed_dim),
                               sparseFeature('cate_id', cate_count, embed_dim),
                               sparseFeature('user_id', user_count, embed_dim * 2)
                           ]]

        behavior_feature_list = ['item_id', 'cate_id']

        # for i in range(len(train_X)):
        #     print(train_X[i].shape)

        return feature_columns, behavior_feature_list, cate_count, (train_X, train_Y), (test_x, test_Y)
