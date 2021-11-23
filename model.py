import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout
from tensorflow.keras.regularizers import l2

from modules import *


class DAMIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_reg=1e-4):
        super(DAMIN, self).__init__()
        self.maxlen = maxlen

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns   # todo dense是空的，sparse只有item_id

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)  # 1

        # other embedding layers    # user_id
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] in behavior_feature_list]

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        # 多头注意力
        self.multi_att_layer_1 = MultiHeadAttention()
        self.multi_att_layer_2 = MultiHeadAttention()
        self.multi_att_layer_3 = MultiHeadAttention()

        self.bn = BatchNormalization(trainable=True)
        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else ffn_activation)\
             for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        # (None, maxlen)， 不等于零的位置为1 [1, 1, 1, 0, 0],
        # 但如果cate_id恰好是零，就有[1, 0, 1, 0, 0]
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)
        multi_att_mask = tf.cast(tf.equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)

        # other
        if self.dense_len > 0:
            other_info = dense_inputs
        else:
            other_info = self.embed_sparse_layers[0](sparse_inputs[:, 0])
        for i in range(1, self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq, item embedding and category embedding should concatenate
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_num)], axis=-1)   # [b, T, behavior_num]
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.behavior_num)], axis=-1)    # [b, behavior_num]
    
        # att
        user_info, weighted_sum = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (b, T, d * 2)

        # 多头注意力 + 残差
        # user_info1 = self.multi_att_layer_1(user_info, multi_att_mask)
        # user_info = user_info + user_info1
        # user_info2 = self.multi_att_layer_2(user_info, multi_att_mask)
        # user_info = user_info + user_info2
        user_info = self.multi_att_layer_3(user_info, multi_att_mask)       # (b, T, d * 2)

        # todo 原论文没有说明最后一层多头注意力输出的维度，有两种
        # user_info = tf.reshape(user_info, [-1, user_info.shape[1] * user_info.shape[2]])    # [b, T*d*2]
        user_info = tf.reduce_sum(user_info, axis=1)                                        # [b, d*2]

        # concat user_info(att hist), cadidate item embedding, other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)  # [b, emb_size]
            # info_all = tf.concat([user_info, weighted_sum, item_embed, other_info], axis=-1)  # todo 保留din输出
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)
            # info_all = tf.concat([user_info, weighted_sum, item_embed], axis=-1)  # todo 保留din输出

        info_all = self.bn(info_all)


        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len, ), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len, ), dtype=tf.int32)
        seq_inputs = Input(shape=(self.maxlen, self.behavior_num), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_num, ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()


def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 64},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 64},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 64}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DAMIN(features, behavior_list)
    model.summary()


# test_model()