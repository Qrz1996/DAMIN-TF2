import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense, PReLU


class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):
        """
        """
        super(Attention_Layer, self).__init__()

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2) 
        # value: hist items  (None, seq_len, d * 2) 
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # 计算权重
        weights = 1 / tf.reduce_sum((q - k) ** 2)       # [b, seq_len]

        paddings = tf.ones_like(weights) * (-2 ** 32 + 1)  # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, weights)  # (None, seq_len)
        # [1 , 0, 1, 0, 0] 其中第二个会被替换成-2^32+1, 最后得分就为0了

        # softmax
        weights = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(weights, axis=1)  # None, 1, seq_len)

        weighted_sum = tf.matmul(outputs, v + q)  # (None, 1, d * 2)
        weighted_sum = tf.squeeze(weighted_sum, axis=1)   # [b, d*2]
        v = v + q   # [b, T, d*2]

        outputs = tf.reshape(weights, [-1, v.shape[1], 1]) * v      # [b, T, d*2]

        return outputs, weighted_sum


class MultiHeadAttention(Layer):
    def __init__(self, d_model=128, h=4):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        # Num of heads
        self.h = h
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.wo = Dense(d_model)
    def scaled_dot_product_attention(self, q, k, v, mask=None):

        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)  # d_model // h

        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            dk)  # (..., q_length, d_k) . (..., d_k, k_lengh) = (..., q_length, k_lengh)
        # print('attention_scores.shape', attention_scores.shape)

        # [b, T] ==> [b, 1, 1, T]
        mask = mask[:, np.newaxis, np.newaxis, :]

        if mask is not None:
            attention_scores += (mask * -1e30)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        out = tf.matmul(attention_weights, v)  # (..., q_length, k_lengh) . (k_lengh, d_v) = (..., q_length, d_v)

        return out, attention_weights

    def splitting_head(self, x):
        # batch_size = tf.shape(x)[0]
        # length = tf.shape(x)[1]
        # d_model = tf.shape(x)[2]
        batch_size = x.shape[0]
        length = x.shape[1]
        d_model = x.shape[2]

        # assert d_model % self.h == 0

        hd_v = d_model // self.h

        x = tf.reshape(x, [-1, length, self.h, hd_v])

        xs = tf.transpose(x, [0, 2, 1, 3])  # (..., h, length, hd_v)
        # print('xs.shape', xs.shape)

        return xs

    def call(self, input, mask=None):
        batch_size = tf.shape(input)[0]
        qw = self.wq(input)  # (..., q_length, d_model)
        kw = self.wk(input)  # (..., k_lengh, d_model)
        vw = self.wv(input)  # (..., k_lengh, d_model)

        # Splitting Head
        heads_qw = self.splitting_head(qw)  # (..., h, q_length, hd_v)
        heads_kw = self.splitting_head(kw)  # (..., h, k_lengh, hd_v)
        heads_vw = self.splitting_head(vw)  # (..., h, k_lengh, hd_v)
        # print('heads_vw.shape',heads_vw.shape)

        # Do Attention
        # attention_weights shape: # (..., h, q_length, k_lengh)
        # out shape: # (..., h, q_length, hd_v)

        out, attention_weights = self.scaled_dot_product_attention(heads_qw, heads_kw, heads_vw, mask)

        # Transpose out back to # (..., q_length, d_model)

        out = tf.transpose(out, [0, 2, 1, 3])  # (..., q_length, h, hd_v)

        # out = tf.reshape(out, (batch_size, tf.shape(qw)[1], self.d_model))  # (..., q_length, d_model)
        out = tf.reshape(out, (-1, qw.shape[1], self.d_model))  # (..., q_length, d_model)

        final = self.wo(out)  # (..., q_length, d_model)

        return final

class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x