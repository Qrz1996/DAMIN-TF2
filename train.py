import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from model import DAMIN
from utils import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = '/public/home/qrz/data/Amazon_electronic/precessed/remap.pkl'
    maxlen = 40

    print(tf.test.is_gpu_available())

    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    
    # embed_dim = 8
    embed_dim = 64
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'      # attention层里面没用到激活函数
    ffn_activation = 'sigmoid'        # 层数较少，可试试sigmoid

    learning_rate = 0.001
    batch_size = 4096
    epochs = 20
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, test = create_amazon_electronic_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = test
    # test_X, test_y = test
    # ============================Build Model==========================
    # feature_columns = [[],
    #                        [sparseFeature('item_id', item_count, embed_dim),
    #                         ]]  # sparseFeature('cate_id', cate_count, embed_dim)
    # behavior_list = ['item_id']  # , 'cate_id'
    model = DAMIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
        ffn_activation, maxlen, dnn_dropout)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # # ===========================Fit==============================
    model.fit(      # 模型的输入是一个列表，列表中有几个array，每个array是一个特征
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(val_X, val_y),
        batch_size=batch_size,
    )
    # ===========================Test==============================
    # print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])

    # 自定义训练过程
    # optimizer = Adam(learning_rate=1.)
    # # loss = tf.keras.losses.BinaryCrossentropy()
    # metrics = AUC()
    #
    # # @tf.function
    # def train_step(x, y):
    #     with tf.GradientTape() as tape:
    #         y = tf.expand_dims(y, axis=-1)
    #         pred = model(x)
    #         loss_value = tf.keras.losses.binary_crossentropy(y, pred)
    #         loss_value = tf.reduce_mean(loss_value)
    #     grads = tape.gradient(loss_value, model.trainable_variables)
    #     print(loss_value)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     return loss_value
    #
    # for epoch in range(epochs):
    #     for step, (x, y) in iter(DataInput(train_X, train_y, batch_size=batch_size)):
    #         # variable_names = model.trainable_variables[0]
    #         # print(variable_names, '更新前值：', variable_names)
    #         loss_value = train_step(x, y)
    #         if step % 100 == 0:
    #             print('step:{} loss:{}'.format(step, loss_value))
    #
    #         if step % 1000 == 0:
    #             # test_auc = model.evaluate(test_X, test_Y, batch_size=test_batch_size)[1]
    #             # print('epoch:{} step:{} train loss:{:.4f}'.format(epoch, step, test_auc))
    #             metrics.reset_states()
    #             for _, (x, y) in iter(DataInput(test_X, test_y, batch_size=512)):
    #                 pred = model(x)
    #                 metrics.update_state(y, pred)
    #             print('epoch:{} step:{} train auc:{:.4f}'.format(epoch, step, metrics.result()))