# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:13:30 2018

@author: wushaowu
"""
import os
import pandas as pd 
import numpy as np
np.random.seed(2018)
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import word2vec
import gensim
from keras.preprocessing import text, sequence
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it
        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

def model_train(save_model_file,n_dim,data):
    # word2vec模型训练
    # model_file_name为训练语料的路径,save_model为保存模型名
    # 模型训练，生成词向量
    sentences = [[word for word in str(document).split(' ')] for document in data]
    model = gensim.models.Word2Vec(sentences, size=n_dim)  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file)
    
train_name = "traindata.csv" #训练数据路径
predict_name = "testdata.csv" #测试数据路径
save_model_name = 'Word300.model' #word2vec模型

###w2v的特征维度
max_features = 30000
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
embed_size = 300
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

#读入数据
train = pd.read_csv(train_name)
train_row = train.shape[0]
df_train = train
predict = pd.read_csv(predict_name)
df_test = predict

if not os.path.exists(save_model_name): #判断word2vec模型是否存在
    model_train(save_model_name,EMBEDDING_DIM,list(df_train['COMMCONTENT'])+list(df_test['COMMCONTENT']))
else:
    print('此训练模型已经存在，不用再次训练')

##加载word2vec模型
w2v_model = word2vec.Word2Vec.load(save_model_name)

##准备数据
X_train =df_train['COMMCONTENT']
X_test =df_test['COMMCONTENT']
X_train.fillna('_na_',inplace=True)
X_test.fillna('_na_',inplace=True)

tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
#训练标签：
y =df_train['COMMLEVEL'].values-1


##找出了lstm权重
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
print('Total %s word vectors.' % nb_words)

###计算权重
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= max_features: 
        continue
    else :
        try:
            embedding_vector = w2v_model[word]
        except KeyError:
            continue
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector


def lstm_model():
    """attention lstm model"""
    embedding_layer = Embedding(nb_words,
                      EMBEDDING_DIM,
                      weights=[embedding_matrix],
                      input_length=MAX_SEQUENCE_LENGTH,
                      trainable=False)
    lstm_layer = LSTM(64, dropout=rate_drop_lstm,\
                      recurrent_dropout=rate_drop_lstm,return_sequences=True)
    
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') #输入层
    embedded_sequences= embedding_layer(inp) #嵌入层
    x = lstm_layer(embedded_sequences) #lstmc层
    merged = Attention(MAX_SEQUENCE_LENGTH)(x) #attention机制层
    merged = Dropout(0.5)(merged)
    outp = Dense(1,activation="linear")(merged)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mse'])
    return model
##########################stacking###############################################
from sklearn.cross_validation import StratifiedKFold
print('lstm stacking')
stack_train = np.zeros((len(y),1))
stack_test = np.zeros((len(test),1))
score_va = 0
n_folds=5
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n_folds, random_state=1)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    model = lstm_model()
    early_stopping =EarlyStopping(monitor='val_loss', patience=2)
    STAMP = 'simple_lstm_w2v_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
    bst_model_path =STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    batch_size = 64
    epochs = 20
    hist = model.fit(x_train[tr], y[tr],
                     validation_data=(x_train[va], y[va]),
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=[early_stopping,model_checkpoint])
    model.load_weights(bst_model_path)
    score_va = model.predict((x_train[va]))
    score_te = model.predict(test, batch_size=256, verbose=1)
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
df_stack_train = pd.DataFrame()
df_stack_test = pd.DataFrame()
for i in range(stack_test.shape[1]):
    df_stack_train['lstm_classfiy{}'.format(i)] = stack_train[:, i]
    df_stack_test['lstm_classfiy{}'.format(i)] = stack_test[:, i]
df_stack_train.to_csv('lstm_stack_train_feat.csv', index=None, encoding='utf8')
df_stack_test.to_csv('lstm_stack_test_feat.csv', index=None, encoding='utf8')
print(str(n_folds)+'_folds_stacking特征已保存\n')