# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:49:06 2018

@author: wushaowu
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge,PassiveAggressiveRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error

#读入数据：
traindata= pd.read_csv('traindata.csv')
testdata= pd.read_csv('testdata.csv')

#缺失值处理：
traindata['COMMCONTENT'].fillna('_na_',inplace=True)
testdata['COMMCONTENT'].fillna('_na_',inplace=True)

y=(traindata['COMMLEVEL']-1).astype(int)
y.to_csv('label.csv',index=None)

tf =TfidfVectorizer(ngram_range=(1,2),min_df=1, max_df=0.9,\
                    use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc= tf.fit_transform(traindata['COMMCONTENT'])
test_term_doc= tf.transform(testdata['COMMCONTENT'])

#########################pac model###########################################
print('pac stacking')
stack_train = np.zeros((len(y), 1))
stack_test = np.zeros((len(testdata), 1))
score_va = 0
n_folds=5
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n_folds, random_state=2018)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    pac = PassiveAggressiveRegressor(random_state=2018) 
    pac.fit(trn_term_doc[tr], y[tr]) 
    score_va = pac.predict(trn_term_doc[va]) 
    score_te = pac.predict(test_term_doc)
    print("model mse:",mean_squared_error(score_va,y[va]))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
df_stack_train = pd.DataFrame()
df_stack_test = pd.DataFrame()
for i in range(stack_test.shape[1]):
    df_stack_train['pac_reg_{}'.format(i)] = stack_train[:, i]
    df_stack_test['pac_reg_{}'.format(i)] = stack_test[:, i]
df_stack_train.to_csv('pac_stack_train_feat_reg.csv', index=None, encoding='utf8')
df_stack_test.to_csv('pac_stack_test_feat_reg.csv', index=None, encoding='utf8')

#########################Ridge model#######################################
n_folds=5
print('ridge stacking')
stack_train = np.zeros((len(y), 1))
stack_test = np.zeros((len(testdata), 1))
score_va = 0
for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=n_folds, random_state=2018)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    ridge =Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250,\
          normalize=False, tol=0.01,random_state=2018)
    ridge.fit(trn_term_doc[tr], y[tr])
    score_va = ridge.predict(trn_term_doc[va])
    score_te = ridge.predict(test_term_doc)
    print("model mse:",mean_squared_error(score_va,y[va]))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
df_stack_train = pd.DataFrame()
df_stack_test = pd.DataFrame()
for i in range(stack_test.shape[1]):
    df_stack_train['ridge_reg_{}'.format(i)] = stack_train[:, i]
    df_stack_test['ridge_reg_{}'.format(i)] = stack_test[:, i]
df_stack_train.to_csv('ridge_stack_train_feat_reg.csv', index=None, encoding='utf8')
df_stack_test.to_csv('ridge_stack_test_feat_reg.csv', index=None, encoding='utf8')
