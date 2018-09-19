# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:46:57 2018
@author: wushaowu
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter

#读入特征数据：
pac_stack_train_feat_reg= pd.read_csv('pac_stack_train_feat_reg.csv',encoding='utf8')
pac_stack_test_feat_reg= pd.read_csv('pac_stack_test_feat_reg.csv',encoding='utf8')

ridge_stack_train_feat_reg= pd.read_csv('ridge_stack_train_feat_reg.csv',encoding='utf8')
ridge_stack_test_feat_reg= pd.read_csv('ridge_stack_test_feat_reg.csv',encoding='utf8')

lstm_stack_train_feat= pd.read_csv('lstm_stack_train_feat.csv',encoding='utf8')
lstm_stack_test_feat= pd.read_csv('lstm_stack_test_feat.csv',encoding='utf8')

#读入训练标签：
label = pd.read_csv('label.csv',header=None)
label=label[0]

# 合并特征集
train_feature = pd.concat([
                        ridge_stack_train_feat_reg,
                        pac_stack_train_feat_reg,
                        lstm_stack_train_feat,
                        ], axis=1)
test_feature = pd.concat([
                        ridge_stack_test_feat_reg,
                        pac_stack_test_feat_reg,
                        lstm_stack_test_feat,
                        ], axis=1)


from sklearn.cross_validation import StratifiedKFold
print('xgb stacking')
stack_train = np.zeros((len(label),1))
stack_test = np.zeros((len(test_feature),1))
score_va = 0
n_folds=5
for i, (tr, va) in enumerate(StratifiedKFold(label, n_folds=n_folds, random_state=1)):
   
    params={'booster':'gbtree',
        'eta':0.1, 
        'max_depth':4,
        'objective':'reg:linear',
        'random_seed':2018,
        'eval_metric':'rmse',
        }
    dtrain=xgb.DMatrix(train_feature.ix[tr],label[tr])
    dval=xgb.DMatrix(train_feature.ix[va],label[va])
    watchlist = [(dtrain,'train'),(dval,'val')]
    num_rounds=200
    model=xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)
    score_va = model.predict(xgb.DMatrix(train_feature.ix[va]))
    score_te = model.predict(xgb.DMatrix(test_feature))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds

#预测值转换成123：
zz=[]
for i in stack_test[:,0]:
    if i<1.5:
        zz.append(1)
    elif i>1.9:
        zz.append(3)
    else:
        zz.append(2)
print('最终类别统计：',Counter(zz))
#结果保存：
np.savetxt("dsjyycxds_semifinal.txt",np.array(zz).astype(int),fmt="%d")
