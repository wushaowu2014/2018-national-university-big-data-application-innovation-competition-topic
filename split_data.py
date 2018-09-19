# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:22:41 2018

@author: shaowu
"""
import jieba
import pandas as pd
############################ 定义分词函数 #######################################  
def clean_str(stri):
    import re
    stri = re.sub(r'[a-zA-Z0-9]+','',stri)
    cut_str = jieba.cut(stri.strip())
    list_str = [word for word in cut_str ]
    stri = ' '.join(list_str)
    return stri
def read_data():
    #读入训练数据：
    traindata= pd.read_csv('training-inspur.csv',encoding='gb18030')
    from sklearn.utils import shuffle
    traindata=shuffle(traindata)
    traindata=traindata.reset_index(drop=True)
    #读入测试数据：
    testdata= pd.read_csv('Preliminary-finals.csv','r','gb18030')
    m=[]
    for line in testdata.values:
        m.append(line[0].split(',')[:2])
    m=pd.DataFrame(m,columns=['ROWKEY','COMMCONTENT'])

    ###分词，并保存：
    traindata['COMMCONTENT'] = traindata['COMMCONTENT'].apply(lambda x:clean_str(str(x)))
    traindata.to_csv('traindata.csv', index=False,encoding='utf8')
    
    m['COMMCONTENT'] = m['COMMCONTENT'].apply(lambda x:clean_str(str(x)))
    m.to_csv('testdata.csv', index=False,encoding='utf8')
read_data()