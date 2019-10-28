#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# ***
# - 分數以網站評分結果為準, 請同學實際將提交檔(*.csv)上傳試試看  
# https://www.kaggle.com/c/titanic/submit

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀查堆疊泛化 (Stacking) 的寫法與效果

# # [作業重點]
# - 完成堆疊泛化的寫作, 看看提交結果, 想想看 : 分類與回歸的堆疊泛化, 是不是也與混合泛化一樣有所不同呢?(In[14])  
# 如果可能不同, 應該怎麼改寫會有較好的結果?  
# - Hint : 請參考 mlxtrend 官方網站 StackingClassifier 的頁面說明 : Using Probabilities as Meta-Features
# http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/

# In[1]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data_path = './data/'
df_train = pd.read_csv(data_path + 'train_data.csv')
df_test = pd.read_csv(data_path + 'test_features.csv')

train_Y = df_train['poi']


df_test=df_test.drop(20)
ids = df_test['name']
df_train = df_train.drop(['name', 'poi','email_address'] , axis=1)
df_test = df_test.drop(['name','email_address'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


# In[2]:


# 檢查 DataFrame 空缺值的狀態
def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))
na_check(df)
 

#df_train = df_train.drop(['loan_advances','director_fees','restricted_stock_deferred','deferral_payments'] , axis=1)
#df_test = df_test.drop(['loan_advances','director_fees','restricted_stock_deferred','deferral_payments'] , axis=1)

 
df_train= df_train.fillna(0)

df_test= df_test.fillna(0)
 
df_train['to_poi_ratio']=(df_train['from_poi_to_this_person']/df_train['to_messages']).fillna(0)
 
df_test['to_poi_ratio']=(df_test['from_poi_to_this_person']/df_test['to_messages']).fillna(0)
 


#df_train = df_train.drop(['from_messages','from_poi_to_this_person','from_this_person_to_poi','to_messages'] , axis=1)
#df_test = df_test.drop(['from_messages','from_poi_to_this_person','from_this_person_to_poi','to_messages'] , axis=1)


# In[9]:


na_check(df_train)
df_train.head()


# In[10]:


# 將資料最大最小化
df = MinMaxScaler().fit_transform(df_train)
df2 = MinMaxScaler().fit_transform(df_test)
 

# 使用三種模型 : 邏輯斯迴歸 / 梯度提升機 / 隨機森林, 參數使用 Random Search 尋找
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
lr = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=15  , max_features=20,
                                  max_depth=6, learning_rate=0.03)
rf = RandomForestClassifier(n_estimators=15, min_samples_split=2, min_samples_leaf=1, 
                            max_features='sqrt', max_depth=8, bootstrap=True)


# In[11]:

df3 = pd.DataFrame([['TOTAL', 0]],columns=['name','poi'])

# In[11]:
# 線性迴歸預測檔 (結果有部分隨機, 請以 Kaggle 計算的得分為準, 以下模型同理)
lr.fit(df, train_Y)
lr_pred = lr.predict_proba(df2)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': lr_pred})
sub=sub.append(df3,ignore_index=True)
sub.to_csv('poi_lr_1.csv', index=False) 


# In[12]:


# 梯度提升機預測檔 
gdbt.fit(df, train_Y)
gdbt_pred = gdbt.predict_proba(df2)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred}) 
sub=sub.append(df3,ignore_index=True) 

sub.to_csv('poi_gdbt3.csv', index=False)


# In[13]:


# 隨機森林預測檔
rf.fit(df, train_Y)
rf_pred = rf.predict_proba(df2)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': rf_pred})
#sub['poi'] = sub['poi'].map(lambda x:1 if x>0.5 else 0) 

sub=sub.append(df3,ignore_index=True) 

sub.to_csv('poi_rf3.csv', index=False)


# # 作業
# * 分類預測的集成泛化, 也與回歸的很不一樣  
# 既然分類的 Blending 要變成機率, 才比較容易集成,
# 那麼分類的 Stacking 要讓第一層的模型輸出機率當特徵, 應該要怎麼寫呢?

# In[14]:


from mlxtend.classifier import StackingClassifier

meta_estimator = GradientBoostingClassifier(subsample=0.70, n_estimators=15, 
                                           max_features='sqrt', max_depth=4, learning_rate=0.3)
"""
Your Code Here
"""
stacking = StackingClassifier(classifiers=[gdbt, rf], meta_classifier=meta_estimator)


# In[15]:


stacking.fit(df, train_Y)
stacking_pred = stacking.predict_proba(df2)
sub = pd.DataFrame({'name': ids, 'poi': stacking_pred[:,1]})
sub=sub.append(df3,ignore_index=True) 
sub.to_csv('poi_stacking_4.csv', index=False)


# In[ ]:




