#!/usr/bin/env python
# coding: utf-8

# # 常用的 DataFrame 操作
# * merge / transform
# * subset
# * groupby

# # [作業目標]
# - 練習填入對應的欄位資料或公式, 完成題目的要求 

# # [作業重點]
# - 填入適當的輸入資料, 讓後面的程式顯示題目要求的結果 (Hint: 填入對應區間或欄位即可, In[4]~In[6], Out[4]~In[6])
# - 填入z轉換的計算方式, 完成轉換後的數值 (Hint: 參照標準化公式, In[7])

# In[4]:


# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# 設定 data_path
dir_data = './data/'


# In[6]:


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


# ## 作業
# 1. 請將 app_train 中的 CNT_CHILDREN 依照下列規則分為四組，並將其結果在原本的 dataframe 命名為 CNT_CHILDREN_GROUP
#     * 0 個小孩
#     * 有 1 - 2 個小孩
#     * 有 3 - 5 個小孩
#     * 有超過 5 個小孩
# 
# 2. 請根據 CNT_CHILDREN_GROUP 以及 TARGET，列出各組的平均 AMT_INCOME_TOTAL，並繪製 baxplot
# 3. 請根據 CNT_CHILDREN_GROUP 以及 TARGET，對 AMT_INCOME_TOTAL 計算 [Z 轉換](https://en.wikipedia.org/wiki/Standard_score) 後的分數

# In[9]:


#1
"""
Your code here
"""
cut_rule = [0,1, 2, 5,99]

app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule, include_lowest=True)
app_train['CNT_CHILDREN_GROUP'].value_counts()


# In[ ]:


#2-1
"""
Your code here
"""
grp = 

grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL']
grouped_df.mean()


# In[ ]:


#2-2
"""
Your code here
"""
plt_column = 
plt_by = 

app_train.boxplot(column=plt_column, by = plt_by, showfliers = False, figsize=(12,12))
plt.suptitle('')
plt.show()


# In[ ]:


#3
"""
Your code here
"""
app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'] = grouped_df.apply(lambda x: )

app_train[['AMT_INCOME_TOTAL','AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET']].head()

