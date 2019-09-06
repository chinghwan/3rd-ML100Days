#!/usr/bin/env python
# coding: utf-8

# # 處理 outliers
# * 新增欄位註記
# * outliers 或 NA 填補
#     1. 平均數 (mean)
#     2. 中位數 (median, or Q50)
#     3. 最大/最小值 (max/min, Q100, Q0)
#     4. 分位數 (quantile)

# # [教學目標]
# 為了要處理離群值, 我們要先學會計算其他的統計量, 並且還有其他的挑整方式

# # [範例重點]
# - 計算並觀察百分位數 (In[4], In[7])
# - 計算中位數的方式 (In[8])
# - 計算眾數 (In[9], In[10])
# - 計算標準化與最大最小化 (In[11])

# In[1]:


# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# 設定 data_path
dir_data = './data/'


# In[2]:


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


# In[3]:


# 如果欄位中有 NA, describe 會有問題
app_train['AMT_ANNUITY'].describe()


# In[4]:


# Ignore NA, 計算五值
five_num = [0, 25, 50, 75, 100]
quantile_5s = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in five_num]
print(quantile_5s)


# In[5]:


app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'].hist(bins = 100)
plt.show()


# In[6]:


# 試著將 max 取代為 q99


app_train[app_train['AMT_ANNUITY'] == app_train['AMT_ANNUITY'].max()] =   np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = 99)


# In[8]:


five_num = [0, 25, 50, 75, 100]
quantile_5s = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in five_num]
print(quantile_5s)


# In[ ]:


# 得到 median 的另外一種方法
np.median(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])


# In[ ]:


# 計算眾數 (mode)
from scipy.stats import mode
import time

start_time = time.time()
mode_get = mode(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])
print(mode_get)
print("Elapsed time: %.3f secs" % (time.time() - start_time))


# In[ ]:


# 計算眾數 (mode)
# 較快速的方式
from collections import defaultdict

start_time = time.time()
mode_dict = defaultdict(lambda:0)

for value in app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY']:
    mode_dict[value] += 1
    
mode_get = sorted(mode_dict.items(), key=lambda kv: kv[1], reverse=True)
print(mode_get[0])
print("Elapsed time: %.3f secs" % (time.time() - start_time))


# ## 連續值標準化
# ### 1. Z-transform: $ \frac{(x - mean(x))}{std(x)} $
# ### 2. Range (0 ~ 1): $ \frac{x - min(x)}{max(x) - min(x)} $
# ### 3. Range (-1 ~ 1): $ (\frac{x - min(x)}{max(x) - min(x)} - 0.5) * 2 $

# In[ ]:


# 以 AMT_CREDIT 為例
app_train['AMT_CREDIT'].hist(bins = 50)
plt.title("Original")
plt.show()
value = app_train['AMT_CREDIT'].values

app_train['AMT_CREDIT_Norm1'] = ( value - np.mean(value) ) / ( np.std(value) )
app_train['AMT_CREDIT_Norm1'].hist(bins = 50)
plt.title("Normalized with Z-transform")
plt.show()

app_train['AMT_CREDIT_Norm2'] = ( value - min(value) ) / ( max(value) - min(value) )
app_train['AMT_CREDIT_Norm2'].hist(bins = 50)
plt.title("Normalized to 0 ~ 1")
plt.show()


# # It's your turn
# ### 1. 列出 AMT_ANNUITY 的 q0 - q100
# ### 2.1 將 AMT_ANNUITY 中的 NAs 暫時以中位數填補
# ### 2.2 將 AMT_ANNUITY 的數值標準化至 -1 ~ 1 間
# ### 3. 將 AMT_GOOD_PRICE 的 NAs 以眾數填補
# 

# In[ ]:




