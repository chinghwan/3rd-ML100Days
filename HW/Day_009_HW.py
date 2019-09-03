#!/usr/bin/env python
# coding: utf-8

# # 檢視與處理 Outliers
# ### 為何會有 outliers, 常見的 outlier 原因
# * 未知值，隨意填補 (約定俗成的代入)，如年齡常見 0,999
# * 可能的錯誤紀錄/手誤/系統性錯誤，如某本書在某筆訂單的銷售量 = 1000 本

# # [作業目標]
# - 依照下列提示與引導, 以幾種不同的方式, 檢視可能的離群值

# # [作業重點]
# - 從原始資料篩選可能的欄位, 看看那些欄位可能有離群值 (In[3], Out[3])
# - 繪製目標值累積密度函數(ECDF)的圖形, 和常態分布的累積密度函數對比, 以確認是否有離群值的情形 (In[6], Out[6], In[7], Out[7])

# In[ ]:


# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 設定 data_path
dir_data = './data' 

# In[ ]:


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


# ## 請參考 HomeCredit_columns_description.csv 的欄位說明，觀察並列出三個你覺得可能有 outlier 的欄位並解釋可能的原因

# In[ ]:


# 先篩選數值型的欄位
"""
YOUR CODE HERE, fill correct data types (for example str, float, int, ...)
"""
dtype_select = ['float64', 'int64']





#在Spyde不正常，先以其他方式解決
#numeric_columns = list(app_train.columns[list(app_train.dtypes.isin(dtype_select))])


numeric_columns = list(app_train.columns[list(app_train.dtypes=='float64')])
print("Numbers of remain columns:" + str(len(numeric_columns)))
# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns:" + str(len(numeric_columns)))

print(numeric_columns)

# In[ ]:
# 檢視這些欄位的數值範圍


app_train[numeric_columns].describe()


 # In[ ]:
app_train[['AMT_INCOME_TOTAL','REGION_POPULATION_RELATIVE','OBS_60_CNT_SOCIAL_CIRCLE']].describe()
# In[ ]:
app_train[['AMT_INCOME_TOTAL']].plot(kind='box')


# In[ ]:
for col in numeric_columns:
    """
    Your CODE HERE, make the box plot
    """
    print(col)
    plt.boxplot(app_train[col])
    plt.xlabel(col)
    plt.show()
    


# 從上面的圖檢查的結果，至少這三個欄位好像有點可疑

# AMT_INCOME_TOTAL
# REGION_POPULATION_RELATIVE
# OBS_60_CNT_SOCIAL_CIRCLE


# ### Hints: Emprical Cumulative Density Plot, [ECDF](https://zh.wikipedia.org/wiki/%E7%BB%8F%E9%AA%8C%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0), [ECDF with Python](https://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function)

# In[ ]:


# 最大值離平均與中位數很遠
print(app_train['AMT_INCOME_TOTAL'].describe())

print(app_train['REGION_POPULATION_RELATIVE'].describe())

print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
YOUR CODE HERE
"""
# In[ ]:

def ecdf(data):
    x=np.sort(data)
    y=np.arange(1,len(x)+1)/len(x)
    return (x,y)







cdf = ecdf(app_train['AMT_INCOME_TOTAL'])


x,y = ecdf(app_train['AMT_INCOME_TOTAL'])
plt.scatter(x=x, y=y);

plt.xlim([x.min(), x.max() * 1.05]) # 限制顯示圖片的範圍


plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)


plt.xlim([x.min(), x.mean() * 3]) # 限制顯示圖片的範圍


plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)



 

# In[ ]:

plt.plot((cdf), np.arange(1,len(cdf)+1)/len(cdf))
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()
# In[ ]:
# 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
#plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
plt.plot(np.log(cdf), np.arange(1,len(cdf)+1)/len(cdf))
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
 
plt.show()


# ## 補充：Normal dist 的 ECDF
# ![ecdf_normal](https://au.mathworks.com/help/examples/stats/win64/PlotEmpiricalCdfAndCompareWithSamplingDistributionExample_01.png)

# In[ ]:


# 最大值落在分布之外
print(app_train['REGION_POPULATION_RELATIVE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
Your Code Here
"""
cdf = 


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()

# 就以這個欄位來說，雖然有資料掉在分布以外，也不算異常，僅代表這間公司在稍微熱鬧的地區有的據點較少，
# 導致 region population relative 在少的部分較為密集，但在大的部分較為疏漏


# In[ ]:


# 最大值落在分布之外
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
Your Code Here

cdf = 


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min() * 0.95, cdf.index.max() * 1.05])
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()
"""
app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False))


# ## 注意：當 histogram 畫出上面這種圖 (只出現一條，但是 x 軸延伸很長導致右邊有一大片空白時，代表右邊有值但是數量稀少。這時可以考慮用 value_counts 去找到這些數值

# In[ ]:


# 把一些極端值暫時去掉，在繪製一次 Histogram
# 選擇 OBS_60_CNT_SOCIAL_CIRCLE 小於 20 的資料點繪製
"""
Your Code Here
"""
loc_a = 
loc_b = 

app_train.loc[loc_a, loc_b].hist()
plt.show()

