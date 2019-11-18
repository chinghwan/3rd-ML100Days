#!/usr/bin/env python
# coding: utf-8

# #   
# 
# # 作業目標:
# 
#     1. 藉由固定的 dataset, 來驗證不同loss function
#     2. Dataset 的特性跟我們選用的loss function 對accrancy 的影響
#     
#     
# # 作業重點: 
#     請分別選用 "MSE", "binary _crossentropy"
#     查看Train/test accurancy and loss rate
#     

# # 導入必要的函數

# In[1]:


from keras.datasets import cifar10
import numpy as np
np.random.seed(10)


# # 資料準備

# In[2]:


#取得Keras Dataset
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()


# In[3]:


#確認 CIFAR10 Dataset 資料維度
print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape) 
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape) 


# In[4]:


#資料正規化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0


# In[5]:


#針對Label 做 ONE HOT ENCODE
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape


# # 建立模型

# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


# In[7]:


model = Sequential()


# In[8]:


#卷積層1


# In[9]:


model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))


# In[10]:


model.add(Dropout(rate=0.25))


# In[11]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[12]:


#卷積層2與池化層2


# In[13]:


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))


# In[14]:


model.add(Dropout(0.25))


# In[15]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[16]:


#建立神經網路(平坦層、隱藏層、輸出層)


# In[17]:


model.add(Flatten())
model.add(Dropout(rate=0.25))


# In[18]:


model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))


# In[19]:


model.add(Dense(10, activation='softmax'))


# In[20]:


#檢查model 的STACK
print(model.summary())


# # 載入之前訓練的模型

# In[21]:


try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")


# # 訓練模型

# In[27]:


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

'''
作業:
請分別選用 "MSE", "binary _crossentropy"
查看Train/test accurancy and loss rate
'''
model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy']) 


# In[28]:


train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=12, 
                        batch_size=128) 


# In[ ]:




