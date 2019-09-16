# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:15:22 2019

@author: sbtithzy
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')

def DNN(X_train,y_train,X_valid,y_valid,X_submit,batch_size,nb_epoch,input_num):
    model = Sequential()
    model.add(Dense(units = 128,kernel_initializer='uniform',activation='relu',input_dim = input_num))
    ###再添加一个隐藏
    model.add(Dense(units = 32,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 8,kernel_initializer='uniform',activation='relu'))
    ###添加一个输出层
    #model.add(Dense(units = 4,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 1,kernel_initializer='uniform',activation='sigmoid'))
    # =============================================================================
    # 编译ANN
    # 编译ANN的时候重要参数,optimizer,loss,这里选择随机梯度下降,损失函数选择 binary_crossentropy,
    # 主要考虑输出是二分类的数据， metrics这里关注准确率 
    # =============================================================================
    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # Dropout  需要断开的连接的比例
    #model.add(Dropout(0.2))
    
    # 打印出模型概况
    print('model.summary:')
    model.summary()
    # 在训练模型之前，通过compile来对学习过程进行配置
    # 编译模型以供训练
    # 包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']
    # 如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
    #classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    
    
    # 训练模型
    # Keras以Numpy数组作为输入数据和标签的数据类型
    # fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
    # nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    # shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。
    
    # fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_valid, y_valid))
    y_submit = model.predict(X_submit)
    return y_submit