# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:09:49 2019

@author: sbtithzy
"""
'''
比赛数据（脱敏后）抽取的是一段时间范围内，客户的购买行为数据，初赛和复赛两个阶段所提供样本的量级有所不同。
参赛选手根据特征字段信息进行建模，预测所有已购客户在未来180天的购买概率。
'''
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:15:02 2019

@author: sbtithzy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import warnings
import time 
import make_feature_and_label,preprocess,lgb,DNN
warnings.filterwarnings('ignore')
batch_size = 100
nb_classes = 10
nb_epoch = 20
# 这个代码是，看看具体主办方的 deta 大概的取值，应该是2-5之间的一个值，可能是一个自然数e
# =============================================================================
# def logloss(y_true, y_pred,deta = 3, eps=1e-15):
#     # Prepare numpy array data
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     assert (len(y_true) and len(y_true) == len(y_pred))
#     # Clip y_pred between eps and 1-eps
#     p = np.clip(y_pred, eps, 1-eps)
#     loss = np.sum(- y_true * np.log(p) * deta - (1 - y_true) * np.log(1-p))
#     return loss / len(y_true)
# =============================================================================
# 读取预处理后数据
print(u'预处理部分')
trian_data,trian,all_customer = preprocess.prepro()
#result['min_time_point'] = result['order_pay_date_1'] + datetime.timedelta(days=180)
#result['max_time_point'] = pd.to_datetime('2014-01-01')+datetime.timedelta(days=180)
validata_date_begin = trian['order_pay_date'].max() - datetime.timedelta(days=180)
# 简单的特征生成部分代码
# 生成训练数据和提交数据
train_history = trian[(trian['order_pay_date'].astype(str)<='2013-06-30')]
online_history = trian[(trian['order_pay_date'].astype(str)<='2013-12-31')]
# train_label 相对于 train_history 的未来180天的数据
train_label = trian[trian['order_pay_date'].astype(str)>='2013-07-01']
#
print(u'特征工程及训练集/测试集部分')
start = time.perf_counter()
train = make_feature_and_label.make_feature_and_label(train_history,train_label,False)
submit = make_feature_and_label.make_feature_and_label(online_history,None,True)
elapsed = (time.perf_counter() - start)#####时间结束点
print (u'特征工程及训练集/测试集累计用时:',elapsed) #####累计用时
# 模型准备
# =============================================================================
# y = train.pop('label')
# feature = [x for x in train.columns if x not in ['customer_id']]
# X = train[feature]
# # 划分训练集和验证集
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
# submit_df = submit[['customer_id']]
# X_submit = submit[feature]
# print(u'模型拟合')
# # lgb
# y_submit = lgb.LGB(X_train,y_train,X_valid,y_valid,X_submit,1000,50)
# # DNN
# y_submit = DNN.DNN(X_train,y_train,X_valid,y_valid,X_submit,batch_size,nb_epoch,X_train.shape[1])
# y_submit = (y_submit > 0.45)
# # result
# submit_df['result'] = y_submit
# 
# all_customer = pd.merge(all_customer,submit_df,on=['customer_id'],how='left',copy=False)
# all_customer = all_customer.sort_values(['customer_id'])
# all_customer['customer_id'] = all_customer['customer_id'].astype('int64')
# all_customer['result'] = all_customer['result'].fillna(0)
# all_customer.to_csv('./hzy_baseline.csv',index=False)
# print('结束')
# =============================================================================
##########################################################
######模型
#####拆分train和test
submit_df = submit[['customer_id']]
test = submit

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import model_selection 
target= 'label'
IDcol = 'customer_id'
######测试模型建立
def modelfit(alg, dtrain, dtest, predictors1,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    global all_customer,submit_df
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors1].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors1].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print (cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors1], dtrain['label'],eval_metric='auc')#error

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors1])
    
    print(dtrain_predictions)
    dtrain_predprob = alg.predict_proba(dtrain[predictors1])[:,1]
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))
#####预测
    dtest_predictions = alg.predict_proba(dtest[predictors])[:,1]
#    test_y = alg.predict(xgtest)
    submit_df['result'] = dtest_predictions
# 
    all_customer = pd.merge(all_customer,submit_df,on=['customer_id'],how='left',copy=False)
    all_customer = all_customer.sort_values(['customer_id'])
    all_customer['customer_id'] = all_customer['customer_id'].astype('int64')
    all_customer['result'] = all_customer['result'].fillna(0)
    all_customer.to_csv('./hzy_baseline.csv',index=False)
# =============================================================================
#     result = test
#     result['label'] = dtest_predictions
#     #result['label'] = result['label'].map(lambda x: 1 if x>= 0.9 else 0)
#     result[['USER_ID','label']].to_csv('/mnt/sd04/sjjs_js06/1_江苏6队.csv',index=False,header=False)
    xgb_fea_imp=pd.DataFrame(list(alg.get_booster().get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
# =============================================================================
    print('',xgb_fea_imp)
#    xgb_fea_imp.to_csv('/mnt/sd04/sjjs_js06/xgb_fea_imp.csv')
predictors = [x for x in train.columns if x not in [target, IDcol]]
#46078
######调参过程
# =============================================================================
# 
# param_test1 = {
#     'max_depth':[4,5,6],
#     'min_child_weight':[6,7,8,9]
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=128, max_depth=5,
#                                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
#                        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# print(gsearch1.fit(train[predictors],train[target]))
# 
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# 
# 
# param_test3 = {
#     'gamma':[i/10.0 for i in range(0,5)]
# }
# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=128, max_depth=5,
#                                         min_child_weight=8, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                        param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# print(gsearch3.fit(train[predictors],train[target]))
# 
# print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
# 
# 
# param_test4 = {
#     'subsample':[i/10.0 for i in range(6,10)],
#     'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=117, max_depth=5,
#                                         min_child_weight=8, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                        param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# (gsearch4.fit(train[predictors],train[target]))
# 
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# 
# 
# param_test6 = {
#     'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# }
# gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=117, max_depth=5,
#                                         min_child_weight=7, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
#                                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                        param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# print(gsearch6.fit(train[predictors],train[target]))
# 
# print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)
# 
# =============================================================================
####最终模型的参数
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=2000,
        max_depth=5,
        min_child_weight=8,
        gamma=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
print(modelfit(xgb1, train, test, predictors))
