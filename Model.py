# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 03:54:27 2019

@author: 43884
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

df = pd.read_excel('./processed_data2.xlsx').values


x = df[:, :-1]
labels = df[:, -1]
sel = VarianceThreshold(threshold=0.2)
x_transform = sel.fit_transform(x)
x_transform.shape
sel.get_support(indices = False)



x_train, x_rest, y_train, y_rest = train_test_split(df[:, :-1], df[:, -1], test_size = 0.4)
x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size = 0.5) 

#before oversampling
print(pd.Series(y_train).value_counts()/len(y_train))
smo = SMOTE(random_state=42)
x_smo, y_smo = smo.fit_sample(x_train, y_train)
#after oversampling
print(pd.Series(y_smo).value_counts()/len(y_smo))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


#RF
clf = RandomForestClassifier(n_estimators=1000, max_depth = 10, oob_score = True)
#scores2 = cross_val_score(clf, x, labels, scoring='accuracy')
#print(scores2.mean())

#original data
#clf.fit(x_train, y_train)
#pred = clf.predict(x_test)
#print(metrics.classification_report(y_test, pred))

#smo data, 0.90
clf.fit(x_smo, y_smo)
pred = clf.predict(x_val)
print(metrics.classification_report(y_val, pred))


#LR
clf2 = LogisticRegression()
#scores = cross_val_score(clf2, x_transform, labels, scoring='accuracy')
#print(scores.mean())

#original data
#clf2.fit(x_train, y_train)
#pred2 = clf2.predict(x_test)
#print(metrics.classification_report(y_test, pred2))

#smo data, 0.88
clf2.fit(x_smo, y_smo)
pred2 = clf2.predict(x_val)
print(metrics.classification_report(y_val, pred2))

#XGBoost
#smo data, 0.92
bst = XGBClassifier(learning_rate = 0.01, n_estimators = 1000, silent = 1, objective = 'binary:logistic', max_depth = 4, colsample_bytree = 0.6, subsample = 0.6, gamma = 0)
bst.fit(x_smo, y_smo)
pred3 = bst.predict(x_val)
print(metrics.classification_report(y_val, pred3))


#NN
#smo data, 
x_nn = Variable(torch.tensor(x_smo).float())
y_nn = Variable(torch.tensor(np.mat(y_smo).T).float())

myNet = nn.Sequential(
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Sigmoid(),
    nn.Linear(10, 10),
    nn.Sigmoid(),
    nn.Linear(10, 5),
    nn.Sigmoid(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)
print(myNet)

optimzer = torch.optim.SGD(myNet.parameters(), lr=0.02)
loss_func = nn.MSELoss()

for epoch in range(5000):
    out = myNet(x_nn)
    loss = loss_func(out, y_nn)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    
print(myNet(Variable(torch.tensor(x_val).float())).data)
pd.Series(myNet(Variable(torch.tensor(x_val).float())).data).value_counts()

pred4 = myNet(Variable(torch.tensor(x_val).float())).data

result = [0 if i < 0.5 else 1 for i in pred4]

print(metrics.classification_report(y_val, result))



#Blending
p1 = clf.predict_proba(x_test)[:, 1]
p2 = clf2.predict_proba(x_test)[:, 1]
p3 = bst.predict_proba(x_test)[:, 1]
p4 = np.array(np.mat(myNet(Variable(torch.tensor(x_test).float())).data).T)
features = np.array(np.mat(np.vstack((p1, p2, p3, p4))).T)

print(pd.Series(y_test).value_counts()/len(y_test))
x_smo1, y_smo1 = smo.fit_sample(features, y_test)
print(pd.Series(y_smo1).value_counts()/len(y_smo1))

new_val = np.array(np.mat(np.vstack((pred, pred2, pred3, np.array(np.mat(pred4)).T))).T)
new_val

clf3 = XGBClassifier(learning_rate = 0.01, n_estimators = 1000, silent = 1, objective = 'binary:logistic', max_depth = 3, colsample_bytree = 0.8, subsample = 0.6, gamma = 0)
clf3.fit(x_smo1, y_smo1)
pred5 = clf3.predict(new_val)
print(metrics.classification_report(y_val, pred5))


#AUC Plot
y_score = clf.predict_proba(x_val)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_val, y_score)
roc_auc = metrics.auc(fpr, tpr)
y_score2 = clf2.predict_proba(x_val)[:, 1]
fpr2, tpr2, threshold2 = metrics.roc_curve(y_val, y_score2)    
roc_auc2 = metrics.auc(fpr2, tpr2)
y_score3 = bst.predict_proba(x_val)[:, 1]
fpr3, tpr3, threshold3 = metrics.roc_curve(y_val, y_score3)    
roc_auc3 = metrics.auc(fpr3, tpr3)
y_score4 = pred4
fpr4, tpr4, threshold4 = metrics.roc_curve(y_val, y_score4)    
roc_auc4 = metrics.auc(fpr4, tpr4)
y_score5 = clf3.predict_proba(new_val)[:, 1]
fpr5, tpr5, threshold5 = metrics.roc_curve(y_val, y_score5)    
roc_auc5 = metrics.auc(fpr5, tpr5)
#plt.stackplot(fpr, tpr, color = 'steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color = 'darkorange', label = 'RF ROC curve (area = %0.3f)' % roc_auc)
plt.plot(fpr2, tpr2, color = 'red', label = 'LR ROC curve (area = %0.3f)' % roc_auc2)
plt.plot(fpr3, tpr3, color = 'green', label = 'XGBoost ROC curve (area = %0.3f)' % roc_auc3)
plt.plot(fpr4, tpr4, color = 'darkblue', label = 'NN ROC curve (area = %0.3f)' % roc_auc4)
plt.plot(fpr5, tpr5, color = 'darkgreen', label = 'Blending ROC curve (area = %0.3f)' % roc_auc5)
plt.plot([0, 1], [0, 1], color='navy', lw = 1, linestyle='--')
#plt.text(0.2, 0.8, 'ROC curve (area = %0.3f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc="lower right")
plt.show()



