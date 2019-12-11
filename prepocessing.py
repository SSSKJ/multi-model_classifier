# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:59:15 2019

@author: 43884
"""
import pandas as pd
from sklearn import preprocessing

df = pd.DataFrame(data = pd.read_csv('./bank-additional-full.csv', sep = ';', header = 0))
df.y = preprocessing.LabelBinarizer().fit_transform(df.y)
df.housing = preprocessing.LabelBinarizer().fit_transform(df.housing)
df.loan = preprocessing.LabelBinarizer().fit_transform(df.loan)
df.default = preprocessing.LabelBinarizer().fit_transform(df.default)

df.job = preprocessing.LabelEncoder().fit(df.job).transform(df.job)
df.marital = preprocessing.LabelEncoder().fit(df.marital).transform(df.marital)
df.month = preprocessing.LabelEncoder().fit(df.month).transform(df.month)
df.day_of_week = preprocessing.LabelEncoder().fit(df.day_of_week).transform(df.day_of_week)
df.education = preprocessing.LabelEncoder().fit(df.education).transform(df.education)
df.contact = preprocessing.LabelEncoder().fit(df.contact).transform(df.contact)
df.poutcome = preprocessing.LabelEncoder().fit(df.poutcome).transform(df.poutcome)

df.job.value_counts()
df.contact.value_counts()
df.poutcome.value_counts()
df.month.value_counts()
df.marital.value_counts()
df.pdays.value_counts()
df['cons.conf.idx'].value_counts()
df['cons.price.idx'].value_counts()
df['emp.var.rate'].value_counts()

df.to_excel('./processed_data2.xlsx', index = False, header = True)
