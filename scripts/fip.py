# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 04:50:41 2019

@author: 43884
"""

import pandas as pd
import csv

with open('bank-full.csv', 'r') as f:
    reader = csv.reader(f)
    
    result = list(reader)
    colnames = result[0][0].replace('"', '').split(';')
    df = pd.DataFrame(columns = colnames)
    df.loc[-1] = result[-1][0].replace('"', '').split(';')
    df.index = df.index + 1
    for row in result[1:-1]:
        df.loc[-1] = row[0].replace('"', '').split(';')
        df.index = df.index + 1
    df.to_excel('./raw_data.xlsx', index = False, header = True)

