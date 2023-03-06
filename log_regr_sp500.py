# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 07:32:41 2023

@author: kuse
"""

import yfinance as yf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




df=yf.download('^GSPC', start='2010-01-01')

df['ret']=df.Close.pct_change()

def lagit(df,lags):
    for i in range(1,lags+1):
        df['lag_'+str(i)]=df['ret'].shift(i)
    return ['lag_'+str(i) for i in range(1,lags+1)]



lagit(df,2)

df['direction']=np.where(df.ret>0,1,0)
df.direction.value_counts()


features=lagit(df,3)
df.dropna(inplace=True)
X=df[features]
y=df['direction']

model= LogisticRegression(class_weight='balanced')

model.fit(X,y)
df['prediction']=model.predict(X)
df['strat']=df['prediction']*df['ret']
(df[['strat','ret']]+1).cumprod()-1;

((df[['strat','ret']]+1).cumprod()-1).plot()
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,shuffle=False)
model.fit(X_train,y_train)

X_test['prediction_LR']=model.predict(X_test)

X_test['ret']=df.ret[X_test.index[0]:]
X_test['strat']=X_test['prediction_LR']*X_test['ret']
(X_test[['strat','ret']]+1).cumprod()-1





















