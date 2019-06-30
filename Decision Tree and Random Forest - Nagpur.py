#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


dataset = pd.read_csv('D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression\\Multiple_Linear_Regression\\50_Startups.csv')


# In[4]:


dataset


# In[6]:


x = dataset.iloc[:,:-1].values


# In[7]:


x


# In[8]:


y = dataset.iloc[:,4].values


# In[9]:


y


# In[10]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()


# In[11]:


pd.DataFrame(x)


# In[12]:


x = x[:,1:]


# In[13]:


pd.DataFrame(x)


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[76]:


from sklearn.tree import DecisionTreeRegressor
regressor  = DecisionTreeRegressor()
regressor.fit(x_bk,y_train)


# In[81]:


y_pred = regressor.predict(x_bk_ts)


# In[79]:


x_bk_ts = x_test[:,3:4]


# In[82]:


y_pred


# In[18]:


y_test


# In[83]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[87]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=5)
regressor.fit(x_bk,y_train)


# In[85]:


len(y_train)


# In[86]:


len(x_bk_ts)


# In[88]:


y_pred = regressor.predict(x_bk_ts)


# In[22]:


y_pred


# In[23]:


y_test


# In[89]:


r2_score(y_test,y_pred)


# In[25]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[26]:


y_pred = regressor.predict(x_test)


# In[27]:


r2_score(y_test,y_pred)


# In[33]:


import seaborn as sns
sns.pairplot(dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']])


# In[90]:


import statsmodels.formula.api as sm
regressor_ols = sm.OLS(y_train,x_bk).fit()


# In[91]:


regressor_ols.summary()


# In[36]:


pd.DataFrame(x_train)


# In[72]:


x_bk = x_train[:, 3:4]


# In[73]:


pd.DataFrame(x_bk)

