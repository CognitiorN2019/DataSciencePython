#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.chdir('D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression\\Multiple_Linear_Regression')


# In[4]:
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')


# In[5]:


dataset


# In[18]:


x = dataset.iloc[:,0:4].values




# In[24]:


dataset.iloc[:,:-1].values


# In[27]:


dataset.iloc[:,0:1].values


# In[10]:


dataset


# In[7]:


x


# In[8]:


y = dataset.iloc[:,4].values


# In[9]:


y


# In[28]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotendoer = OneHotEncoder(categorical_features=[3])
x = onehotendoer.fit_transform(x).toarray()


# In[30]:


pd.DataFrame(x)


# In[31]:


x = x[:,1:]



# In[32]:


pd.DataFrame(x)


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[41]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[42]:


y_pred = regressor.predict(x_test)


# In[36]:


y_test


# In[37]:


y_pred


# In[43]:


regressor.coef_


# In[40]:

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[39]:


regressor.intercept_


# In[5]:


import statsmodels.formula.api as sm


# In[7]:


pd.DataFrame(x)


# In[8]:


import numpy as np


# In[9]:


x  = np.append(arr=x, values=np.ones((50,1)).astype(int), axis=1)


# In[10]:


pd.DataFrame(x)


# In[ ]:


y = mx + 1*c 


# In[11]:


regressor_OLS = sm.OLS(endog=y, exog=x).fit()


# In[12]:


regressor_OLS.summary()


# In[ ]:


#P value
#Decision Tree 
#Random Forest 
#Multicolinearity code 

