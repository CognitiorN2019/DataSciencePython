#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Simple_Linear_Regression\\Simple_Linear_Regression\\')


# In[2]:


import pandas as pd


# In[3]:


dataset = pd.read_csv('Salary_Data.csv')


# In[4]:


dataset


# In[5]:


x = dataset.iloc[:,:-1].values


# In[6]:


x


# In[7]:


y = dataset.iloc[:,1].values


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)


# In[12]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[13]:


y_test


# In[14]:


y_pred = regressor.predict(x_test)


# In[15]:


y_pred


# In[16]:


x_test


# In[17]:


import matplotlib.pyplot as plt


# In[19]:


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')


# In[20]:


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')


# In[32]:


dataset_pred = pd.read_excel('prediction_new.xlsx')


# In[33]:


x_new=dataset_pred.iloc[:,].values


# In[34]:


x_new


# In[35]:


y_pred_new =regressor.predict(x_new)


# In[36]:


y_pred_new


# In[30]:


regressor.coef_


# In[31]:


regressor.intercept_


# In[ ]:


salary = 9439.22*0 + 25907.49

