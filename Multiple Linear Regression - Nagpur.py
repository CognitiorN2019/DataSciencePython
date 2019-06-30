#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


#salary = 9439.22*0 + 25907.49


# In[2]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[3]:


import os
os.chdir('D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression\\Multiple_Linear_Regression')


# In[4]:


dataset = pd.read_csv('50_Startups.csv')


# In[5]:


dataset


# In[18]:


x = dataset.iloc[:,0:4].values


# In[19]:


x


# In[20]:


import os
os.chdir('D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Simple_Linear_Regression\\Simple_Linear_Regression\\')


# In[21]:


dataset = pd.read_csv('Salary_Data.csv')


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


# In[ ]:


Salary = m1*Male + m2*Female + m3*Exp 


# In[ ]:


Salary = m1*Male + m3*Exp 


# In[ ]:


profit = m1*s1 + m2*s2  + m4*RD +m5*As + m6*ms + c


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


r2_score(y_test,y_pred)


# In[39]:


regressor.intercept_

