#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data Visualization


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[9]:


year = [2014,2015,2016,2017,2018]
sales = [1234,4344,4454,3222,3343]
profit = [123,443,445,432,222]


# In[7]:


plt.plot(year, sales, 'red')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Sales vs Year')


# In[8]:


customer = ['Customer1','Customer2','Customer3','Customer4','Customer5']


# In[10]:


plt.scatter(sales,profit)


# In[11]:


plt.plot(year,sales,'red')
plt.plot(year, profit,'blue')


# In[41]:


plt.axes([0.05,1.5,0.9,0.9])
plt.plot(year, sales,'red')
plt.title('Year and Sales')
plt.axes([0.05,3,0.9,0.9])
plt.plot(year, profit,'blue')
plt.title('Year and Profit')


# In[42]:


plt.subplot(2,1,1)
plt.plot(year, sales,'red')
plt.title('Year and Sales')
#plt.xlim((2014,2015))
#plt.ylim((1500,2500))
plt.axis((2014,2015,1500,2500))
plt.subplot(2,1,2)
plt.plot(year, profit,'blue')
plt.title('Year and Profit')
plt.tight_layout()


# In[38]:


plt.scatter(sales,profit, color='blue', marker='+')
plt.annotate('2014',xy=(1234,123))
plt.style.use('classic')


# In[28]:


sales


# In[29]:


profit


# In[30]:


print(plt.style.available)


# In[43]:


import os 
os.chdir('C:\\Users\\HP\\Desktop\\Desktop\\Tosh Viz\\Feb 16 - Viz\\')


# In[44]:


dataset = pd.read_excel('Sample - Superstore.xls')


# In[45]:


dataset


# In[49]:


dataset['Profit'].plot.hist()


# In[51]:


type(dataset['Quantity'])


# In[54]:


sns.distplot(dataset['Quantity'],kde=False,color='g')


# In[61]:


sns.regplot(x='Sales', y='Profit', data=dataset, fit_reg=False,col='Segment')


# In[62]:


sns.lmplot(x='Sales',y='Profit', data=dataset)


# In[65]:


sns.jointplot(x='Sales',y='Quantity', data=dataset,kind='kde')


# In[67]:


sns.countplot(data=dataset, x='Category')


# In[71]:


sns.barplot(x='Region', y='Sales', data=dataset)


# In[73]:


sns.stripplot(y='Sales',x='Category',data=dataset)


# In[ ]:




