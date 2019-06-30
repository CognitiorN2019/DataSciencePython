#!/usr/bin/env python
# coding: utf-8

# In[1]:


#relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


year = [2014, 2015, 2016, 2017, 2018]
profit = [230, 235, 250, 350, 125]
plt.plot(year, profit)
plt.show()


# In[3]:


customer = ["Customer 1", "Customer 2", "Customer 3", "Customer 4", "Customer 5"]


# In[4]:


sales = [1234, 2314, 3456, 5445, 3245]


# In[5]:


plt.plot(customer, sales, 'red')
plt.plot(customer,profit, 'blue')
plt.xlabel('customer')
plt.title('Customer vs Sales and Profit')
plt.ylabel('Sales and Profit')
plt.show()


# In[9]:


plt.axes([0.05,0.05,0.9,0.9])
plt.plot(customer,sales, 'red')
plt.xlabel('Customers')
plt.title('Customer vs Sales')
plt.ylabel('Sales')
plt.axes([1,0.05,0.9,0.9])
plt.plot(customer,profit, 'blue')
plt.xlabel('Customers')
plt.title('Customer vs Profit')
plt.ylabel('Profit')
plt.show()


# In[10]:


plt.subplot(2,1,1)
plt.plot(customer, sales, 'red')
plt.xlabel('Customer')
plt.ylabel('Sales')
plt.title('Customer by Sales')
plt.subplot(2,1,2)
plt.plot(customer,profit,'blue')
plt.xlabel('Customer')
plt.ylabel('Profit')
plt.title('Customer by Profit')
plt.tight_layout()
plt.show()


# In[13]:


plt.plot(customer, sales)
plt.xlabel('Customers')
plt.ylabel('Sales')
plt.title('Customer by Sales')
plt.xlim(('Customer 3', 'Customer 4'))
plt.ylim((3000,6000))
plt.show()


# In[14]:


plt.plot(customer,sales)
plt.xlabel('Customers')
plt.ylabel('Sales')
plt.title('Customers by Sales')
plt.axis(('Customer 3','Customer 4', 3000, 6000))
plt.show()


# In[15]:


profit = [230, 235, 250, 350, 125]
sales = [1234, 2314, 3456, 5445, 3245]
customer = ["Customer 1", "Customer 2", "Customer 3", "Customer 4", "Customer 5"]
plt.scatter(sales, profit, marker='o', color='blue', label='Sales vs Profit')
plt.legend(loc='upper right')
plt.title('Customer data')
plt.xlabel('Sales in $')
plt.ylabel('Profit in $')
plt.show()


# In[ ]:


"""
location of legend
'upper left' = 2
'center left' = 6
'lower left' = 3
'best' = 0
'upper center' = 9
'center' = 10
'lower center'=8
'upper right'=1
'center right'=7
'lower right' = 4
'right'=5
"""


# In[21]:


profit = [230, 235, 250, 350, 125]
sales = [1234, 2314, 3456, 5445, 3245]
customer = ["Customer 1", "Customer 2", "Customer 3", "Customer 4", "Customer 5"]
plt.scatter(sales, profit, marker='o', color='blue', label='Sales vs Profit')
plt.legend(loc=2)
plt.title('Customer data')
plt.xlabel('Sales in $')
plt.ylabel('Profit in $')
plt.show()


# In[38]:


plt.scatter(sales,profit,marker ='o', color='blue', label='Sales vs Profit')
plt.legend(loc=2)
plt.title('Customers data')
plt.xlabel('Sales in $')
plt.ylabel('Profit in $')
plt.annotate('Customer 1', xy=(1234,230))
plt.show()
plt.style.use('tableau-colorblind10')


# In[29]:


print(plt.style.available)


# In[40]:


import os
os.chdir('C:\\Users\\HP\\Desktop\\Tosh Viz\\')


# In[42]:


df_retail = pd.read_excel('Sample - Superstore.xls')


# In[43]:


df_retail


# In[44]:


fig, ax = plt.subplots()
ax.hist(df_retail['Sales'])
plt.show()


# In[45]:


type(df_retail['Sales'])


# In[47]:


df_retail['Sales'].plot.hist()


# In[48]:


sns.set()
df_retail['Sales'].plot.hist()


# In[49]:


sns.distplot(df_retail['Sales'])


# In[52]:


sns.distplot(df_retail['Sales'], kde=False, bins=30)


# In[58]:


sns.distplot(df_retail['Quantity'])


# In[59]:


fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7,4))
sns.distplot(df_retail['Quantity'], ax=ax0)
sns.distplot(df_retail['Quantity'], ax=ax1)
ax1.set(xlabel='Quantity', xlim=(0,10))
ax1.axvline(x=5.5, label='My Target', linestyle='--')
ax1.legend()


# In[61]:


fig, ax = plt.subplots()
sns.distplot(df_retail['Quantity'])
ax.set(xlabel='Quantity', ylabel='Distribution', 
       xlim=(0,10), title='Quantity and Distribution')
plt.show()


# In[64]:


sns.set_style('white')
sns.distplot(df_retail['Quantity'])
sns.despine(left=False, right=False)


# In[71]:


sns.set(color_codes=True)
sns.distplot(df_retail['Quantity'], color='g')


# In[72]:


for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.show()


# In[73]:


#Circular colors when the data is not ordered
sns.palplot(sns.color_palette("Paired",12))


# In[79]:


#Sequential colors when the data has a consistent range from low to high
sns.palplot(sns.color_palette("Blues",12))


# In[80]:


#Diverging color when both the low and high have its own importance
sns.palplot(sns.color_palette("BrBG",12))


# In[81]:


for style in ['white','dark','whitegrid','darkgrid','ticks']:
    sns.set_style(style)
    sns.distplot(df_retail['Quantity'])
    plt.show()


# In[82]:


sns.distplot(df_retail['Quantity'], hist=False, rug=True)
plt.show()


# In[83]:


sns.distplot(df_retail['Quantity'], hist=False, rug=True, kde_kws={'shade':True})
plt.show()


# In[84]:


sns.regplot(x='Sales', y='Profit', data=df_retail)
plt.show()


# In[89]:


sns.regplot(data=df_retail, x='Sales', y='Profit', marker='+')
plt.show()


# In[86]:


sns.lmplot(x='Sales', y='Profit', data=df_retail)
plt.show()


# In[87]:


sns.lmplot(x='Sales', y='Profit', data=df_retail, hue='Segment')
plt.show()


# In[90]:


sns.lmplot(x='Sales', y='Profit', data=df_retail, col='Segment')
plt.show()


# In[91]:


sns.jointplot(x='Sales', y='Profit', data=df_retail)
plt.show()


# In[93]:


sns.jointplot(x='Sales', y='Quantity', data=df_retail, 
              kind='kde')
plt.show()


# In[95]:


sns.countplot(data=df_retail, y='Category', 
              hue='Sub-Category')
plt.show()


# In[96]:


sns.pointplot(data=df_retail, y='Sales', x='Sub-Category')
plt.show()


# In[97]:


sns.countplot(data=df_retail, x='Category', hue='Sub-Category')
plt.show()


# In[98]:


sns.barplot(data=df_retail, x='Category', y='Sales')
plt.show()


# In[100]:


sns.barplot(data=df_retail, x='Category', y='Profit', hue='Sub-Category')
plt.show()


# In[101]:


sns.stripplot(y='Sales', data=df_retail)
plt.show()


# In[102]:


sns.stripplot(x='Sub-Category', y='Sales', 
              data=df_retail)
plt.show()


# In[104]:


sns.stripplot(x='Sub-Category', y='Sales', 
              data=df_retail, size=10, jitter=False)


# In[106]:


sns.swarmplot(x='Category', y='Sales', data=df_retail)
plt.show()


# In[107]:


plt.subplot(1,2,1)
sns.boxplot(x='Category', y='Sales', data=df_retail)
plt.ylabel('Sales')
plt.subplot(1,2,2)
sns.violinplot(x='Category', y='Sales', data=df_retail)
plt.ylabel('Sales')
plt.tight_layout()
plt.show()


# In[108]:


sns.violinplot(x='Category', y='Sales', data=df_retail, inner=None, color='lightgray')
sns.stripplot(x='Category', y='Sales', data=df_retail, size=4, jitter=True)
plt.ylabel('Sales')
plt.show()


# In[113]:


df_pp = pd.read_excel('pairplot.xls')


# In[114]:


df_pp


# In[115]:


sns.pairplot(df_pp)


# In[119]:


covariance = df_pp.corr()


# In[120]:


covariance


# In[118]:


sns.heatmap(covariance)
plt.title('Covariance plot')
plt.show()


# In[122]:


sns.heatmap(covariance, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, linewidths=.5)


# In[144]:


df_ts = pd.read_excel('Timeseries.xls')


# In[145]:


df_ts


# In[146]:


df_ts.set_index('Order Date', inplace=True)


# In[147]:


df_ts


# In[148]:


timeseries = df_ts['Sales']
timeseries.plot()
plt.legend()


# In[151]:


timeseries = df_ts['Sales']
timeseries.rolling(12).mean().plot(label='12 months rolling mean')
timeseries.rolling(12).std().plot(label='12 month rolling std')
timeseries.plot()
plt.legend()
plt.show()

