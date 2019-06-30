#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('D:\\Cognitior\\Courses\\Dataset\\Python DataScience\\')


# In[2]:


import pandas as pd


# In[3]:


dataset = pd.read_csv('data.csv')


# In[4]:


dataset


# In[13]:


x=dataset.iloc[:,:-1].values


# In[16]:


from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])


# In[18]:


onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()


# In[20]:


pd.DataFrame(x)


# In[21]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


# In[23]:


pd.DataFrame(x)


# In[24]:


from sklearn.preprocessing import Normalizer
nm_x = Normalizer()
x = nm_x.fit_transform(x)

