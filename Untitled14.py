#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np


# In[62]:


data=pd.read_csv(r"C:\Users\Aryan\Desktop\python\Titanic (1).csv")
data.head(4)


# In[63]:


data.info()


# In[64]:


mean_age=data.age.mean()
mean_fare=data.fare.mean()
mean_fare


# In[65]:


data.age.fillna(mean_age,inplace=True)
data.fare.fillna(mean_fare,inplace=True)


# In[66]:


data.info()


# In[67]:


new_data=pd.get_dummies(data)
new_data.info()


# In[68]:


new_data.head(10)


# In[110]:


x=new_data.drop(["survived"],axis='columns')
y=new_data.survived


# In[112]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5 ,random_state=42)


# In[114]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()


# In[116]:


log.fit(x_train,y_train)


# In[118]:


log.score(x_train,y_train)







