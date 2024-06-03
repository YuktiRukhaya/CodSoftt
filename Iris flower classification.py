#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("Downloads\\iris.csv")
df.head()
df.describe()
df.info()


# In[2]:


df['species'].value_counts()


# In[3]:


df.isnull().sum()


# In[4]:


# histograms
df['sepal_length'].hist()


# In[5]:


df['sepal_width'].hist()


# In[6]:


df['petal_length'].hist()


# In[7]:


df['petal_width'].hist()


# In[8]:


# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[9]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[10]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[11]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[12]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_width'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[13]:


df.corr()
corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df.head()


# In[18]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[19]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# model training
model.fit(x_train, y_train)


# In[20]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[21]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[22]:


model.fit(x_train, y_train)


# In[23]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[24]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[25]:


model.fit(x_train, y_train)


# In[26]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




