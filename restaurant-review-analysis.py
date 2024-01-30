import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


pip install kaggle


# In[3]:


get_ipython().system('kaggle datasets list -s "Restaurant_Reviews"')


# In[4]:


get_ipython().system('kaggle datasets download -d d4rklucif3r/restaurant-reviews')


#  Loading the dataset

# In[5]:


import zipfile


# In[6]:


with zipfile.ZipFile("restaurant-reviews.zip","r") as file:
    file.extractall("restaurant-reviews")


# In[10]:


import os
os.listdir()


# In[8]:


dataset = pd.read_csv("C:/Users/Dell/Desktop/wise project ml/restaurant-reviews/Restaurant_Reviews.tsv",
                      delimiter='\t', quoting=3)


# Data Exploration

# In[9]:


dataset.head()


# In[10]:


dataset.tail()


# In[11]:


dataset.describe()


# In[12]:


dataset.info()


# In[13]:


dataset.shape


# In[14]:


dataset['Liked'].value_counts()


# In[15]:


corpus = []


# Connecting to SQLite database

# In[16]:


import sqlite3

conn = sqlite3.connect('Restaurant_food_data.db')
c = conn.cursor()


