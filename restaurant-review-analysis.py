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

# In[17]:


import nltk
nltk.download('stopwords')


# Text Preprocessing

# In[18]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]

    review = ' '.join(review)
    corpus.append(review)


# In[20]:


cv = CountVectorizer(max_features=1500)


# Visualizations

# In[21]:


get_ipython().system('pip install wordcloud')


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[23]:


# Bar Chart for Liked/Disliked Reviews
sns.countplot(x='Liked', data=dataset)
plt.title('Distribution of Liked/Disliked Reviews')
plt.show()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'corpus' is your list of preprocessed reviews
review_lengths = [len(review.split()) for review in corpus]
plt.figure(figsize=(10, 6))
sns.histplot(review_lengths, bins=range(min(review_lengths), max(review_lengths) + 2), kde=True, color='skyblue', binwidth=1)
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words in Review')
plt.ylabel('Frequency')

# Customize x-axis ticks and labels
plt.xticks(range(min(review_lengths), max(review_lengths) + 1))

plt.show()


