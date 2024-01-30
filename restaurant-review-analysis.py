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

# In[25]:


get_ipython().system('pip install nltk matplotlib pandas seaborn wordcloud')


# In[26]:


get_ipython().system('pip install tk')


# In[28]:


get_ipython().system('pip install Pillow')


# In[71]:


from tkinter import Tk, Label, Button, W, E
import sqlite3

def take_review():
    # Implement the functionality for the take_review function
    pass

def login():
    # Implement the functionality for the login function
    pass

root1 = Tk()
main = "Restaurant Review Analysis System/"
root1.title(main + "Welcome Page")

# Set the background color of the Tkinter window
root1.configure(bg='#f0f0f0')  # Replace '#f0f0f0' with the desired background color code

label = Label(root1, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
              bd=2, font=('Arial', 46, 'bold', 'underline'), bg='#f0f0f0')

ques = Label(root1, text="Are you a Customer or Owner ???", bg='#f0f0f0')

cust = Button(root1, text="Customer", font=('Arial', 20),
              padx=80, pady=20, command=take_review, bg='#4CAF50', fg='white')  # Set background and foreground color

owner = Button(root1, text="Owner", font=('Arial', 20),
               padx=100, pady=20, command=login, bg='#008CBA', fg='white')  # Set background and foreground color

# Define the database connection
conn = sqlite3.connect('Restaurant_food_data.db')
c = conn.cursor()

# Uncomment the following lines if you want to create a table
'''
c.execute("CREATE TABLE item (Item_name text,No_of_customers text,\
            No_of_positive_reviews text,No_of_negative_reviews text,\
            Positive_percentage text,Negative_percentage text) ")
'''

# Uncomment the following line if you want to delete data from the table
# c.execute("DELETE FROM item")

root1.state('zoomed')  # Maximize the window
label.grid(row=0, column=0)
ques.grid(row=1, column=0, sticky=W + E)
ques.config(font=("Helvetica", 30))
cust.grid(row=2, column=0)
owner.grid(row=3, column=0)

# Commit and close the database connection
conn.commit()
conn.close()

root1.mainloop()
