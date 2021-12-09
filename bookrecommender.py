#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from rake_nltk import Rake

import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('booksummaries.txt', sep="\t", header=None)


# In[3]:


data


# In[4]:


data.columns = ['aid', 'fid', 'title', 'author', 'pdate', 'genre', 'summary']


# In[5]:


data


# In[6]:


data.dropna(inplace = True)


# In[7]:


data['genre']


# In[8]:


data.drop('genre',axis='columns', inplace=True)
data.drop('aid',axis='columns', inplace=True)
data.drop('fid',axis='columns', inplace=True)
data.drop('pdate',axis='columns', inplace=True)
data.reset_index(drop=True, inplace=True)


# In[9]:


data


# In[10]:


data


# In[11]:


data['summary'] = data['summary'].str.replace('[^\w\s]','')


# In[12]:


data


# In[13]:


data['plotpoints'] = ''
r = Rake()

for index, row in data.iterrows():
    r.extract_keywords_from_text(row['summary'])
    key_words_dict_scores = r.get_word_degrees()
    row['plotpoints'] = list(key_words_dict_scores.keys())


# In[14]:


data


# In[15]:


data['author'] = data['author'].map(lambda x: x.split(','))


# In[16]:


for index, row in data.iterrows():
    row['author'] = [x.lower().replace(' ','') for x in row['author']]


# In[17]:


data['relinfo'] = ''
columns = ['author', 'plotpoints']
data


# In[18]:


for index, row in data.iterrows():
    w = ''
    for col in columns:
        w += ' '.join(row[col]) + ' '
    row['relinfo'] = w


# In[19]:


data


# In[20]:


data['relinfo']


# In[21]:


data


# In[22]:


data['relinfo'] = data['relinfo'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')
data['relinfo'] = data['relinfo'].str.strip().str.replace('.', ' ').str.replace('  ', ' ')


# In[23]:


data['relinfo']


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data['relinfo'])


# In[25]:


from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[26]:


indices = pd.Series(data['title'])


# In[27]:


def recommend(title, cosine_sim = cosine_sim):
    recommended_club = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_indices = list(score_series.iloc[1:3].index)
 
    for i in top_indices:
        recommended_club.append(list(data['title'])[i])
        
    for classes in recommended_club:
       if title.casefold()!=classes.casefold():
           return recommended_club