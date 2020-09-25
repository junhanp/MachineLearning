#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Bio import SeqIO
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import pandas as pd


# In[2]:


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strand lengths are not equal!")
    return sum(ch1 != ch2 for ch1,ch2 in zip(s1,s2))


# In[3]:


seqList=[]


# In[4]:


for record in SeqIO.parse("HW2.fas", "fasta"):
    seqList.append(record.seq)


# In[5]:


hdList=[]


# In[6]:


for x in seqList:
    for y in seqList:
        hdList.append(hamming_distance(x,y))


# In[7]:


len(hdList)


# In[8]:


matrix=np.array(hdList)
matrix


# In[9]:


matrix.shape


# In[10]:


matrix=matrix.reshape(120,120)


# In[11]:


matrix


# In[12]:


embedding=MDS(n_components = 2)


# In[13]:


xtransform=embedding.fit_transform(matrix[:120])


# In[14]:


xtransform.shape


# In[15]:


xtransform


# In[16]:


df=pd.DataFrame(xtransform)
df


# In[17]:


x=df[0]


# In[18]:


y=df[1]


# In[19]:


plt.scatter(x,y)


# Estimate: K=3

# In[20]:


kmeans=KMeans(n_clusters=3, random_state=0).fit(df)


# In[21]:


kmeans.labels_


# In[22]:


kmeans=kmeans.cluster_centers_


# In[23]:


df=pd.DataFrame(kmeans)
df


# In[24]:


x=df[0]


# In[25]:


y=df[1]


# In[26]:


plt.scatter(x,y, c=['red','blue','green'])


# In[ ]:




