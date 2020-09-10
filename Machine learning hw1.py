#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


from matplotlib.image import imread


# In[3]:


import numpy as np


# In[4]:


import cv2 as cv


# In[5]:


import os


# In[6]:


from numpy import array
from scipy.linalg import svd


# In[7]:


#plt.rcParams['figure.figsize']=[16,8]
A = imread('image4.jpg')
#X = np.mean(A)

image = plt.imshow(A)
image.set_cmap('gray')
plt.axis('off')
plt.show()


# In[8]:


U, S, VT = svd(A, full_matrices = False)
#print(U)
#print(S)
#print(VT)


# In[9]:


S = np.diag(S)


# In[14]:


j = 0
for r in (1, 5, 10, 50, 100):
    Xapp = U[:,:r] @ S[:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j+=1
    img = plt.imshow(Xapp)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r - ' +str(r))
    plt.show()


# In[ ]:




