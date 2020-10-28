#!/usr/bin/env python
# coding: utf-8

# 10 Countries:
# 1. China
# 2. Japan
# 3. Sweden
# 4. Russia
# 5. Argentina
# 6. Ukraine
# 7. Netherlands
# 8. Spain
# 9. Denmark
# 10. Germany

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[2]:


#title = % Population Over 65
O65P = pd.DataFrame()
O65P['Country Name'] = ['China', 'Japan', 'Sweden', 'Russia', 'Argentina', 'Ukraine', 'Netherlands', 'Spain', 'Denmark', 'Germany']
O65P['%'] = [10.6, 27, 19.9, 14.2, 11.2, 16.5, 18.8, 19.4, 19.7, 21.5]
O65P


# In[3]:


Data = pd.read_excel('API_SH.MED.BEDS.ZS_DS2_en_excel_v2_1495658.xls')


# In[4]:


Columns = Data.loc[Data['Data Source'] == 'Country Name'].values
newArr = Columns.reshape(65)
List = newArr.tolist()
List


# In[5]:


Data.columns = List


# In[6]:


Data


# In[7]:


Data = Data.drop([0,1,2])
Data


# In[8]:


Data = Data.reset_index(drop = True)


# In[9]:


Data.columns


# In[10]:


df = pd.DataFrame()
China = Data.loc[Data['Country Name'] == 'China']
Japan = Data.loc[Data['Country Name'] == 'Japan']
Sweden = Data.loc[Data['Country Name'] == 'Sweden']
Russia = Data.loc[Data['Country Name'] == 'Russian Federation']
Argentina = Data.loc[Data['Country Name'] == 'Argentina']
Ukraine = Data.loc[Data['Country Name'] == 'Ukraine']
Netherlands = Data.loc[Data['Country Name'] == 'Netherlands']
Spain = Data.loc[Data['Country Name'] == 'Spain']
Denmark = Data.loc[Data['Country Name'] == 'Denmark']
Germany = Data.loc[Data['Country Name'] == 'Germany']
df = df.append(China)
df = df.append(Japan)
df = df.append(Sweden)
df = df.append(Russia)
df = df.append(Argentina)
df = df.append(Ukraine)
df = df.append(Netherlands)
df = df.append(Spain)
df = df.append(Denmark)
df = df.append(Germany)
df


# In[11]:


beds = pd.DataFrame()


# In[12]:


beds['Country Name'] = df['Country Name']


# In[13]:


noBed = [4.2, 13.4, 2.6, 8.2, 5, 8.8, 4.7, 3, 2.5, 8.3]


# In[14]:


beds['beds1k'] = noBed
beds = beds.reset_index(drop = True)


# In[15]:


case = pd.DataFrame()
case['Country Names'] = beds['Country Name']
response = [0.34, 1.36, 58.26, 17.91, 64.94, 14.71, 41.25, 74.38, 12.11, 12.13]
case['Death/100kPop'] = response
case


# In[16]:


features = pd.DataFrame()
features['Country Names'] = O65P['Country Name']
features['beds1k'] = beds['beds1k']
features['%Over65'] = O65P['%']
features['Death/100kPop'] = case['Death/100kPop']
features


# USA response rate aka Death/100kPop = 68.84

# In[17]:


target = pd.DataFrame()
target['response'] = features['Death/100kPop']


# In[18]:


X = features[['beds1k', '%Over65']]
y = target['response']


# In[19]:


reg = LinearRegression().fit(X,y)
reg.score(X,y)


# In[20]:


reg.coef_


# In[21]:


reg.intercept_


# In[22]:


pred = reg.predict(X)
print(pred)


# In[23]:


reg.predict(np.array([[5, 11.2]]))


# In[24]:


Usa = pd.DataFrame()
Usa['Country Names'] = ['USA']
Usa['beds1k'] = [2.9]
Usa['%Over65'] = [15.4]
Usa['Death/100kPop'] = [68.84]
Usa


# In[25]:


reg.predict(np.array([[2.9, 15.4]]))

