#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB 


# In[2]:


P1="I’ve seen a lot of bad reviews for this phone based on issues with the seller. Granted, some of those reviews say it took a few weeks for the problems to appear so I’ll edit this if that happens, but wow was I happy with what I got. Not only did it come with a charger (there’s some debate on that in other reviews), but it even had a clear bumper case. That was neither expected nor necessary but I appreciated it (I bought a Unicorn Beetle case which I have used and loved before on other phones). There wasn’t a scratch on this phone and it started working right away for me. The battery seems to be holding up fine. All in all I’d say this seems like a steal. If it self destructs on me in the next few weeks I’ll update this. UPDATE: It's been a few months and a trip overseas since I wrote that initial review, and it remains a solid decision I'm very happy with."


# In[3]:


P2="This phone looks and performs great like it's brand new. Not one scratch. The phone came with a screen protector and a charger. I was surprised as other reviews said they did not get one. For $269, I feel like this was a steal, compared to other listings. Hopefully nothing goes wrong with the phone later. But with the Amazon 90 day guarantee I'm a little more at ease about possible return. Never bought a refurbished phone before. Not sure what to expect. As far as my order, I am happy with it."


# In[4]:


P3="Don't listen to bad reviews! My phone arrived in great condition. There are no scratches on the glass, and there is no visible wear and tear on the case. It works perfectly. I inserted my carrier-provided SIM card in the SIM tray and it was immediately available on AT&T's network. A SIM tray key was included in the box along with protective plastic covers for the screen. A charging cable and standard outlet plug were also included in the box. This version of the iPhone does not have a headphone jack. I did not receive a headphone insert in the box, but (#1) I don't need one as all of my headphones are Bluetooth and (#2) I don't know if Apple included this in the original packaging so this is just a courtesy note for potential buyers of the iPhone 7, not a complaint. The seller contacted me after I received my phone to make sure I was happy with the purchase and I am."


# In[5]:


P4="Love this phone! I am so glad I bought a refurbished one. I took it to the Apple store just in case to do a diagnostics on it and said that it was refurbished and bought through Amazon, and Apple checked it and said everything is great. Very happy with my purchase."


# In[6]:


P5="First, seller did a great job and I think I got a good price for an iPhone 7, I just think ALL CELL PHONES are way way way too expensive. When a Cell phone costs more than a good laptop computer that is too expensive. Second all Smart phones have bad battery life. Apple's iPhones are no exception. There is a mode on the iPhone 7 to allow for an extended battery life setting. But I see no difference between the extended setting and the normal setting. I do not use my phone except for emergencies so I would expect the phone to last 5-6 days between charging, but I am averaging 3-4 days between charging. I am having an issue that the WiFi doesn't see both of my wireless networks (dual band router). Seller tried to help but Apple's support said if it sees a network that's all they care about. Phone appears to be working fine and so far I am happy with it."


# In[7]:


P6="Received prompt delivery of the phone. I inserted my 'sim card' and the phone was functional with no issues and I could make and receive calls right away, so far so good. I received the phone which is cosmetically in very good condition and I am quite happy with my purchase with exception of two minor issues which I believe someone could provide me guidance to resolve or trouble shoot."


# In[8]:


N1="Overall, the phone isn't too bad for the price. It came already scratched up, overheats more than a normal iPhone (I've had tons of iPhones). The delivery process of just getting the phone was pretty stressful, I'm a month and half in using the iPhone and I called customer service to see if they could replace my iphone because it got to the point where my hands feel the burning from the phone... the lady was so unhelpful, bland and kind of rude. The return proccess would be such a hassle and leave me phoneless so I decided to keep the phone instead. All the functions work fine, it's just that the iphone started heating up the moment I got it. I don't usually write reviews no matter how good or bad a product is, but I've never received such bad service from a company, especially amazon sellers. I'm basically stuck with the phone, or be phoneless. I would recommend the phone, but just know there will definitely some things you need to deal with. HAVE A NICE DAY TO WHOEVER IS READING :)"


# In[9]:


N2="The iPhone 7 I purchased was \"certified refurbished\" and labeled as \"new\" quality but doesn't work. The phone looks great, but when I first turned it on it was in a restart loop. This was a bad sign to begin with, but I gave it the benefit of the doubt and connected it to my computer. When I finally got it to restore to factory settings, the screen started glitching to the point where there was nothing to stop it, and if it did get to the startup screen, it was non-responsive."


# In[10]:


N3="Initially I was happy with the phone. It looked great physically and had no signs of wear and tear. However, the battery health was lower than I wanted; the phone said the battery health was 88%. However, I knew from the ad, that it could ship with as low as 85%, so I can't complain too much about that. The biggest issue with the phone that was an absolute deal breaker was that it frequently crashed and closed apps on me. Other times it would freeze up. Imagine having an emergency and having to make a phone call, only to find out that your phone decided to freeze up?! I have a family, so that's completely unacceptable. The phone also seemed to have connectivity issues and would not connect well with my wifi. It was slower than my other devices on my wifi and would sometimes freeze up. With the problems that I was having, I'm thinking it was a bad main board or \"motherboard\". The seller was MobileSpree. I contacted them and asked for an exchange. They refused to do an exchange and said my only option was to return it. I returned it with the shipping label provided by Amazon. However, even after 5 days of having the phone back, they would not refund my money. I had to get Amazon involved to get a refund. Overall, don't buy. It was a waste of time and money and a hassle to get refunded."


# In[11]:


N4="Be cautious - if you have ANY issues at all, return phone immediately. We got one for my daughter, paid $244 and it didn’t last 4 months. Seller will not replace/return as it is past 90 days. She had intermittent issues with service connections shortly after receiving the phone. When it finally stopped connecting at all and we had it checked at the AT&T store, they told us it was an internal issue with the SIM card brackets that connects to the mother board. Basically causing a fatal error and cannot get any service connection. I contacted the seller and received the generic “past the 90 day warranty” so there is nothing they will do about it. We may try to have it repaired, but the repair shop is looking at $100 to inspect and possibly repair, if it can be repaired. I guess that’s our expensive mistake, but at least we can warn others."


# In[12]:


p1=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",P1)
p2=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",P2)
p3=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",P3)
p4=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",P4)
p5=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",P5)
p6=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",P6)
n1=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",N1)
n2=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",N2)
n3=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",N3)
n4=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",N4)

# print(p1)
# print(p2)
# print(p3)
# print(p4)
# print(p5)
# print(p6)
# print(n1)
# print(n2)
# print(n3)
# print(n4)


# In[13]:


data=pd.DataFrame(index=np.arange(5),columns=np.arange(0))


# In[14]:


data['p1']=pd.Series(p1)
data['p2']=pd.Series(p2)
data['p3']=pd.Series(p3)
data['p4']=pd.Series(p4)
data['p5']=pd.Series(p5)
data['p6']=pd.Series(p6)
data['n1']=pd.Series(n1)
data['n2']=pd.Series(n2)
data['n3']=pd.Series(n3)
data['n4']=pd.Series(n4)

data


# In[15]:


data = data.transpose()
data


# In[16]:


data['result']=['positive','positive','positive','positive','positive','positive','negative','negative','negative','negative']


# In[17]:


data


# In[18]:


data=data.fillna(0)


# In[19]:


data=data.replace(to_replace ='great', value =1) 
data=data.replace(to_replace ='happy', value =1) 
data=data.replace(to_replace ='bad', value =-1) 
data=data.replace(to_replace ='return', value =-1) 

# data=data.replace(to_replace ='positive', value =1) 
# data=data.replace(to_replace ='negative', value =-1) 
data


# In[20]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[21]:


#result = ['Positive','Positive','Positive','Positive','Positive','Positive','Negative','Negative','Negative','Negative']
result=data['result']
data=data.drop(columns=['result'])
data['sum']=data.sum(axis=1)
#if sum is 0 replace it with -2
data['sum']=data['sum'].replace(to_replace =0, value =-2)

data


# In[22]:


X=data
y=result


# In[23]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.partial_fit(X, y, np.unique(y)) 


# In[24]:


print(gnb.predict([[1,-1,0,0,0,-2]]))


# In[25]:


U1="The phone arrived in pretty decent condition. The front screen was scratch-free and the display is great, but there is a long scratch on the back of the phone. This doesn't bother me much because I always have a case on my phone. However, the issue with this phone is that the cellular signal won't work; the device detects the sim but the signal is bad. Apparently this is an issue with some iPhone 7 models, but the any free of charge repair is not valid because the phone is coming from a third party seller. After speaking with Apple, Verizon (my mobile carrier), AND Amazon, I've reached the conclusion that the issue is with the phone. I've tried everything to troubleshoot, but I will unfortunately have to return the item and get another one."


# In[26]:


U2="iPhone 7 Black came in excellent condition. Like new. No scratches or scuffs. Works great. Was happy for couple months until phone started to develop issues with hearing callers and vs versa. Callers can’t hear me and I can’t hear callers, the sound is bad. Checked settings . Disabled WiFi calling. Hard reset phone. Updated iOS. Happens randomly. Suspect possible known defects on iPhone 7 with audio IC chips. I want to return the phone but I’m waiting to se for a month"


# In[27]:


u1=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",U1)
u2=re.findall("Great|great|Happy|happy|Bad|bad|Return|return",U2)
print(u1)
print(u2)


# In[28]:


print(gnb.predict([[1,-1,-1,0,0,-1]]))
print(gnb.predict([[1,1,-1,-1,0,-2]]))

