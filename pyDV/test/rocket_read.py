
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

file = open('2.txt')
val_list = file.readlines() 
lists =[]
for string in val_list:
    string = string.split(',',2)
    lists.append(string)
a = np.array(lists)
a = a.astype(float)


# In[10]:


a.shape


# In[11]:


a


# In[15]:


times = []
n=0
while(n<675070):
    time = (1/100000)*n
    n+=1
    times.append(time)


# In[16]:


c = np.array(times)
c.shape


# In[17]:


res = np.c_[c,a]
res.shape


# In[21]:


x= res[0:5000,0]
y= res[0:5000,1]
plt.plot(x,y)
plt.show()

