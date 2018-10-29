
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


w = tf.Variable(0,dtype=tf.float32)
cost = tf.add(tf.add(w**2,tf.multiply(-10.0,w)),25.0)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# In[5]:


init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(w)
session.run(train)


# In[6]:


print("W after one iteration:",session.run(w))
for i in range(1,1000):
    session.run(train)

print("W after 1000 iteration:", session.run(w))

