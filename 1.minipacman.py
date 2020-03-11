#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from minipacman import MiniPacman


# In[2]:


from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def displayImage(image, step, reward):
    s = "step" + str(step) + " reward " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()


# In[4]:


keys = {
    'w': 2,
    'd': 1,
    'a': 3,
    's': 4,
    ' ': 0
}


# <p>
# 
# W - up <br>
# A - left <br>
# D - right <br>
# S - down <br>
# 
# </p>

# In[ ]:


MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
frame_cap = 1000

mode = 'rush'

env = MiniPacman(mode, 1000)

state = env.reset()
done = False

total_reward = 0
step = 1

displayImage(state.transpose(1, 2, 0), step, total_reward)

while not done:
    x = raw_input()
    clear_output()
    try:
        keys[x]
    except:
        print "Only 'w' 'a' 'd' 's'"
        continue
    action = keys[x]
    
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    displayImage(next_state.transpose(1, 2, 0), step, total_reward)
    step += 1


# In[ ]:




