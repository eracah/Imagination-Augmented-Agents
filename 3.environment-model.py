#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from common.actor_critic import ActorCritic
from common.multiprocessing_env import SubprocVecEnv
from common.minipacman import MiniPacman


# In[2]:


from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h4>USE CUDA</h4>

# In[3]:


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


# <h3>Pixels and Rewards</h3>

# <p>Outputs of the Environment Model is trained to predict the next frame and reward by stochastic gradient decent on the Bernoulli cross-entropy between network outputs and data.<br>
# </p>

# In[4]:


#7 different pixels in MiniPacman
pixels = (
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0), 
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0), 
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
)
pixel_to_categorical = {pix:i for i, pix in enumerate(pixels)} 
num_pixels = len(pixels)

#For each mode in MiniPacman there are different rewards
mode_rewards = {
    "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9]
}
reward_to_categorical = {mode: {reward:i for i, reward in enumerate(mode_rewards[mode])} for mode in mode_rewards.keys()}


# In[5]:


def pix_to_target(next_states):
    target = []
    for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
        target.append(pixel_to_categorical[tuple([np.ceil(pixel[0]), np.ceil(pixel[1]), np.ceil(pixel[2])])])
    return target

def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_categorical.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_categorical[mode][reward])
    return target

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('loss %s' % losses[-1])
    plt.plot(losses)
    plt.show()
    
def displayImage(image, step, reward):
    s = str(step) + " " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()


# <h2>Environment Model</h2>
# <p>The input and output frames are of size 3 x 15 x 19 (RGB x width x height). The model consisted of a size preserving, multi-scale CNN architecture with additional fully connected layers for reward prediction. <br>In order to capture long-range dependencies across pixels, we also make use of a layer called pool-and-inject, which applies global max-pooling over each
# feature map and broadcasts the resulting values as feature maps of the same size and concatenates the
# result to the input. Pool-and-inject layers are therefore size-preserving layers which communicate the
# max-value of each layer globally to the next convolutional layer.</p>

# In[6]:


class BasicBlock(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super(BasicBlock, self).__init__()
        
        self.in_shape = in_shape
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=1, stride=2, padding=6),
            nn.ReLU(),
            nn.Conv2d(n1, n1, kernel_size=10, stride=1, padding=(5, 6)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2,  n3, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, inputs):
        x = self.pool_and_inject(inputs)
        x = torch.cat([self.conv1(x), self.conv2(x)], 1)
        x = self.conv3(x)
        x = torch.cat([x, inputs], 1)
        return x
    
    def pool_and_inject(self, x):
        pooled     = self.maxpool(x)
        tiled      = pooled.expand((x.size(0),) + self.in_shape)
        out        = torch.cat([tiled, x], 1)
        return out


# In[7]:


class EnvModel(nn.Module):
    def __init__(self, in_shape, num_pixels, num_rewards):
        super(EnvModel, self).__init__()
        
        width  = in_shape[1]
        height = in_shape[2]
        
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.ReLU()
        )
        
        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU()
        )
        self.image_fc = nn.Linear(256, num_pixels)
        
        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reward_fc    = nn.Linear(64 * width * height, num_rewards)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        
        image = self.image_conv(x)
        image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        image = self.image_fc(image)

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)
        
        return image, reward


# <h3>Creating environments</h3>

# In[15]:


mode = "regular"
num_envs = 16

def make_env():
    def _thunk():
        env = MiniPacman(mode, 1000)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = envs.observation_space.shape
num_actions = envs.action_space.n


# In[9]:


env_model    = EnvModel(envs.observation_space.shape, num_pixels, len(mode_rewards["regular"]))
actor_critic = ActorCritic(envs.observation_space.shape, envs.action_space.n)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(env_model.parameters())

if USE_CUDA:
    env_model    = env_model.cuda()
    actor_critic = actor_critic.cuda()


# <h3>Loading pretrained Actor Critic from previous notebook.</h3>

# In[10]:


actor_critic.load_state_dict(torch.load("actor_critic_" + mode))


# In[11]:


def get_action(state):
    if state.ndim == 4:
        state = torch.FloatTensor(np.float32(state))
    else:
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
        
    action = actor_critic.act(Variable(state, volatile=True))
    action = action.data.cpu().squeeze(1).numpy()
    return action


# In[12]:


def play_games(envs, frames):
    states = envs.reset()
    
    for frame_idx in range(frames):
        actions = get_action(states)
        next_states, rewards, dones, _ = envs.step(actions)
        
        yield frame_idx, states, actions, rewards, next_states, dones
        
        states = next_states


# <h2>Training</h2>

# In[16]:


reward_coef = 0.1
num_updates = 5000

losses = []
all_rewards = []

for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, num_updates):
    states      = torch.FloatTensor(states)
    actions     = torch.LongTensor(actions)

    batch_size = states.size(0)
    
    onehot_actions = torch.zeros(batch_size, num_actions, *state_shape[1:])
    onehot_actions[range(batch_size), actions] = 1
    inputs = Variable(torch.cat([states, onehot_actions], 1))
    
    if USE_CUDA:
        inputs = inputs.cuda()

    imagined_state, imagined_reward = env_model(inputs)

    target_state = pix_to_target(next_states)
    target_state = Variable(torch.LongTensor(target_state))
    
    target_reward = rewards_to_target(mode, rewards)
    target_reward = Variable(torch.LongTensor(target_reward))

    optimizer.zero_grad()
    image_loss  = criterion(imagined_state, target_state)
    reward_loss = criterion(imagined_reward, target_reward)
    loss = image_loss + reward_coef * reward_loss
    loss.backward()
    optimizer.step()
    
    losses.append(loss.data[0])
    all_rewards.append(np.mean(rewards))
    
    if frame_idx % 10 == 0:
        plot(frame_idx, all_rewards, losses)


# <h2>Saving the model</h2>

# In[52]:


torch.save(env_model.state_dict(), "env_model_" + mode)


# <h2>Imagination!</h2>

# In[55]:


import time

env = MiniPacman(mode, 1000)
batch_size = 1

done = False
state = env.reset()
iss = []
ss  = []

steps = 0

while not done:
    steps += 1
    actions = get_action(state)
    onehot_actions = torch.zeros(batch_size, num_actions, *state_shape[1:])
    onehot_actions[range(batch_size), actions] = 1
    state = torch.FloatTensor(state).unsqueeze(0)
    
    inputs = Variable(torch.cat([state, onehot_actions], 1))
    if USE_CUDA:
        inputs = inputs.cuda()

    imagined_state, imagined_reward = env_model(inputs)
    imagined_state = F.softmax(imagined_state)
    iss.append(imagined_state)
    
    next_state, reward, done, _ = env.step(actions[0])
    ss.append(state)
    state = next_state
    
    imagined_image = target_to_pix(imagined_state.view(batch_size, -1, len(pixels))[0].max(1)[1].data.cpu().numpy())
    imagined_image = imagined_image.reshape(15, 19, 3)
    state_image = torch.FloatTensor(next_state).permute(1, 2, 0).cpu().numpy()
    
    clear_output()
    plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.title("Imagined")
    plt.imshow(imagined_image)
    plt.subplot(132)
    plt.title("Actual")
    plt.imshow(state_image)
    plt.show()
    time.sleep(0.3)
    
    if steps > 30:
        break

