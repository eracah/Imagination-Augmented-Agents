#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from common.multiprocessing_env import SubprocVecEnv
from common.minipacman import MiniPacman
from common.environment_model import EnvModel
from common.actor_critic import OnPolicy, ActorCritic, RolloutStorage
import wandb
from pathlib import Path
from types import SimpleNamespace
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import namedtuple

device = "cuda" if torch.cuda.is_available() else "cpu"

args = SimpleNamespace(mode = "regular",
num_envs = 16,
gamma = 0.99,
entropy_coef = 0.01,
value_loss_coef = 0.5,
max_grad_norm = 0.5,
num_steps = 5,
num_frames = int(10e5),
lr    = 7e-4,
eps   = 1e-5,
alpha = 0.99,
full_rollout = True)




pixels = (
    (0.0, 1.0, 0.0), 
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0), 
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0)
)
pixel_to_onehot = {pix:i for i, pix in enumerate(pixels)} 
num_pixels = len(pixels)

task_rewards = {
    "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9]
}
reward_to_onehot = {mode: {reward:i for i, reward in enumerate(task_rewards[mode])} for mode in task_rewards.keys()}

def pix_to_target(next_states):
    target = []
    for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
        target.append(pixel_to_onehot[tuple([np.round(pixel[0]), np.round(pixel[1]), np.round(pixel[2])])])
    return target

def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_onehot.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_onehot[mode][reward])
    return target
    
def displayImage(image, step, reward):
    s = str(step) + " " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()




# In[6]:




def make_env():
    def _thunk():
        env = MiniPacman(args.mode, 1000)
        return env

    return _thunk

envs = [make_env() for i in range(args.num_envs)]
envs = SubprocVecEnv(envs)

state_shape = envs.observation_space.shape
num_actions = envs.action_space.n
num_rewards = len(task_rewards[args.mode])


# <h1>I2A components</h1> 

# <p>The Rollout Encoder is an GRU with convolutional encoder which sequentially processes
# a trajectory</p>

# In[7]:


class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size):
        super(RolloutEncoder, self).__init__()
        
        self.in_shape = in_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        self.gru = nn.GRU(self.feature_size() + num_rewards, hidden_size)
        
    def forward(self, state, reward):
        num_steps  = state.size(0)
        batch_size = state.size(1)
        
        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)
    
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


# <p>For the model-free path of the I2A, it's used a standard network of convolutional layers plus one fully
# connected one</p>

# In[8]:


class I2A(OnPolicy):
    def __init__(self, in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True):
        super(I2A, self).__init__()
        
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        
        self.imagination = imagination
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)
        
        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, 256),
                nn.ReLU(),
            )
        
        self.critic  = nn.Linear(256, 1)
        self.actor   = nn.Linear(256, num_actions)
        
    def forward(self, state):
        batch_size = state.size(0)
        
        imagined_state, imagined_reward = self.imagination(state.data)
        hidden = self.encoder(imagined_state, imagined_reward)
        hidden = hidden.view(batch_size, -1)
        
        state = self.features(state)
        state = state.view(state.size(0), -1)
        
        x = torch.cat([state, hidden], 1)
        x = self.fc(x)
        
        logit = self.actor(x)
        value = self.critic(x)
        
        return logit, value
        
    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


# <p>The imagination core (IC) predicts the next time step conditioned on an action sampled from the rollout policy (distil_policy).<br>
# See Figure 1 a. in the paper
# </p>

# In[9]:


class ImaginationCore(object):
    def __init__(self, num_rolouts, in_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=True):
        self.num_rolouts  = num_rolouts
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.env_model     = env_model
        self.distil_policy = distil_policy
        self.full_rollout  = full_rollout
        
    def __call__(self, state):
        state      = state.cpu()
        batch_size = state.size(0)

        rollout_states  = []
        rollout_rewards = []

        if self.full_rollout:
            state = state.unsqueeze(0).repeat(self.num_actions, 1, 1, 1, 1).view(-1, *self.in_shape)
            action = torch.LongTensor([[i] for i in range(self.num_actions)]*batch_size)
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        else:
            action = self.distil_policy.act(state)
            action = action.data.cpu()
            rollout_batch_size = batch_size

        for step in range(self.num_rolouts):
            onehot_action = torch.zeros(rollout_batch_size, self.num_actions, *self.in_shape[1:])
            onehot_action[range(rollout_batch_size), action] = 1
            inputs = torch.cat([state, onehot_action], 1)

            imagined_state, imagined_reward = self.env_model(inputs)

            imagined_state  = F.softmax(imagined_state, dim=1).max(1)[1].data.cpu()
            imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1].data.cpu()

            imagined_state = target_to_pix(imagined_state.numpy())
            imagined_state = torch.FloatTensor(imagined_state).view(rollout_batch_size, *self.in_shape)

            onehot_reward = torch.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(onehot_reward.unsqueeze(0))

            state  = imagined_state
            action = self.distil_policy.act(state)
            action = action.data.cpu()
        
        return torch.cat(rollout_states), torch.cat(rollout_rewards)


# <h3>Full Rollout</h3>
# <p>
# if full_rollout == True: perform rollout for each possible action in the environment. <br>
# if full_rollout == False: perform rollout for one action from distil policy.
# </p>

# In[10]:

wandb.init(project="mbrl", dir=".", tags=["I2A"])
wandb.config.update(vars(args))


# In[11]:


env_model     = EnvModel(envs.observation_space.shape, num_pixels, num_rewards)
if Path("env_model_" + args.mode).exists():
    env_model.load_state_dict(torch.load("env_model_" + args.mode))

distil_policy = ActorCritic(envs.observation_space.shape, envs.action_space.n)
distil_optimizer = optim.Adam(distil_policy.parameters())

imagination = ImaginationCore(1, state_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=args.full_rollout)

actor_critic = I2A(state_shape, num_actions, num_rewards, 256, imagination, full_rollout=args.full_rollout)
#rmsprop hyperparams:
optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)



env_model     = env_model.to(device)
distil_policy = distil_policy.to(device)
actor_critic  = actor_critic.to(device)


# <h2>Training</h2>

# In[12]:



rollout = RolloutStorage(args.num_steps, args.num_envs, envs.observation_space.shape, device=device)


all_rewards = []
all_losses  = []


# In[13]:


state = envs.reset()
current_state = torch.FloatTensor(np.float32(state))

rollout.states[0].copy_(current_state)

episode_rewards = torch.zeros(args.num_envs, 1)
final_rewards = torch.zeros(args.num_envs, 1)

for i_update in range(args.num_frames):

    for step in range(args.num_steps):

        current_state = current_state.to(device)
        action = actor_critic.act(current_state)

        next_state, reward, done, _ = envs.step(action.cpu().data.numpy())

        reward = torch.FloatTensor(reward).unsqueeze(1)
        episode_rewards += reward
        masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1).to(device)
        final_rewards *= masks
        final_rewards += (1-masks) * episode_rewards
        episode_rewards *= masks
        

        current_state = torch.FloatTensor(np.float32(next_state))
        rollout.insert(step, current_state, action.data, reward, masks)


    _, next_value = actor_critic(rollout.states[-1])
    next_value = next_value.data

    returns = rollout.compute_returns(next_value, args.gamma)

    logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
        rollout.states[:-1].view(-1, *state_shape),
        rollout.actions.view(-1, 1)
    )
    
    distil_logit, _, _, _ = distil_policy.evaluate_actions(
        rollout.states[:-1].view(-1, *state_shape),
        rollout.actions.view(-1, 1)
    )
        
    distil_loss = 0.01 * (F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)).sum(1).mean()

    values = values.view(args.num_steps, args.num_envs, 1)
    action_log_probs = action_log_probs.view(args.num_steps,args.num_envs, 1)
    advantages = returns - values

    value_loss = advantages.pow(2).mean()
    action_loss = -(advantages.data * action_log_probs).mean()

    optimizer.zero_grad()
    loss = value_loss * args.value_loss_coef + action_loss - entropy * args.entropy_coef
    loss.backward()
    nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
    optimizer.step()
    
    distil_optimizer.zero_grad()
    distil_loss.backward()
    optimizer.step()
    wandb.log({"loss": loss.item()})
    wandb.log({"reward": final_rewards.mean()})
    
    if i_update % 100 == 0:
        all_rewards.append(final_rewards.mean())
        all_losses.append(loss.data)
        print("Update {}:".format(i_update))
        print("\t Mean Loss: {}".format(np.mean(all_losses[-10:])))
        print("\t Last 10 Mean Reward: {}".format(np.mean(all_rewards[-10:])))
        wandb.log({"Mean Reward":np.mean(all_rewards[-10:])})
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))
        plt.plot(all_rewards)
        plt.subplot(132)
        plt.title('loss %s' % all_losses[-1])
        plt.plot(all_losses)
        plt.show()
        
    rollout.after_update()


# <h2>Save the model</h2>

# In[11]:


torch.save(actor_critic.state_dict(), wandb.run.dir + "/i2a_" + args.mode)

