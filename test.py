#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:46:10 2021

@author: ocail
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from veh_env import vehEnv

                       
manualSeed = 999 # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)

env = vehEnv(T=20,rho=0.)
env.reset()

done = False

def th_based(x,th=0.01):
    if (x[1] - 4*np.sin(2*np.pi/50*x[0]))**2>th:
        action = 1.0
    else:
        action = 0.0
    return action

current_state = torch.tensor(env.x0)
x = torch.unsqueeze(torch.cat((torch.tensor(env.x0),torch.tensor(env.x0)),0),0)
r = 0
na = 0
while not done:
    # if random.randint(0,9)>=6:
    #     act = 1.0
    # else:
    #     act = 0.0
    act = th_based(current_state,th=0.1)
    next_state, reward, done, t = env.step(act)
    x = torch.cat((x,torch.unsqueeze(next_state,0)),0)
    r += reward.item()
    current_state = next_state
    na += act

e = (x[:,1] - 4*np.sin(2*np.pi/50*x[:,0]))**2
plt.plot(x[:,0],e)
print(torch.abs(e).mean())
print(r)
