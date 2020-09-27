# -*- coding: utf-8 -*-
"""
@author:yym
"""
import sys
import os

path_network = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path_network)

import numpy as np
import torch
import torch.nn as nn
from model.combine import actor_net
from torch.utils.tensorboard import SummaryWriter
from simulation import simulate
from num_in_out import get_data, get_data_prefrontal

def train_combine(N_step=5,
    load_model=True,
    save_model=True,
    learning_rate=1e-4,
    iters=20000,
    gpu='0',
    possion_num=50,
    speed_limiter=100,
    lenth=2 * 1000):

    actor = actor_net(N_step,load_model,save_model,learning_rate,iters,gpu,possion_num,speed_limiter,lenth)

    print("777mode")
    ############## Hyperparameters ##############
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 200      # update policy every n timesteps
    lr = learning_rate
    
    writer = SummaryWriter(comment='Train_combine')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    #############################################
    
    '''
    creating environment:
    state: (f_x,f_y,x_z,m_z)
    action: (m_x,m_y,m_z)
    '''

    env = simulate(0,0,0,0,0,lenth)
    state_dim = 4
    action_dim = 3
        
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        action = torch.zeros(action_dim).float()
        reward_max = 0
        reward_min = 0
        actor.opts_m_z = 0
        for t in range(40):
            time_step +=1

            # with torch.no_grad():
            #     action = actor.forward(state[0],state[1],state[2],action[0],action[1],action[2])
            action = actor.forward(state[0],state[1],state[2],action[0],action[1],action[2])

            state,done = env.step(action.cpu().detach().numpy().astype(np.float32).reshape(-1))

            loss_x = criterion(action[0],torch.tensor(actor.opts_mx).float().cuda())
            loss_y = criterion(action[1],torch.tensor(actor.opts_my).float().cuda())
            loss = loss_x + loss_y + criterion(action[2],torch.tensor(actor.opts_m_z).float().cuda())
            
            if i_episode < 50 and state[2]>0.3 * lenth and state[2]<0.6 * lenth:
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(action.cpu(),'||opts:',actor.opts_mx,'||',actor.opts_my,'||',actor.opts_m_z)
            print(state)
            print("*************ep:",i_episode,"****************")

            with writer:
                writer.add_scalar('Train/x_z', np.array(state[2]), time_step)
                writer.add_scalar('loss', loss, time_step)
                writer.add_scalars('m_x',{'out_mx': action[0],'opts_mx': torch.tensor(actor.opts_mx).float()}, time_step)
                writer.add_scalars('m_y',{'out_my': action[1],'opts_my': torch.tensor(actor.opts_my).float()}, time_step)
                writer.add_scalars('m_z',{'out_mz': action[2],'opts_mz': torch.tensor(actor.opts_m_z).float()}, time_step)
                    
            # # update if its time
            # if time_step % update_timestep == 0 and time_step!=0:

            #     #update here

            #     time_step = 0

        state_reset = np.zeros_like(state)
        action_reset = torch.zeros_like(action)
        actor.forward(state_reset[0],state_reset[1],state_reset[2],action_reset[0],action_reset[1],action_reset[2])

        actor.reset()

        

        avg_length += t
        print('time_step',time_step)
        
        # save every 100 episodes
        if i_episode % 10 == 0 and i_episode != 0:
            actor.save()



