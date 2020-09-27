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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from network.NetworkClasses_possion import SNN, cere_model
from coding.coding_and_decoding import poisson_spike
from num_in_out import get_data


def train_cerebellar(load_model=False,save_model=True,learning_rate=1e-4,iters=20000,gpu='0',possion_num=50):

    best_loss = 5
    dt = 0.1

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")

    if load_model != 1:
        network_x = cere_model(batch_size=1,num_in_MF=1,num_out_MF=16,num_out_GC=1000,num_out_PC=400,num_out_DCN=1,possion_num=possion_num,gpu=gpu)

    else:
        print( "loading model from " + "network_x and network_y")
        network_x = torch.load("cere/network_x.pkl")
        print("load model successfully")

    network_x = network_x.to(device)

    criterion = torch.nn.MSELoss()
    optimizer_x = torch.optim.Adam(network_x.parameters(), lr=learning_rate)

    loss_store = []

    writer = SummaryWriter(comment='cerebellar_runs')
    #print(network_x.cpu().state_dict())

    for i in range(iters):

        if i % 50 == 0 and i != 0:
            print(i)
            if save_model == True:
                torch.save(network_x,"cere/network_x.pkl")
                print("saved model successfully")

        #get input signal
        m_x = np.random.rand(1)*4 - 2 # [-2,2]
        u_x = np.random.rand(1)*30-15 #[-15,15]
        sigma_x = np.random.rand(1)*200 #[0,200]
        f_x = np.random.rand(1)*400 - 200 #[-200,200]
        mx = ux = []

        opts_mx,opts_ux = get_data(m_x,u_x,sigma_x,f_x) #opts_mx:[-2,2] opts_ux:[-180,180]

        if np.isnan(opts_mx) or np.isnan(opts_ux):
            i = i-1
        else:
            m_x_coding = poisson_spike(t=possion_num*dt,f=100*(m_x+2)/4,dim=4)
            u_x_coding = poisson_spike(t=possion_num*dt,f=13*(u_x+15)/4,dim=4)
            sigma_x_coding = poisson_spike(t=possion_num*dt,f=2*sigma_x/4,dim=4)
            f_x_coding = poisson_spike(t=possion_num*dt,f=(f_x+200)/4,dim=4)
            opts_mx_coding = poisson_spike(t=possion_num*dt,f=50*(opts_mx+4)/4,dim=4)
            opts_ux_coding = poisson_spike(t=possion_num*dt,f=(opts_ux+200)/4,dim=4)

            sum_out = network_x.forward(m_x_coding,u_x_coding,sigma_x_coding,f_x_coding,time_window=possion_num)

            #loss
            sum_mx = sum_out[:,0:4]
            sum_ux = sum_out[:,4:8]
            opts_mx_torch = torch.sum(torch.from_numpy(opts_mx_coding).to(device).float(),dim=1)

            opts_ux_torch = torch.sum(torch.from_numpy(opts_ux_coding).to(device).float(),dim=1)

            opts_output = torch.cat([opts_mx_torch,opts_ux_torch],0)

            out_mx = torch.sum(sum_mx)/100-2
            out_ux = torch.sum(sum_ux)-100
            

            loss_x = criterion(sum_mx[0],opts_mx_torch) + criterion(sum_ux[0],opts_ux_torch)

            print("out_mx",torch.sum(sum_mx).cpu().item(),"opts_mx",torch.sum(opts_mx_torch).cpu().item())
            print("out_ux",out_ux.cpu().item(),"opts_ux_torch",opts_ux)
            print("loss",loss_x)

            # Backward and optimize
            optimizer_x.zero_grad()
            loss_x.backward()
            #loss_ux.backward()
            optimizer_x.step()

            with writer:
                writer.add_scalar('loss_cere', loss_x, i)
                writer.add_scalars('m_x',{'out_mx': torch.sum(sum_mx),'opts_mx': torch.sum(opts_mx_torch)}, i)
                writer.add_scalars('u_x',{'out_ux': out_ux,'opts_ux': torch.tensor(opts_ux)}, i)
            


            

