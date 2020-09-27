# -*- coding: utf-8 -*-
"""
@author:yym
"""
import sys
import os

path_network = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path_network)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from network.NetworkClasses_possion import prefrontal_model
from num_in_out import get_data, get_data_prefrontal
from coding.coding_and_decoding import poisson_spike, poisson_spike_multi


def train_prefrontal(load_model=False,save_model=True,learning_rate=1e-4,iters=20000,gpu='0',possion_num=50,n_step=5):
    
    device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    best_loss = 5
    dt = 0.1
    lenth = 2000
    N_step = n_step
    x_sum = 0
    speed_limiter = 100

    print("load_model",load_model)

    if load_model != 1:
        network = prefrontal_model(batch_size=1,num_hidden1=256,num_hidden2=256,num_hidden3=256,N_step=N_step,gpu=gpu)

    else:
        print( "loading model from " + "network")
        network = torch.load("prefrontal/network.pkl")
        print("load model successfully")
    
    network = network.to(device)

    writer = SummaryWriter(comment='prefrontal_runs')

    #Fix partial model parameters
    # network.fc11.weight.requires_grad = False
    # network.fc11.bias.requires_grad = False
    # network.fc12.weight.requires_grad = False
    # network.fc12.bias.requires_grad = False
    # network.fc21.weight.requires_grad = False
    # network.fc21.bias.requires_grad = False
    # network.fc22.weight.requires_grad = False
    # network.fc22.bias.requires_grad = False

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate)
    loss_store = []

    control = 0

    for i in range(iters):

        if i % 10  == 0 and i != 0:
            print(i)
            if save_model == True:
                torch.save(network,"prefrontal/network_half.pkl")
                print("saved model successfully")
        if i == 200:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate/10)
        #get signal\
        m_z = 0
        m_dot_z = 10 
        belta = 0.5
        dt = 0.1
        x_sum = 0
        delt_z = 0

        flag_for_train = 0

        while (x_sum<lenth+100 and flag_for_train<40):

            f_x = np.random.rand(1)*400 - 200 #[-200,200]
            f_y = np.random.rand(1)*400 - 200 #[-200,200]

            delt_z = np.array(delt_z)
            opts = np.array([math.sqrt(delt_z)/4+np.random.randn()*0.1,math.sqrt(delt_z)/4+np.random.randn()*0.1,math.sqrt(delt_z),math.sqrt(delt_z)])
            opts = np.array([opts*(i+1) for i in range(N_step)])
            opts_ux = opts_uy = np.zeros_like(opts)
            m_x = np.random.rand(1)*8 - 4 # [-4,4]
            m_y = np.random.rand(1)*8 - 4 # [-4,4]
            for k in range(N_step):
                opts_mx,opts_ux[k,0] = get_data(m_x,opts[k,0],opts[k,2],f_x) #opts_mx:[-4,4] opts_ux:[-180,180]
                opts_my,opts_uy[k,0] = get_data(m_y,opts[k,1],opts[k,3],f_y) #opts_mx:[-4,4] opts_ux:[-180,180]
                m_x = opts_mx
                m_y = opts_my

            u_x = opts_ux[:,0]
            u_y = opts_uy[:,1]
            sigma_x = opts[:,2]
            sigma_y = opts[:,3]

            m_z = delt_z

            flag, m_dot_zt_true, belta_next,l_pt_mean, _ = get_data_prefrontal(f_x,f_y,u_x,u_y,sigma_x,sigma_y,m_dot_z,belta,lenth,x_sum)

            if np.isnan(l_pt_mean) :
                l_pt_mean = 1

            f_xt = torch.from_numpy(poisson_spike(t=possion_num*dt,f=(f_x+200)/4/2,dim=4))
            f_yt = torch.from_numpy(poisson_spike(t=possion_num*dt,f=(f_y+200)/4/2,dim=4))
            
            u_xt = torch.from_numpy(poisson_spike_multi(t=possion_num*dt,f=(u_x+200)/4,dim=4))
            u_yt = torch.from_numpy(poisson_spike_multi(t=possion_num*dt,f=(u_y+200)/4,dim=4))
            sigma_xt = torch.from_numpy(poisson_spike_multi(t=possion_num*dt,f=2*sigma_x/4,dim=4))
            sigma_yt = torch.from_numpy(poisson_spike_multi(t=possion_num*dt,f=2*sigma_y/4,dim=4))

            #x_sum_half = abs(lenth/2-abs(x_sum/2-lenth/2))
            x_sum_half = x_sum
            
            lenth_coding = torch.from_numpy(poisson_spike(t=possion_num*dt,f=lenth/20/2,dim=4))
            x_sum_coding = torch.from_numpy(poisson_spike(t=possion_num*dt,f=x_sum_half/10/2,dim=4))
            m_z_coding = torch.from_numpy(poisson_spike(t=possion_num*dt,f=m_z/4/2,dim=4))
            
            beltat = torch.from_numpy(poisson_spike(t=possion_num*dt,f=belta*400/4/2,dim=4))
            speed_limiter_coding = torch.from_numpy(poisson_spike(t=possion_num*dt,f=speed_limiter/2,dim=4))
            input11 = f_xt
            input12 = f_yt
            input2 = torch.cat([u_xt,u_yt,sigma_xt,sigma_yt],0) * 0.5
            input3 = torch.cat([lenth_coding,x_sum_coding,m_z_coding],0)
            input4 = torch.cat([beltat,m_z_coding,speed_limiter_coding],0)

            #target signal
            opts1 = torch.from_numpy(poisson_spike(t=possion_num*dt,f=(l_pt_mean+1)*200/4/2,dim=4))

            opts2 = torch.from_numpy(poisson_spike(t=possion_num*dt,f=flag*400/4/2,dim=4))

            if x_sum >= lenth:
                m_dot_zt_true = 0
            elif x_sum > 0.9 * lenth and x_sum<lenth:
                m_dot_zt_true = (lenth-x_sum)/2
            elif x_sum < 0.1 * lenth:
                if x_sum == 0:
                    m_dot_zt_true = 0

                if m_dot_zt_true < 85:
                    m_dot_zt_true += 15
                else:
                    m_dot_zt_true = 99

            m_dot_zt_true_num = torch.from_numpy(poisson_spike(t=possion_num*dt,f=m_dot_zt_true/4/2,dim=4))
            belta_nextt = torch.from_numpy(poisson_spike(t=possion_num*dt,f=belta_next*400/4/2,dim=4))
            opts3 = torch.cat([m_dot_zt_true_num,belta_nextt],0).float()

            if torch.cuda.is_available():
                input11 = input11.cuda().float()
                input12 = input12.cuda().float()
                input2 = input2.view(-1,50).cuda().float()
                input3 = input3.cuda().float()
                input4 = input4.cuda().float()
                opts1 = opts1.cuda().float()
                opts2 = opts2.cuda().float()
                opts3 = opts3.cuda().float()


            sum1, sum2, sum3 = network.forward(input11=input11,input12=input12,input2=input2,input3=input3,input4=input4,time_window=possion_num)


            #loss1 = criterion(torch.sum(sum1), torch.sum(opts1))
            #loss2 = criterion(torch.sum(sum2), torch.sum(opts2))

            if x_sum > lenth:
                punishment = 1
            else:
                punishment = flag_for_train
            

            output3 = torch.sum(sum3[0][0:4])
            output3_belta = torch.sum(sum3[0][4:8])
            loss3 = criterion(output3, torch.tensor(m_dot_zt_true).float().to(device)) + criterion(output3_belta, torch.tensor(belta_next*400).float().to(device))/100
            print("********************************************")
            print("output3",output3.cpu().item(),"m_dot_zt_true",m_dot_zt_true," distance",x_sum)
            


            #loss = loss1 + loss2 + loss3
            
            loss = loss3
            if torch.cuda.is_available():

                print("loss",loss.cpu().item())
            else:
                
                print("loss",loss.item())

            with writer:
                writer.add_scalar('loss_pre', loss, control)
                writer.add_scalars('delt_z',{'output3': output3,'m_dot_zt_true': torch.tensor(m_dot_zt_true)}, control)
                writer.add_scalar('x_sum', x_sum, control)
            control += 1

            

            if i > 100:

                if x_sum > 0.3*lenth and x_sum < 0.6*lenth:
                    pass
                else:
                    #Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                x_sum = x_sum + output3.cpu().item() * 10 * dt
                m_dot_z = output3.cpu().item()
                delt_z = output3.cpu().item()
            else:

                #Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                x_sum = x_sum + m_dot_zt_true * 10  * dt
                m_dot_z = m_dot_zt_true
                delt_z = m_dot_zt_true
            belta = belta_next
            flag_for_train += 1


