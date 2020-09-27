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
from network.NetworkClasses_possion import SNN, LSMNetwork, cere_model, prefrontal_model
from coding.coding_and_decoding import poisson_spike, poisson_spike_multi

from num_in_out import get_data, get_data_prefrontal

class actor_net(nn.Module):
    def __init__(
        self,
        N_step=2,
        load_model=True,
        save_model=True,
        learning_rate=1e-4,
        iters=20000,gpu='0',
        possion_num=50,
        speed_limiter=100,
        lenth=2 * 1000):
        super(actor_net,self).__init__()
        self.N_step = N_step  # predict step
        self.load_model = load_model
        dims = (7,7,7)
        self.bias = 4
        self.n_in = dims[0]*dims[1]*dims[2]
        w_mat = 4*20*np.array([[3, 6],[-2, -2]])
        self.steps = 50
        self.ch = 5 #input num
        self.possion_num_snu = possion_num
        self.possion_num = possion_num
        self.speed_limiter = speed_limiter

        self.dt = 0.1
        self.lenth = lenth
        self.x_sum = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")

        writer = SummaryWriter(comment='Actor_Net')
        if load_model != 1:
            sys.stderr.write("create new model\n")
            self.reservoir_network = LSMNetwork(dims, 0.2, w_mat, 7, self.steps, self.ch, t_ref=0, ignore_frac=0)
            self.snu = SNN(batch_size=1,input_size=self.n_in,hidden_size=256,num_classes=self.bias,possion_num=self.possion_num,gpu=gpu)

            self.network_x = cere_model(batch_size=1,num_in_MF=1,num_out_MF=4,num_out_GC=2000,num_out_PC=200,num_out_DCN=1,possion_num=self.possion_num,gpu=gpu)
            self.network_y = cere_model(batch_size=1,num_in_MF=1,num_out_MF=4,num_out_GC=2000,num_out_PC=200,num_out_DCN=1,possion_num=self.possion_num,gpu=gpu)

            self.network = prefrontal_model(batch_size=1,num_hidden1=256,num_hidden2=256,num_hidden3=256,N_step=self.N_step,gpu=gpu)

        else:
            sys.stderr.write( "loading model from " + "my_snu_model\n")
            self.reservoir_network = torch.load("LSM_SNU/my_lsm_model_rl.pkl",encoding='unicode_escape')
            self.snu = torch.load("LSM_SNU/my_snu_model_rl.pkl")
            sys.stderr.write("load lsm_snu model successfully\n")

            sys.stderr.write( "loading model from " + "network_x and network_y\n")
            self.network_x = torch.load("cere/network_x_rl.pkl")
            self.network_y = torch.load("cere/network_x_rl.pkl")
            sys.stderr.write("load model successfully\n")

            sys.stderr.write( "loading model from " + "network\n")
            self.network = torch.load("prefrontal/network_rl.pkl")
            sys.stderr.write("load model successfully\n")

        self.snu = self.snu.float().to(self.device)
        self.network_x = self.network_x.float().to(self.device)
        self.network_y = self.network_y.float().to(self.device)
        self.network = self.network.float().to(self.device)

        #Fix partial model parameters
        self.snu.requires_grad = False

        #self.network_x.requires_grad = False
        #self.network_y.requires_grad = False

        # self.network.fc11.weight.requires_grad = False
        # self.network.fc11.bias.requires_grad = False
        # self.network.fc12.weight.requires_grad = False
        # self.network.fc12.bias.requires_grad = False
        # self.network.fc21.weight.requires_grad = False
        # self.network.fc21.bias.requires_grad = False
        # self.network.fc22.weight.requires_grad = False
        # self.network.fc22.bias.requires_grad = False

        #init matrix
        self.train_in_spikes = np.zeros((self.ch, self.steps))
        self.snu_output_np = np.zeros([self.N_step,self.bias])

        self.possion_rate_coding = np.zeros([self.n_in,self.possion_num_snu])

        self.m_x = torch.tensor([0]).float().to(self.device)
        self.m_y = torch.tensor([0]).float().to(self.device)
        self.m_dot_z_true = torch.tensor([0]).float().to(self.device)
        self.belta_true = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=0.5*400/4/2,dim=4)).float().to(self.device)
        self.opts_mx = 0
        self.opts_my = 0
        self.opts_mz = 0
        
    def forward(self,f_x,f_y,x_z,m_x,m_y,m_z):
        '''
        f_x: the force in x axis
        f_y: the force in y axis
        x_z: The distance that the part has traveled
        m_z: Current Z-direction speed of the part
        '''
        
        f_x_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_x+200)/4,dim=4)).float().to(self.device) #[0,100]
        f_y_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_y+200)/4,dim=4)).float().to(self.device) #[0,100]
        m_x_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_x+4)/4,dim=4)).float().to(self.device) #[0,100]
        m_y_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_y+4)/4,dim=4)).float().to(self.device) #[0,100]

        cere_out = torch.zeros([self.N_step,self.bias*4,self.possion_num])

        #init parms for SNU_LSM
        ipts = np.zeros(5)
        ipts[0] = torch.clamp(m_z,0,100)

        with torch.no_grad():
            for iteration in range(self.N_step):
                for j in range(self.ch):
                    spike_interval = np.floor((self.steps+0.1)/(ipts[j]/2  + 0.0001)).astype(np.int32)
                    jitter = np.random.randint(20)
                    spikes = np.zeros(self.steps-20)
                    spikes[jitter::spike_interval] = 60
                    self.train_in_spikes[j,:-20] = spikes

                #LSM mode
                self.reservoir_network.add_input(self.train_in_spikes)
                rate_coding = self.reservoir_network.simulate()
                
                #SNU mode
                self.snu_output = self.snu.forward(input=rate_coding,task="LSM",time_window=self.possion_num)

                #build next step data
                for m in range(4):
                    self.snu_output_np[0][m] = torch.sum(self.snu_output[0][4*m:4*m+4]).cpu().item()
                ipts,_,__ = np.split(ipts,[1,1],axis = 0) # get delt z
                ipts = np.hstack((ipts,self.snu_output_np[0])) #t+1 input

                #cere mode
                u_x_coding = self.snu.monitor_fc2[:,0:4,:].view(4,self.possion_num)
                u_y_coding = self.snu.monitor_fc2[:,4:8,:].view(4,self.possion_num)
                sigma_x_coding = self.snu.monitor_fc2[:,8:12,:].view(4,self.possion_num)
                sigma_y_coding = self.snu.monitor_fc2[:,12:16,:].view(4,self.possion_num)

                sum_x = self.network_x.forward(m_x_coding,u_x_coding,sigma_x_coding,f_x_coding,time_window=self.possion_num)
                sum_y = self.network_y.forward(m_y_coding,u_y_coding,sigma_y_coding,f_y_coding,time_window=self.possion_num)

                sum_x = sum_x.view(8,-1)
                sum_mx = sum_x[0:4]
                sum_ux = sum_x[4:8]
                sum_y = sum_y.view(8,-1)
                sum_my = sum_y[0:4]
                sum_uy = sum_y[4:8]

                cere_out[iteration][0:4,:] = self.network_x.monitor_DCN_m[:,4:8,:]
                cere_out[iteration][4:8,:] = self.network_y.monitor_DCN_m[:,4:8,:]
                cere_out[iteration][8:12,:] = sigma_x_coding
                cere_out[iteration][12:16,:] = sigma_y_coding

                m_x_coding = self.network_x.monitor_DCN_m[:,0:4,:].view(4,self.possion_num)
                m_y_coding = self.network_y.monitor_DCN_m[:,0:4,:].view(4,self.possion_num)

        cere_out = cere_out.to(self.device)
        #pre mode
        u_xt = cere_out[:,0:4,:].clone()
        u_xt = u_xt.view(-1,self.possion_num)
        u_yt = cere_out[:,4:8,:].clone()
        u_yt = u_yt.view(-1,self.possion_num)
        sigma_x = cere_out[:,8:12,:].clone()
        sigma_x = sigma_x.view(-1,self.possion_num)
        sigma_y = cere_out[:,12:16,:].clone()
        sigma_y = sigma_y.view(-1,self.possion_num)

        lenth_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=self.lenth/20/2,dim=4)).float().to(self.device) #[0，50]
        x_sum_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=x_z/10/2,dim=4)).float().to(self.device) #[0,50]
        m_z_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=m_z/4/2,dim=4)).float().to(self.device) #[0,25]
        
        beltat = self.belta_true
        speed_limiter_coding = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=self.speed_limiter/2,dim=4)).float().to(self.device) #[0,25]
        
        input11 = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_x+200)/4/2,dim=4)).float().to(self.device)
        input12 = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=(f_y+200)/4/2,dim=4)).float().to(self.device)
        input2 = torch.cat([u_xt,u_yt,sigma_x,sigma_y],0).float().to(self.device) * 0.5
        input3 = torch.cat([lenth_coding,x_sum_coding,m_z_coding],0).float().to(self.device)
        input4 = torch.cat([beltat,m_z_coding,speed_limiter_coding],0).float().to(self.device)


        sum1, sum2, sum3 = self.network.forward(input11=input11,input12=input12,input2=input2,input3=input3,input4=input4,time_window=self.possion_num)

        self.m_dot_z_true = torch.sum(sum3[0][0:4])
        self.belta_true = self.network.monitor_h32[:,4:8,:].view(4,self.possion_num)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=1e-4)

        ###########target signal#############

        sigma_x_pre = torch.sum(cere_out[0,8:12,:]).cpu().item()/2
        u_x_pre = torch.sum(cere_out[0,0:4,:]).cpu().item() - 100
        # self.opts_mx,opts_ux = get_data(m_x,u_x_pre,sigma_x_pre,f_x)
        self.opts_mx = f_x/100

        sigma_y_pre = torch.sum(cere_out[0,12:16,:]).cpu().item()/2
        u_y_pre = torch.sum(cere_out[0,4:8,:]).cpu().item() - 100
        # self.opts_my,opts_uy = get_data(m_y,u_y_pre,sigma_y_pre,f_y)
        self.opts_my = f_y/100

        if x_z >= self.lenth:
            self.opts_m_z = 0
        elif x_z > 0.9 * self.lenth and x_z<self.lenth:
            self.opts_m_z = (self.lenth-x_z)/2
        elif x_z < 0.1 * self.lenth:
            if x_z == 0:
                self.opts_m_z = 0

            if self.opts_m_z < 85:
                self.opts_m_z += 15
        else:
            self.opts_m_z = 99

        ######################################

        

        #cere
        u_x_coding_predict = cere_out[0,0:4,:]
        u_x_coding_predict = u_x_coding_predict.view(-1,self.possion_num)
        u_y_coding_predict = cere_out[0,4:8,:]
        u_y_coding_predict = u_y_coding_predict.view(-1,self.possion_num)
        sigma_x_coding_predict = cere_out[0,8:12,:]
        sigma_x_coding_predict = sigma_x_coding_predict.view(-1,self.possion_num)
        sigma_y_coding_predict = cere_out[0,12:16,:]
        sigma_y_coding_predict = sigma_y_coding_predict.view(-1,self.possion_num)

        m_x_coding_predict = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_x+4)/4,dim=4)).float().to(self.device).float().to(self.device)
        m_y_coding_predict = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=50*(m_y+4)/4,dim=4)).float().to(self.device).float().to(self.device)

        sum_x_predict = self.network_x.forward(m_x_coding_predict,u_x_coding_predict,sigma_x_coding_predict,f_x_coding,time_window=self.possion_num)
        sum_y_predict = self.network_y.forward(m_y_coding_predict,u_y_coding_predict,sigma_y_coding_predict,f_y_coding,time_window=self.possion_num)
   
        sum_x_predict = sum_x_predict.view(8,-1)
        sum_mx = sum_x_predict[0:4]
        sum_ux = sum_x_predict[4:8]
        sum_y_predict = sum_y_predict.view(8,-1)
        sum_my = sum_y_predict[0:4]
        sum_uy = sum_y_predict[4:8]

        self.m_x = torch.sum(sum_mx)/100-2 +1
        self.m_y = torch.sum(sum_my)/100-2 +1

        final_output = torch.zeros([1,3]).to(self.device)
        

        # the reverse is right?
        final_output[0][0] = -self.m_x
        final_output[0][1] = -self.m_y
        final_output[0][2] = self.m_dot_z_true

        #return final_output, out_ux_true, out_uy_true,self.snu_output_np[:,2], self.snu_output_np[:,3]
        return final_output[0]
    
    def save(self):
        torch.save(self.snu,"LSM_SNU/m_snu_model_rl.pkl")
        torch.save(self.network_x,"cere/network_x_rl.pkl")
        torch.save(self.network_y,"cere/network_y_rl.pkl")
        torch.save(self.network,"prefrontal/network_rl.pkl")
        sys.stderr.write("saved model successfully\n")
    
    def reset(self):
        #init matrix
        self.x_sum = 0
        self.train_in_spikes = np.zeros((self.ch, self.steps))
        self.snu_output_np = np.zeros([self.N_step,self.bias])

        self.possion_rate_coding = np.zeros([self.n_in,self.possion_num_snu])

        self.m_x = torch.tensor([0]).float().to(self.device)
        self.m_y = torch.tensor([0]).float().to(self.device)
        self.m_dot_z_true = torch.tensor([0]).float().to(self.device)
        self.belta_true = torch.from_numpy(poisson_spike(t=self.possion_num*self.dt,f=0.5*400/4/2,dim=4)).float().to(self.device)
        self.opts_mx = 0
        self.opts_my = 0
        self.opts_mz = 0