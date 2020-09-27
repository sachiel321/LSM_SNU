# -*- coding: utf-8 -*-
"""
@author:yym
"""
import torch
import numpy as np
import argparse
from cere.my_cerebellar_model import train_cerebellar
from LSM_SNU.LSM_SNU import train_LSM_SNU
from prefrontal.my_prefrontal_model import train_prefrontal
from coding.coding_and_decoding import seed_everything
from model.combine_rl import train_combine
from model.combine import actor_net

def get_parser():
    parser = argparse.ArgumentParser(description='Train network respectively.')
    parser.add_argument("--load_model","-l", default=True,action="store_true",
                        help="Run or not.")
    parser.add_argument("--save_model","-s", default=True,action="store_true",
                        help="Run or not.")
    parser.add_argument("--gpu","-g",type=str,default="0",help='Choose GPU number.')
    parser.add_argument("--learning_rate","-lr",type=float,default=1e-4)
    parser.add_argument("--N_step","-n",type=int,default=2,help='How many step do you want to predict.')
    parser.add_argument("--iters","-i",type=int,default=100,help='The training step.')
    parser.add_argument("--possion_num","-p",type=int,default=50)
    parser.add_argument("--seed","-seed",type=int,default=2)
    parser.add_argument("--mode","-m",type=str,default="LSM_SNU",help='You can input LSM_SNU, cerebellar, train_combine and prefrontal to train different part.')
    return parser

parser = get_parser()
args = parser.parse_args()
seed_everything(args.seed)

def train(
    mode=args.mode,
    load_model=args.load_model,
    save_model=args.save_model,
    gpu = args.gpu,
    learning_rate = args.learning_rate,
    N_step = args.N_step,
    iters = args.iters,
    possion_num = args.possion_num):

    print("load_model",load_model)

    if mode == 'LSM_SNU':
        print(N_step)
        print(gpu)
        train_LSM_SNU(N_step,load_model,save_model,learning_rate,iters,gpu,possion_num)
    elif mode == 'cerebellar':
        train_cerebellar(load_model,save_model,learning_rate,iters,gpu,possion_num)
    elif mode == 'prefrontal':
        train_prefrontal(load_model,save_model,learning_rate,iters,gpu,possion_num,N_step)
    elif mode == 'train_combine':

        train_combine(N_step=N_step,
                    load_model=load_model,
                    save_model=save_model,
                    learning_rate=learning_rate,
                    iters=iters,
                    gpu=gpu,
                    possion_num=possion_num,
                    speed_limiter=100,
                    lenth=2 * 1000)
    else:
        print('Training mode error! Only LSM_SNU, cerebellar and prefrontal are available')
    

if __name__ == "__main__":
    train()
