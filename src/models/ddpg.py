import argparse
import os
import random
import sys
import time

import jsbsim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# torch.set_num_threads(8)

sys.path.append(str(jsbsim.get_default_root_dir()) + '/DBRL/')

# from src.reward import reward
from src.environments import jsbsimEnv as Env


def parse_args():
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--cuda', default='0', metavar='int', help='specifies the GPU to be used')
    parser.add_argument('--playSpeed', default=0, metavar='double', help='specifies to run in real world time')
    parser.add_argument('--modelPath', default='/data/wnn_data/bestModel/', metavar='str', help='specifies the pre-trained model')
    args = parser.parse_args()
    return args


class Actor(nn.Module):

    def __init__(
        self,
        status_dim=12,
        hidden_dim_1=64,
        hidden_dim_2=64,
        hidden_dim_3=64,
        num_of_actions=4,  # temporary
        dropout=.5,
    ):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(status_dim, hidden_dim_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.hidden2action = nn.Linear(hidden_dim_3, num_of_actions)
        self.tanh = nn.Tanh()
        
    def forward(self, status):

        out = self.fc1(status)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.hidden2action(out)
        action = self.tanh(out)

        return action


class Critic(nn.Module):

    def __init__(
        self,
        status_dim=12,
        num_of_actions=4,  # temporary
        hidden_dim_1=64,
        hidden_dim_2=64,
        hidden_dim_3=64,
        dropout=.5,
    ):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(status_dim + num_of_actions, hidden_dim_1)
        self.fc11 = nn.Linear(status_dim, hidden_dim_1)
        self.fc12 = nn.Linear(num_of_actions, hidden_dim_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)

        self.hidden2q = nn.Linear(hidden_dim_3, 1)
        
    def forward(self, status, action):

        out = self.fc11(status) + self.fc12(action)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        q = self.hidden2q(out)

        return q


class DDPG():

    def __init__(
        self,
        cuda=0,
        modelPath='./bestModel',
    ) -> None:
    
        device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        self.model_actor = Actor().to(device)
        self.target_model_actor = Actor().to(device)
        self.model_critic = Critic().to(device)
        self.target_model_critic = Critic().to(device)

        self.modelPath = modelPath
        if not os.path.exists(self.modelPath):
            os.mkdir(self.modelPath)
        if os.listdir(self.modelPath) != []:
            try:
                self.model_actor.load_state_dict(torch.load(self.modelPath + 'Actor.pt'))
                self.model_critic.load_state_dict(torch.load(self.modelPath + 'Critic.pt'))
                self.target_model_actor.load_state_dict(torch.load(self.modelPath + 'Actor_target.pt'))
                self.target_model_critic.load_state_dict(torch.load(self.modelPath + 'Critic_target.pt'))
            except:
                print("Model Loading Error!")
                time.sleep(1)


    def _critic_learn(
        self, 
        status, 
        action, 
        reward, 
        next_status, 
        terminate,
    ):
        self.model_critic.train()
        self.gamma = .9
        self.weight_decay = 1e-2
        self.optimizer = optim.SGD(self.model_critic.parameters(), lr = 1e-2, momentum=0.9, weight_decay=self.weight_decay)

        next_action = self.target_model_actor(next_status)
        next_q = self.target_model_critic(next_status, next_action)

        if terminate:
            target_q = reward
        else:
            target_q = reward + self.gamma * next_q

        # target_q = reward if terminate else reward + self.gamma * next_q
        target_q = target_q.detach()
        
        q = self.model_critic(status, action.detach())
        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target_q)
        # print("/*/*{}".format(loss))
        self.model_critic.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss

    def _actor_learn(
        self,
        status,
    ):
        self.model_actor.train()
        self.weight_decay = 1e-2
        self.optimizer = optim.SGD(self.model_actor.parameters(), lr = 1e-2, momentum=0.9, weight_decay=self.weight_decay)
        
        action = self.model_actor(status)
        # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        # action = torch.rand([4]).to(device)
        q = self.model_critic(status, action)
        # for p in self.model_critic.parameters():
        #     p.requires_grad = False
        loss = -1.0 * q
        # loss.requires_grad_(True)
        self.model_actor.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss

    def sync_target(
        self,
        decay=.9
    ):

        for target_param, source_param in zip(self.target_model_actor.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1-decay) * target_param.data + decay * source_param.data)
        for target_param, source_param in zip(self.target_model_critic.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1-decay) * target_param.data + decay * source_param.data)


    def getStatus(
        self,
        env,
        id
    ):  
        return torch.rand([12])
        status = []
        # status.append(env.getFdm(id).getProperty("propulsion/total-fuel-lbs") / 3000)  # Ego_fuel
        # status.append(env.getFdm(id).getHP())  # Ego_hp
        # status.append(env.getFdm(id^3).getHP())  # Oppo_hp

        status = status + env.getFdm(id).getProperty("attitudeRad")  # Euler angle
        status = status + env.getFdm(id).getProperty("position")[0:2]  # Location
        status.append((env.getFdm(id).getProperty("position")[2] - 30000) / 1000)  # Height
        # status.append(env.getFdm(id).getProperty("velocity")[0] / 500)  # Velocity
        # status.append(env.getFdm(id).getProperty("velocity")[1] / 500)  # Velocity
        # status.append(env.getFdm(id).getProperty("velocity")[2] / 500)  # Velocity

        status = status + env.getFdm(id^3).getProperty("attitudeRad")
        status = status + env.getFdm(id^3).getProperty("position")[0:2]
        status.append((env.getFdm(id^3).getProperty("position")[2] - 30000) / 1000)
        # status.append(env.getFdm(id^3).getProperty("velocity")[0] / 500)
        # status.append(env.getFdm(id^3).getProperty("velocity")[1] / 500)
        # status.append(env.getFdm(id^3).getProperty("velocity")[2] / 500)

        return torch.Tensor(status)


    def getReward(
        self,
        env,
        id
    ):
        return random.random()
        # r_rp = reward.R_rp(env, id)
        # r_closure = reward.R_closure(env, id)
        # r_gunsnap = reward.R_gunsnap(env, id)
        # r_deck = reward.R_deck(env, id)
        # r_too_close = reward.R_too_close(env, id)
        # r_win = reward.R_win(env, id)
        # print("*Reward: {}\t{}\t{}\t{}\t{}\t{}".format(r_rp, r_closure, r_gunsnap, r_deck, r_too_close, r_win))
        # return r_rp + r_closure + r_gunsnap + r_deck + r_too_close + r_win


    def episode(
        self,
        device,
        fgfs_1,
        fgfs_2,
        playSpeed,
    ):
        env = Env(
            fgfs_1,
            fgfs_2
        )
        
        pre_status_1 = torch.zeros([12])
        pre_action_1 = torch.zeros([4])
        pre_reward_1 = 0
        pre_status_2 = torch.zeros([12])
        pre_action_2 = torch.zeros([4])
        pre_reward_2 = 0
        pre_terminate = 0

        while True:
            terminate = env.step(playSpeed=playSpeed)
            if terminate != 0:
                break
            
            if env.getNof() % 12 == 0:
                status_1 = self.getStatus(env, 1)
                status_2 = self.getStatus(env, 2)
                action_1 = self.model_actor(self.getStatus(env, 1).to(device))
                action_2 = self.model_actor(self.getStatus(env, 2).to(device))
                reward_1 = self.getReward(env, 1)  # 当前状态下的状态价值函数，可以理解为上一状态的动作价值函数
                reward_2 = self.getReward(env, 2)

                # action_1 = action_1 + torch.rand([4]).to(device) - 0.5

            
                env.getFdm(1).sendAction(action_1.unsqueeze(0))
                # env.getFdm(2).sendAction(action_2.unsqueeze(0))
                # print("\n\n!action:\t{}\n!status:\t{}\n\n".format(action_1, status_1))
                env.getFdm(2).sendAction([[0, -.07, 0, 0]])

                pre_status_1 = pre_status_1.to(device)
                pre_status_2 = pre_status_2.to(device)
                pre_action_1 = pre_action_1.to(device)
                pre_action_2 = pre_action_2.to(device)
                status_1 = status_1.to(device)
                status_2 = status_2.to(device)

                self._actor_learn(pre_status_1)

                self._critic_learn(pre_status_1, pre_action_1, reward_1, status_1, pre_terminate)

                # self._actor_learn(pre_status_2)

                # self._critic_learn(pre_status_2, pre_action_2, reward_1, status_2, pre_terminate)
                
                pre_status_1 = status_1
                pre_status_2 = status_2
                pre_action_1 = action_1
                pre_action_2 = action_2
                pre_reward_1 = reward_1
                pre_reward_2 = reward_2
                pre_terminate = env.terminate()

        self.sync_target()

    def train(
        self,
        epochs=20000,
        cuda='0',
        fgfs_1=False,
        fgfs_2=False,
        playSpeed=0,
    ):
        device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        for _ in tqdm(range(epochs)):
            self.episode(device, fgfs_1, fgfs_2, playSpeed)
            torch.save(self.model_actor.state_dict(), self.modelPath + 'Actor.pt')
            torch.save(self.model_critic.state_dict(), self.modelPath + 'Critic.pt')
            torch.save(self.target_model_actor.state_dict(), self.modelPath + 'Actor_target.pt')
            torch.save(self.target_model_critic.state_dict(), self.modelPath + 'Critic_target.pt')



if __name__ == "__main__":
    args = parse_args()

    ddpg = DDPG(
        cuda=args.cuda,
        modelPath=args.modelPath,
    )

    ddpg.train(
        cuda=args.cuda,
        fgfs_1=args.fgfs_1,
        fgfs_2=args.fgfs_2,
        playSpeed=float(args.playSpeed),
    )





























