'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import display
# from reacher import Reacher

import argparse
import time


import gym_Vibration


import logging
# logger = logging.getLogger('vibration-example')
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', filename='example.log', level=logging.DEBUG)    
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')
# print('$$$$$$')
# input()

# logging.basicConfig(filename='example.log', level=logging.INFO)    
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import nni
# GPU = True
# device_idx = 0
# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)
    # use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    # torch.manual_seed(args['seed'])

    # device = torch.device("cuda" if use_cuda else "cpu")


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)  # True  False
parser.add_argument('--go_on_train', dest='go_on_train', action='store_true', default=False)  # True  False
parser.add_argument("--env_name", default="VibrationEnv-v0")  # OpenAI gym environment name  VibrationEnv  Pendulum

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=2,
                    help='# of levels (default: 2)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='sequence length (default: 400)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='initial learning rate (default: 3e-4)')
# parser.add_argument('--optim', type=str, default='Adam',
#                     help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer (default: 256)')



# args = parser.parse_args()
args, _ = parser.parse_known_args()


# use_cuda = not args['no_cuda'] and torch.cuda.is_available()

# device = torch.device("cuda" if use_cuda else "cpu")




class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def action(self, a):
        l = self.action_space.low
        h = self.action_space.high

        a = l + (a + 1.0) * 0.5 * (h - l)
        a = np.clip(a, l, h)

        return a

    def reverse_action(self, a):
        l = self.action_space.low
        h = self.action_space.high

        a = 2 * (a -l)/(h - l) -1 
        a = np.clip(a, l, h)

        return a

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        

        self.model = nn.Sequential(
                    nn.Linear(num_inputs, hidden_size),           # 输入10维，隐层20维
                    # nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=True),     # BN层，参数为隐层的个数
                    nn.LayerNorm(hidden_size, elementwise_affine=True),
                    nn.ReLU(),                     # 激活函数

                    nn.Linear(hidden_size, hidden_size),           # 输入10维，隐层20维
                    # nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=True),     # BN层，参数为隐层的个数
                    nn.LayerNorm(hidden_size, elementwise_affine=True),
                    nn.ReLU(),   

                    # nn.Linear(num_inputs + num_actions, hidden_size),           # 输入10维，隐层20维
                    # # nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=True),     # BN层，参数为隐层的个数
                    # # nn.LayerNorm(hidden_size, elementwise_affine=True),
                    # nn.ReLU(),   

                    # nn.Linear(hidden_size, hidden_size),           # 输入10维，隐层20维
                    # # nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=True),     # BN层，参数为隐层的个数
                    # nn.LayerNorm(hidden_size, elementwise_affine=True),
                    # nn.ReLU()            
                )

        self.linear3 = nn.Linear(hidden_size + num_actions, hidden_size)

        self.linear4 = nn.Linear(hidden_size, 1)


        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # x = torch.cat([state, action], 1) # the dim 0 is number of samples
        # x = self.linear1(x)
        # # x = self.bn1(x)
        # x = F.relu(x)

        # x = self.linear2(x)
        # # x = self.bn2(x)
        # x = F.relu(x)

        # x = self.linear3(x)
        # # x = self.bn3(x)
        # x = F.relu(x)

        # x = self.linear4(x)

        xs = self.model(state)
        x = torch.cat([xs, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear3(x))

        return self.linear4(x)
        

ENV = ['Pendulum', 'Reacher'][0]

# end_id = "Pendulum-v0"
env = NormalizedActions(gym.make(args.env_name))  # VibrationEnv  Pendulum
action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

from model import TCN
from TCN.tcn import TemporalConvNet
input_channels = state_dim
output_channels = action_dim
# num_channels = [30, 30, 30, 30, 30, 30, 30, 30]
num_channels = [256, 256]
kernel_size = 7
state_batch = 1
state_seq_len = 1

dropout = 0

class PolicyNetwork(nn.Module):
    def __init__(self, args, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()


         # # Example of using Sequential  
        # model = nn.Sequential(  
        #         nn.Conv2d(1,20,5),  
        #         nn.ReLU(),  
        #         nn.Conv2d(20,64,5),  
        #         nn.ReLU()  
        #         )  
       
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.num_inputs = num_inputs  
        # self.output_dim = num_actions 
        # self.hidden_dim = hidden_size  
      
        # self.bn1 = nn.BatchNorm1d(state_dim)
        self.tcn = TemporalConvNet(input_channels, [args['nhid']]*args['levels'], kernel_size=kernel_size, dropout=args['dropout'])
        # self.fc1 = nn.Linear(num_channels[-1], hidden_size)


        self.model = nn.Sequential(
                    nn.Linear(num_channels[-1], hidden_size),           # 输入10维，隐层20维
                    # nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=True),     # BN层，参数为隐层的个数
                    nn.LayerNorm(hidden_size, elementwise_affine=True),
                    nn.ReLU(),                     # 激活函数

                    # nn.Linear(hidden_size, hidden_size),           # 输入10维，隐层20维
                    # # nn.BatchNorm1d(hidden_size, affine=True, track_running_stats=True),     # BN层，参数为隐层的个数
                    # nn.LayerNorm(hidden_size, elementwise_affine=True),
                    # nn.ReLU(),   
         
                )


        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

        
    def forward(self, state):
       
        state = state.reshape(-1, input_channels, state_seq_len)
        x = self.tcn(state)
        x = x[:, :, -1]
        x = self.model(x)

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        action = self.action_range*mean.detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()


class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range, args):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(args, state_dim, action_dim, hidden_dim, action_range).to(device)        
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        # self.soft_q_net1.eval()
        # self.soft_q_net2.eval()
        # self.target_soft_q_net1().eval()
        # self.target_soft_q_net2().eval()
        # self.policy_net.eval()


        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        # soft_q_lr = 3e-4
        # policy_lr = 3e-4
        # alpha_lr  = 3e-4

        soft_q_lr = args['lr']
        policy_lr = args['lr']
        alpha_lr  = args['lr']

        # soft_q_lr = args.lr
        # policy_lr = args.lr
        # alpha_lr  = args.lr


        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path, ep):

        epoch, frame_idx, rewards = ep

        #保存模型的状态，可以设置一些参数，后续可以使用
        state = {'epoch': epoch + 1,#保存的当前轮数
                'frame_idx': frame_idx + 1,#保存的当前轮数
                'rewards': rewards,
                'soft_q_net1': self.soft_q_net1.state_dict(),#训练好的参数
                'soft_q_net2': self.soft_q_net2.state_dict(),#训练好的参数
                'policy_net': self.policy_net.state_dict(),#训练好的参数
                'soft_q_net1_optimizer': self.soft_q_optimizer1.state_dict(),#优化器参数,为了后续的resume
                'soft_q_net2_optimizer': self.soft_q_optimizer2.state_dict(),#优化器参数,为了后续的resume
                'policy_optimizer': self.policy_optimizer.state_dict(),#优化器参数,为了后续的resume                
                'alpha_optimizer':self.alpha_optimizer.state_dict(),#优化器参数,为了后续的resume
                # 'loss': best_pred#当前最好的精度

                }

        torch.save(state, path+'checkpoint.pth.tar')

        # torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        # torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        # torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):



        checkpoint = torch.load(path+'checkpoint.pth.tar')

        self.soft_q_net1.load_state_dict(checkpoint['soft_q_net1'])
        self.soft_q_net2.load_state_dict(checkpoint['soft_q_net2'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])

        self.soft_q_optimizer1.load_state_dict(checkpoint['soft_q_net1_optimizer'])
        self.soft_q_optimizer2.load_state_dict(checkpoint['soft_q_net2_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        self.epoch = checkpoint['epoch']
        self.frame_idx = checkpoint['frame_idx']
        self.rewards = checkpoint['rewards']

        if args.test:
            self.soft_q_net1.eval()
            self.soft_q_net2.eval()
            self.policy_net.eval()
        
        if args.go_on_train:
            self.soft_q_net1.train()
            self.soft_q_net2.train()
            self.policy_net.train()

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']



        # self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        # self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        # self.policy_net.load_state_dict(torch.load(path+'_policy'))

        # self.soft_q_net1.eval()
        # self.soft_q_net2.eval()
        # self.policy_net.eval()


def plot(rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    plt.show()


replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)


action_range=1.

# hyper-parameters for RL training
max_episodes  = 0  # 5000
# max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
max_steps = 500
frame_idx   = 0
batch_size  = 256
explore_steps = 200  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
rewards     = []
model_path = './model_save/sac_v2_'
# eps = 0
# model_path = './model_save/sac_v2_eps_{}_'.format(eps)

tensorboard_path = './model_save/'

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(tensorboard_path)

def tarin(sac_trainer, eps, frame_idx):
    # import os
    # is_go_on = os.path.exists('./model_save/sac_v2_checkpoint.pth.tar')        

    # if is_go_on:  # False 
    #     sac_trainer.load_model(model_path)
    #     eps = sac_trainer.epoch
    #     frame_idx = sac_trainer.frame_idx
    #     rewards = sac_trainer.rewards
    # else: 
    #     eps = 0
    #     frame_idx = 0

    # # print(eps,frame_idx)
    # print('eps: {}, frame_idx:{}'.format(eps,frame_idx))
    # input()
    # for eps in range(max_episodes):     
    
    while eps <= max_episodes:
        print('train:')

        state =  env.reset()
        episode_reward = 0
        
        
        for step in range(max_steps):
            if frame_idx > explore_steps:
                action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
            else:
                action = sac_trainer.policy_net.sample_action()

            next_state, reward, done, info = env.step(action)
            # env.render()       
                
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1


            # if args.env_name == 'VibrationEnv-v0':
            #     writer.add_scalar('Rewards/NoiseAmplitude', info['NoiseAmplitude'], frame_idx)
            #     writer.add_scalar('Rewards/VibrationAmplitude', info['VibrationAmplitude'], frame_idx)
            #     # writer.add_scalar('Rewards/input', info['input'], frame_idx)

            #     writer.add_scalars('states', {'x1_position':state[0],
            #                                     'x2_position':state[1],
            #                                     'x3_velocity':state[2],
            #                                     'x4_velocity':state[3],
            #                                     'x5_Acceleration':state[4],
            #                                     'x6_Acceleration':state[5]
            #                                     }, frame_idx)

            #     writer.add_scalars('states', {'x1_position':state[0],
            #                                     'x2_position':state[1] }, frame_idx)
            #     writer.add_scalars('states', {'x3_velocity':state[2],
            #                                     'x4_velocity':state[3] }, frame_idx)
            #     writer.add_scalars('states', {'x5_Acceleration':state[4],
            #                                     'x6_Acceleration':state[5] }, frame_idx)        
            #     pass  

            
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    _=sac_trainer.update(batch_size, reward_scale=1e-1, auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)

            if done:
                break

        if eps % 5 == 0 and eps>0: # plot and model saving interval
            # plot(rewards)
            # model_path = './model_save/sac_v2_eps_{}_'.format(eps)
            # sac_trainer.save_model(model_path, [eps, frame_idx, rewards] )
            pass

        # if episode_reward > max(rewards):
        #     print('save_model')
            # sac_trainer.save_model(model_path, [eps, frame_idx, rewards] )
        

        logging.info("train: the eps is {}, the t is {}, Episode Reward {}, NoiseAmplitude: {}, VibrationAmplitude: {}, input: {}"\
                .format(eps, max_steps, episode_reward, info['NoiseAmplitude'], info['VibrationAmplitude'], info['input'] ))

        if args.env_name == 'VibrationEnv-v0':
            print("the eps is {}, the t is {}, Episode Reward {}, NoiseAmplitude: {}, VibrationAmplitude: {}, input: {}"\
                .format(eps, max_steps, episode_reward, info['NoiseAmplitude'], info['VibrationAmplitude'], info['input'] ))           
            # logger.info("the eps is {}, the t is {}, Episode Reward {}, NoiseAmplitude: {}, VibrationAmplitude: {}, input: {}"\
            #         .format(eps, max_steps, episode_reward, info['NoiseAmplitude'], info['VibrationAmplitude'], info['input'] ))

        else:
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)

        # writer.add_scalar('Rewards/ep_r', episode_reward, global_step=eps)

        rewards.append(episode_reward)

        eps += 1

    # sac_trainer.save_model(model_path, [eps, frame_idx] )

    # return rewards.mean()
    return frame_idx

def test(sac_trainer):    
    # sac_trainer.load_model(model_path)

    env_id = 'VibrationEnv-v0'
    # print(env_id)
    
    for eps in range(1):  #10
        print('test:')
        eval_states = []
        episodes =[]

        eval_BottomLayerForce = []
        eval_BottomLayerForceRate = []

        eval_input = []
        eval_action = []


        state =  env.reset()
        episode_reward = 0
        with torch.no_grad():
            for step in range(int(0.5*2*max_steps)):   #max_steps
                action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
                # if ENV ==  'Reacher':
                #     next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                # elif ENV ==  'Pendulum':
                #     next_state, reward, done, _ = env.step(action)
                #     env.render()   


                next_state, reward, done, info = env.step(action)
                # env.render()   

                episode_reward += reward
                state=next_state

                episodes.append(env.counts)
                eval_states.append(state)

                eval_BottomLayerForce.append(info['BottomLayerForce'])
                eval_BottomLayerForceRate.append(info['BottomLayerForceRate'])

                eval_input.append(info['input'])
                eval_action.append(action)

            # print('Episode: ', eps, '| Episode Reward: ', episode_reward)


            # print("the eps is {}, the t is {}, Episode Reward {}, NoiseAmplitude: {}, VibrationAmplitude: {}, input: {},\
            #         BottomLayerForceRate: {}"  \
            #     .format(eps, max_steps, episode_reward, info['NoiseAmplitude'], info['VibrationAmplitude'], info['input'],\
            #         info['BottomLayerForceRate']  \
            #             ))           

            logging.info("test: the eps is {}, the t is {}, Episode Reward {}, NoiseAmplitude: {}, VibrationAmplitude: {}, input: {}"\
                    .format(eps, max_steps, episode_reward, info['NoiseAmplitude'], info['VibrationAmplitude'], info['input'] ))
            print(type(episode_reward),episode_reward)

            return episode_reward



def main(args):
    
    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, args = args  )
    import os
    is_go_on = os.path.exists('./model_save/sac_v2_checkpoint.pth.tar')     
    # print('###', is_go_on)   
    is_go_on = False
    if is_go_on:  # False 
        sac_trainer.load_model(model_path)
        eps = sac_trainer.epoch
        frame_idx = sac_trainer.frame_idx
        rewards = sac_trainer.rewards
    else: 
        eps = 0
        frame_idx = 0
    

    # print(eps,frame_idx)
    print('eps: {}, frame_idx:{}'.format(eps,frame_idx))



    # for epoch in range(1, args['epochs'] + 1):
    for epoch in range(1, args['epochs'] + 1):
        # train(args, model, device, train_loader, optimizer, epoch)
        # test_acc = test(args, model, device, test_loader)
        start, end = np.zeros(2), np.zeros(2)
        start = time.process_time(), time.perf_counter()

        frame_idx = tarin(sac_trainer, eps, frame_idx)
        # print('&&&',frame_idx)
        test_acc = test(sac_trainer)
        # test_acc = np.array(1) * args['lr'] * args['nhid']
        end = time.process_time(), time.perf_counter()

        print('本地时间：{}，第{}次执行时间：{},性能：{}'\
                    .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, np.array(end) - np.array(start), test_acc))
        logging.info('本地时间：{}，第{}次执行时间：{},性能：{}'\
                    .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, np.array(end) - np.array(start), test_acc))

        if epoch < args['epochs']:
        # if epoch < args.epochs:
            # report intermediate result
            nni.report_intermediate_result(test_acc)
            logging.debug('test accuracy %g', test_acc)
            logging.debug('Pipe send intermediate result done.')
        else:
            # report final result
            nni.report_final_result(test_acc)
            logging.debug('Final result is %g', test_acc)
            logging.debug('Send final result done.')



if __name__ == '__main__':
    # main(args)

    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logging.debug(tuner_params)
        params = vars(args)
        params.update(tuner_params)
        print(args)
        # input()
        main(params)
    except Exception as exception:
        logging.exception(exception)
        raise

