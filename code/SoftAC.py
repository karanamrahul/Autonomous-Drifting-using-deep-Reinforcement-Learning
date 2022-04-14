#  Soft Actor Critic (SAC) agent for Autonomous Drifting using Deep Reinforcement Learning


# -*- coding: utf-8 -*-


"""
Created on Mon April 11 2022

@author: Rahul Karanam

@brief Implementation of Soft Actor Critic (SAC) agent for Autonomous Drifting using Deep Reinforcement Learning.

I have written code following from the original Paper to implement the SAC algorithm but with slight modifications to make it work for my problem.

References: https://arxiv.org/abs/1801.01290
"""

# Importing the libraries
import numpy as np
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,MultivariateNormal
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global path
path = "./SAC_Results"


class Actor(nn.Module):
    """Actor (Policy) Model.
    Policy Network
                        
    @brief This class takes in an observation of the environment and returns the action that the actor chooses to execute.
    
    """

    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.log_std_head = nn.Linear(fc2_units,action_size) # log_std head for the policy network 
        self.min_log_std = -20 # Minimum value of log_std
        self.max_log_std = 2 # Maximum value of log_std

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_log_std = self.log_std_head(x) # log_std head for the policy network
        x_log_std = torch.clamp(x_log_std, self.min_log_std, self.max_log_std) # Clamp the log_std to be between the min and max values
        
        return x, x_log_std
    
    
class Critic(nn.Module):
    """Critic (Value) Model.
    Value Network
    
    @brief This class takes in an observation of the environment and returns the value of the state.
    
    """

    def __init__(self, state_size, fcs1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fcs1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class Q_net(nn.Module):
    """ Q Network """
    
    def __init__(self, state_size, action_size, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
        """
        super(Q_net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size+action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, 1)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = state.reshape(-1,state.shape[-1])
        action = action.reshape(-1,action.shape[-1])
        x = torch.cat((state,action),-1) # Concatenate the state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
class ReplayBuffer:
    """ Replay Buffer 
    
    @brief This class is used to store the experience tuples in the form of (state, action, reward, next_state ,done)
    
    This is used for training the agent to learn from the experience state transition.
    
    """
    
    def __init__(self, buffer_size,state_size,action_size):
        """Initialize parameters and build model.
        Params
        ======
            buffer_size (int): Maximum size of the replay buffer
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        self.state_arr = torch.zeros(self.buffer_size,self.state_size).float().to(device)
        self.action_arr = torch.zeros(self.buffer_size,self.action_size).float().to(device)
        self.reward_arr = torch.zeros(self.buffer_size,1).float().to(device)
        self.next_state_arr = torch.zeros(self.buffer_size,self.state_size).float().to(device)
        self.done_arr = torch.zeros(self.buffer_size,1).float().to(device)
        self.ptr = 0
        
    def add(self,state,action,reward,next_state,done):
        """Add a new experience to the replay buffer"""
        self.ptr = (self.ptr) % self.buffer_size
        state = torch.tensor(state,dtype=torch.float32).to(device)
        action = torch.tensor(action,dtype=torch.float32).to(device)
        reward = torch.tensor(reward,dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state,dtype=torch.float32).to(device)
        done = torch.tensor(done,dtype=torch.float32).to(device)
        
        # Now we add the experience to the replay buffer (i.e to individual arrays)
        for array,element in zip([self.state_arr,self.action_arr,self.reward_arr,self.next_state_arr,self.done_arr],
                                 [state,action,reward,next_state,done]):
            array[self.ptr] = element
        self.ptr += 1
       
        
                 
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer"""
        ind = np.random.randint(0, self.buffer_size, size=batch_size,replace = False)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.state_arr[ind], self.action_arr[ind], self.reward_arr[ind], self.next_state_arr[ind], self.done_arr[ind]
                                                                                
    
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done
   
   
   
class SAC:
    """ Soft Actor-Critic (SAC) 
    
    @brief This class implements the Soft Actor Critic using the above defined classes
           for Policy Network ( Actor ) , Value Network + Q Network (Q_net) --> ( Critic )
           
    """
    def __init__(self,state_size,action_size,alpha,buffer_size=400000,lr=3e-4,gamma=0.99,tau=0.005,batch_size=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            lr (float): Learning rate
            gamma (float): Discount factor
            tau (float): Soft update parameter
            alpha (float): Entropy parameter
            beta (float): Entropy parameter
            batch_size (int): Batch size for training
            buffer_size (int): Size of the replay buffer
        """
        super(SAC,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Initialize the policy network 
        self.policy_net = Actor(state_size,action_size).to(device) 
        
        
        # Initialize the value network and the target network
        self.value_net = Critic(state_size).to(device)
        self.target_net = Critic(state_size).to(device)
        
        # Initialize the Q network 
        self.q_net_1 = Q_net(state_size,action_size).to(device)
        self.q_net_2 = Q_net(state_size,action_size).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(),lr=lr)
        self.q_net_1_optimizer = optim.Adam(self.q_net_1.parameters(),lr=lr)
        self.q_net_2_optimizer = optim.Adam(self.q_net_2.parameters(),lr=lr)
        
        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size,state_size,action_size)
        self.ptr = 0
        self.writer = SummaryWriter("runs/sac")
        
        self.value_critic_loss = nn.MSELoss()
        self.q_net_1_loss = nn.MSELoss()
        self.q_net_2_loss = nn.MSELoss()
        
        for trg_params,src_params in zip(self.target_net.parameters(),self.value_net.parameters()):
            trg_params.data.copy_(src_params.data)
            
        self.steering_range = (-0.8,0.8) # steering angle range for the car according to the simulator(CARLA) to showcase highspeed during drifting without rolling over.
        self.throttle_range = (0.6,1.0) # throttle range for the car according to the simulator(CARLA) to showcase highspeed during drifting.
        
        
 
    def select_action(self,state):
        """Select an action from the current policy"""
        state = torch.tensor(state,dtype=torch.float32).to(device)
        mu,sigma = self.policy_net(state)
        sigma = torch.exp(sigma)
        dist_space = Normal(mu,sigma)
        z = dist_space.sample()
        
        steer_action = float(torch.tanh(z[0,0])).detach().cpu().numpy()
        throttle_action = float(torch.sigmoid(z[0,1])).detach().cpu().numpy()
        
        steer_action = np.clip(steer_action,self.steering_range[0],self.steering_range[1])
        throttle_action = np.clip(throttle_action,self.throttle_range[0],self.throttle_range[1])
        
        return np.array([steer_action,throttle_action])
    
    
    def test(self,state):
        """Test the current policy"""
        state = torch.tensor(state,dtype=torch.float32).to(device)
        mu,sigma = self.policy_net(state)
    
        z = mu
        
        steer_action = float(torch.tanh(z[0,0])).detach().cpu().numpy()
        throttle_action = float(torch.sigmoid(z[0,1])).detach().cpu().numpy()
        
        steer_action = np.clip(steer_action,self.steering_range[0],self.steering_range[1])
        throttle_action = np.clip(throttle_action,self.throttle_range[0],self.throttle_range[1])
        
        return np.array([steer_action,throttle_action])
    
    def evaluate(self,state):
        "Evaulation of the current model"
        batch = state.size()[0]
        batch_mu,batch_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_sigma)
        dist_space = Normal(batch_mu,batch_sigma)
        noise = Normal(0,1).sample(sample_shape=(batch,self.action_size))
        
        action = torch.tanh(batch_mu + batch_sigma*noise.to(device))
        
        prob_log = dist_space.log_prob(batch_mu + batch_sigma*noise.to(device)) - torch.log(1-torch.pow(action,2)+1e-6)
        
        prob_log_0= prob_log[:,0].reshape(batch,1)
        prob_log_1= prob_log[:,1].reshape(batch,1)
        prob_log = torch.cat((prob_log_0,prob_log_1),dim=1)
        
        return action,prob_log,noise,batch_mu,batch_sigma
    
    
    def update(self):
        "This function will update the policy and the value network"
        if self.ptr % 500 == 0:
            print("Updating the target network")
            print("---- Started Training ----")
            print("Train - \t{} times".format(self.ptr))
        
        self.ptr = 0
        
        # Sample a batch of transitions
        state,action,reward,next_state,done = self.replay_buffer.sample(self.batch_size)
        
        target_val= self.target_net(next_state)
        next_q_val=reward + (1-done)*self.gamma*target_val
        
        expect_val = self.value_net(state)
        expect_q1,expect_q2 = self.q_net_1(state,action),self.q_net_2(state,action)
        sample_action,prob_log,noise,batch_mu,batch_sigma = self.evaluate(state)
        expect_q  = torch.min(self.q_net_1(state,sample_action),self.q_net_2(state,sample_action))
        next_val = expect_q - prob_log
        
         # Calculate the value loss
        # Loss function for the value network
        J_V_loss = self.value_critic_loss(expect_val,next_val.detach()).mean()
        
        # Loss function for the Q network
        Q1_loss = self.q_net_1_loss(expect_q1,next_q_val.detach()).mean()
        Q2_loss = self.q_net_2_loss(expect_q2,next_q_val.detach()).mean()
        
        pi_loss = -(expect_q - prob_log).mean()  # Policy Loss
        #  The above line is adapted from the original paper
        
        self.writer.add_scalar("Loss/J_V_loss",J_V_loss,self.ptr)
        self.writer.add_scalar("Loss/Q1_loss",Q1_loss,self.ptr)
        self.writer.add_scalar("Loss/Q2_loss",Q2_loss,self.ptr)
        self.writer.add_scalar("Loss/pi_loss",pi_loss,self.ptr)
        
        
        # Update the networks
        # Update the value network
        self.value_optimizer.zero_grad()
        J_V_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.value_net.parameters(),0.5)
        self.value_optimizer.step()
        
        # Update the policy network (Q network)
        self.q_net_1_optimizer.zero_grad()
        Q1_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.q_net_1.parameters(),0.5)
        self.q_net_1_optimizer.step()
        
        self.q_net_2_optimizer.zero_grad()
        Q2_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.q_net_2.parameters(),0.5)
        self.q_net_2_optimizer.step()
        
        
        # Update the policy network (Policy network)
        self.policy_optimizer.zero_grad()
        pi_loss.backward(retain_graph = True)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),0.5)
        self.policy_optimizer.step()
        
        
        # Update the target networks
        for trg_params,src_params in zip(self.target_net.parameters(),self.value_net.parameters()):
            trg_params.data.copy_(trg_params.data*self.tau + src_params.data*(1-self.tau))
            
        self.ptr += 1
        
        
    def save_model(self,path,epoch,buffer_size):
        
        os.makedirs(path+str(buffer_size),exist_ok=True)
        torch.save(self.policy_net.state_dict(),path+str(buffer_size)+"/policy_net_"+str(epoch)+".pth")
        torch.save(self.value_net.state_dict(),path+str(buffer_size)+"/value_net_"+str(epoch)+".pth")
        torch.save(self.q_net_1.state_dict(),path+str(buffer_size)+"/q_net_1_"+str(epoch)+".pth")
        torch.save(self.q_net_2.state_dict(),path+str(buffer_size)+"/q_net_2_"+str(epoch)+".pth")
        
        print("Model Saved")
        
        
    def load_model(self,path,epoch,buffer_size):
        
        self.policy_net.load_state_dict(torch.load(path+str(buffer_size)+"/policy_net_"+str(epoch)+".pth"))
        self.value_net.load_state_dict(torch.load(path+str(buffer_size)+"/value_net_"+str(epoch)+".pth"))
        self.q_net_1.load_state_dict(torch.load(path+str(buffer_size)+"/q_net_1_"+str(epoch)+".pth"))
        self.q_net_2.load_state_dict(torch.load(path+str(buffer_size)+"/q_net_2_"+str(epoch)+".pth"))
        
        print("Model Loaded")
        
        
    def save_buffer(self,path,epoch,buffer_size):
        path = path+str(buffer_size)+'/'
        self.policy_net.load_state_dict(torch.load(path+"policy_net_"+str(epoch)+".pth"))
        self.value_net.load_state_dict(torch.load(path+"value_net_"+str(epoch)+".pth"))
        self.q_net_1.load_state_dict(torch.load(path+"q_net_1_"+str(epoch)+".pth"))
        self.q_net_2.load_state_dict(torch.load(path+"q_net_2_"+str(epoch)+".pth"))
        print("Model Loaded")