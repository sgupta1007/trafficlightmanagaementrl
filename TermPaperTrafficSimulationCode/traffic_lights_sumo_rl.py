import math
from scipy import special
from tqdm import trange
import numpy as np
import copy
import os
import sumo_rl
from sumo_rl import env,parallel_env
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import pickle


#setting default tensor type
torch.set_default_tensor_type(torch.DoubleTensor)
#get current working directory
curdir=os.getcwd()



#creating a grid 4*3 custom environment this environment will not use gui

def grid4x3(parallel=True,**kwargs):
    kwargs.update({'net_file':curdir+'/3_4trafficgrid.net.xml',
                   'route_file':curdir+'/resultgrid30.rou.xml',
                   'num_seconds':10000,'use_gui':True})
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)

#calculating one hop neighbourhood
def get_one_hop_neigh(env):
    tll=env.unwrapped.env.sumo.trafficlight.getIDList()
    trafficlights={}
    for i in tll:
        trafficlights[i]=env.unwrapped.env.sumo.trafficlight.getControlledLanes(i)
    one_hop_neigh={}
    for i in trafficlights.keys():
        one_hop_neigh[i]=set()
        one_hop_neigh[i].add(i)
    for i in trafficlights.keys():
        t_l_1=trafficlights[i]
        for j in trafficlights.keys():
            if i!=j:
                t_l_2=trafficlights[j]
                for k in range(len(t_l_1)):
                    for l in range(len(t_l_2)):
                        if(t_l_1[k]==('-'+t_l_2[l]) or t_l_2[l]==('-'+t_l_1[k])):
                            one_hop_neigh[i].add(j)
    return one_hop_neigh


#replay buffer created for environment
    
class ReplayBuffer:
    def __init__(self,cap): 
        self.buffer=[]
        self.cap=cap
    # x needs to be well formed by the algo which is sending the 
    def insert(self,x):
        if(len(self.buffer)<self.cap):
              self.buffer.append(x)
        else:
              self.buffer.pop(0)
              self.buffer.append(x)
    # get sample from buffer
    def sample(self):
        return rd.choice(self.buffer)
    #return a sample from replay buffer
    def sample_batch(self,batch_size):
        batch=[]
        for i in range(batch_size):
            batch.append(self.sample())
        return batch
    def capacity(self):
        return self.cap
    #return buffer's current capacity  
    def current_size(self):
        return len(self.buffer)
        
class Network(nn.Module):
    def __init__(self,input_layer_dim,hidden_layer_dim,output_layer_dim):
        super(Network,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(11,10),nn.ReLU(),nn.Linear(10,2),nn.ReLU())
    def forward(self,x):
        return self.layer1(x)


class SmartTraffic:
    def __init__(self,env,hidden_layer_size,replay_buffer_cap,batch_size,max_time_steps_per_episode,upt_time,gamma=0.9,eps=0.1):
        #self.env is env
        self.env=env
        #initialization states of the environment
        self.state=self.env.reset()
        #one hop neighbourhood
        self.ohn=get_one_hop_neigh(env)
        #dictionary of policy networks
        self.policy_networks={}
        #dictionary of target networks
        self.target_networks={}
        #dictionary of optimizer
        self.optimizers={}
        #dictionary of loss objects
        self.losses={}
        #dictionary of history
        self.thistory={}
        #dictionary of oldhistory
        self.ohistory={}
        #replay buffers for each of the agents
        self.replay_buffers={}
        #replay buffer_cap
        self.replay_buffer_cap=replay_buffer_cap
        
        #batch size of replay buffer
        if batch_size>replay_buffer_cap:
            batch_size=replay_buffer_cap
        #assiging the batch size    
        self.batch_size=batch_size
        #max_time_steps_per_episode
        self.max_time_steps_per_episode=max_time_steps_per_episode
        #update_time_steps_for_target
        self.upttime=upt_time
        
        #tll is traffic light list or agent list
        tll=env.unwrapped.env.sumo.trafficlight.getIDList()

        
        #getting max observation size
        self.max_obs_size=0
        for i in tll:
            temp_size=self.env.observation_spaces[i].shape[0]
            if temp_size>self.max_obs_size:
                self.max_obs_size=temp_size
        

        #constructing policy networks  target networks and replay buffers
        for i in tll:
            self.replay_buffers[i]=ReplayBuffer(replay_buffer_cap)
            self.policy_networks[i]=Network(self.max_obs_size,hidden_layer_size,self.env.action_spaces[i].n)
            self.target_networks[i]=Network(self.max_obs_size,hidden_layer_size,self.env.action_spaces[i].n)
            #loading the policy network weights in target networks
            self.target_networks[i].load_state_dict(self.policy_networks[i].state_dict())
            self.target_networks[i].eval()
            #optimizers for network i
            self.optimizers[i]=optim.RMSprop(self.policy_networks[i].parameters())
            #loss for network i
            self.losses[i]=nn.SmoothL1Loss()
            
            #history for all agents
            #make all history observation size same
            self.ohistory[i]=np.concatenate((self.state[i],np.zeros(self.max_obs_size-self.state[i].shape[0])))
            

            
        #epsilon gamma rewards
        self.rewards=[]
        self.eps=eps
        self.gamma=gamma
            
        
        
        
    def info(self):
        print(self.state)
        return self.ohn
        
    #update new history as old history
    def update_history(self):
        #use clear to remove all the elements of dictionary
        pass
    #get action from an agent
    
    def get_action(self,agent_id):
        obs=[]
        rand_n=rd.random()
        # for eps exploration
        if rand_n>1-self.eps:
            return self.env.action_spaces[agent_id].sample()
        #getting observation for all one hop neighbourhood
        for i in self.ohn[agent_id]:
            obs.append(self.ohistory[i])
        #print(obs)
        obs=sum(obs)
        #print(obs)
        obs=obs/len(self.ohn[agent_id])
        #print(obs)
        #print(obs.dtype)
        obs=obs.tolist()
        action=self.policy_networks[agent_id](torch.tensor(obs))
        action=action.detach()
        action=np.argmax(action.numpy())
        return action
    def all_done(self,d1):
        status=1
        for i in d1.keys():
            if d1[i]==False:
                status=0
                return False
        return True

    #no of steps for pretraining and filling the buffer   
    def pretraining(self,max_steps):
        for i in range(max_steps):
            actions={}
            #choosing random action for each agent
            for j in self.env.agents:
                actions[j]=self.env.action_spaces[j].sample()
            obs,reward,done,info=self.env.step(actions)
            #updating replay buffer for each agent
            for j in self.env.agents:
                t=(np.concatenate((self.state[j],np.zeros(self.max_obs_size-self.state[j].shape[0]))),actions[j],reward[j],np.concatenate((obs[j],np.zeros(self.max_obs_size-obs[j].shape[0]))),done[j])
                #insert sample into replay buffer
                self.replay_buffers[j].insert(t)
            self.state=obs
            #updating history
            for j in self.env.agents:
                self.ohistory[j]=np.concatenate((self.state[j],np.zeros(self.max_obs_size-self.state[j].shape[0])))
    #get one epoch done:
    def run_epoch(self):

        ep_rewards=0
        for t in range(self.max_time_steps_per_episode):
            
        #actions for all the agents get one action
            actions={}
            for i in self.env.agents:
                actions[i]=self.get_action(i)
            obs,reward,done,info=self.env.step(actions)
            #iteration reward is 0
            iter_rwd=0
            #average all agents rewards
            for i in reward.keys():
                iter_rwd+=reward[i]
            if(len(reward)==0):
                iter_rwd=0
            else:
                iter_rwd=iter_rwd/len(reward)   
            ep_rewards+=(iter_rwd*(self.gamma**t))
            if(self.all_done(done)):
                break
            #insert this sample into replay buffers
            for j in self.env.agents:
                t1=(np.concatenate((self.state[j],np.zeros(self.max_obs_size-self.state[j].shape[0]))),actions[j],reward[j],np.concatenate((obs[j],np.zeros(self.max_obs_size-obs[j].shape[0]))),done[j])
                #insert sample into replay buffer
                self.replay_buffers[j].insert(t1)
            self.state=obs        
            #updating history
            for j in self.env.agents:
                self.ohistory[j]=np.concatenate((self.state[j],np.zeros(self.max_obs_size-self.state[j].shape[0])))
            #sample and update paramaters
            for j in self.env.agents:
                #print("agent",j)
                batch=self.replay_buffers[j].sample_batch(self.batch_size)
                non_final_mask=[]
                non_final_states_batch=[]
                states_batch=[]
                action_batch=[]
                reward_batch=[]
                for k in batch:
                    if not k[4]:
                        non_final_mask.append(1)
                        non_final_states_batch.append(torch.tensor(k[3]))
                    else:
                        non_final_mask.append(0)
                non_final_mask=torch.tensor(non_final_mask,dtype=torch.bool)
                for k in batch:
                    states_batch.append(torch.tensor(k[0]))
                    action_batch.append(k[1])
                    reward_batch.append(torch.tensor(k[2]).detach())
                states_batch=torch.stack(states_batch)
                reward_batch=torch.stack(reward_batch)
                non_final_states_batch=torch.stack(non_final_states_batch)
                #states batch  done now we need to get action values and gradient descent
                next_state_values=torch.zeros(self.batch_size)
                #computing target Q
                vals=self.target_networks[j](non_final_states_batch).max(1)[0].detach()
                #non final state Q values are assigned 
                next_state_values[non_final_mask]=vals#self.target_networks[j](non_final_states_batch).max(1)[0].detach()
                expected_next_state_values=next_state_values*self.gamma+reward_batch
                #computing Q(s,a) for batch
                temp_val=self.policy_networks[j](states_batch)
                #state action values for batch
                state_action_values=torch.stack([temp_val[m][action_batch[m]] for m in range(len(action_batch))] )
                #computing loss
                loss = self.losses[j](state_action_values, expected_next_state_values)
                #optimizers zero out
                self.optimizers[j].zero_grad()
                loss.backward()
                #clamping  the gradients
                for param in self.policy_networks[j].parameters():
                    param.grad.data.clamp(-1,1)
                #stepping optimizers
                self.optimizers[j].step()
                
                if t%(self.upttime)==0:
                    self.target_networks[j].load_state_dict(self.policy_networks[j].state_dict())
            
            
        return ep_rewards
                
        
            
        
    


    

env=grid4x3()
env.reset()
#random walk

#for i in range(500):
    #actions={}
    #for j in env.agents:
    #    actions[j]=rd.choice([0,1])
    #env.step(actions)




#print("Successful Random walk")
#replay buffer with capacity 10
d1=SmartTraffic(env,100,10,5,5,2)
#print(d1.info())
#print(d1.get_action('J0'))
d1.pretraining(d1.replay_buffer_cap)
#print('rb')
#print(d1.replay_buffers['J0'].buffer)
for i in range(2000):
    print(d1.run_epoch())
#print('getting k hop neighbourhood')
#ohn=get_one_hop_neigh(env)
#print(ohn)

    

