from collections import deque, namedtuple
from itertools import count
import random
import torch
import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

GAMMA = 0.999
replayMemory = namedtuple("replayMemory",("state","action","reward","nextState","done"))

class Neural(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.fc1 = nn.Linear(input, 1024)
        self.out = nn.Linear(1024,output)

    def forward(self,x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class Agent:
    def __init__(self):
        self.size = 100000
        self.LR = 0.0000000001
        self.memory = deque(maxlen=self.size)
        self.env = gym.make("maze-sample-5x5-v0")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.onlineNetwork = Neural(self.env.observation_space.shape[0],self.env.action_space.n).to(self.device)
        self.targetNetwork = Neural(self.env.observation_space.shape[0],self.env.action_space.n).to(self.device)
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        self.adamOptimizer = optim.Adam(self.onlineNetwork.parameters(),self.LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.adamOptimizer,1000,0.001,verbose=False)
        self.epStart = 1
        self.epDecay = 0.9999
        self.epEnd = 0.1
        self.batch = 512
        self.scores = []
        self.episodeLoss = []
        self.lossPlot = []
        

    def greedyPolicy(self):
        if self.epStart > self.epEnd:
            self.epStart *= self.epDecay
            return self.epStart
        return self.epStart

    def getAction(self,state):
        if self.greedyPolicy() > np.random.rand():
            return np.random.choice(self.env.action_space.n)
        else:
            with torch.no_grad():
                state = state.clone().detach()
                return self.onlineNetwork(state).argmax(0).item()

    def experienceReplay(self,experience):
        if len(self.memory) < self.size:
            self.memory.append(experience)
        else:
            self.memory.popleft()
            self.memory.append(experience)
    
    def canLearn(self):
        return len(self.memory) >= self.batch

    def dataStructureManagement(self,batch):
        state = torch.zeros((self.batch,self.env.observation_space.shape[0]))
        nextState = torch.zeros((self.batch,self.env.observation_space.shape[0]))
        state = torch.stack(batch.state).to(self.device)
        reward = torch.cat(batch.reward).to(self.device)
        action = torch.cat(batch.action).to(self.device)
        nextState = torch.stack(batch.nextState).to(self.device)
        done = torch.cat(batch.done).to(self.device)
        return state,nextState,reward,action,done
    
    def updateNetworks(self):
        batch = replayMemory(*zip(*(random.sample(self.memory,self.batch))))
        state,nextState,rewards,actions,done = self.dataStructureManagement(batch)
        currentQ = self.onlineNetwork(state.float()).gather(dim=1,index=actions.unsqueeze(-1))
        a = self.onlineNetwork(nextState).argmax(-1,keepdim=True)
        targetQ = self.targetNetwork(nextState).gather(-1,a)
        formula = (GAMMA * targetQ.squeeze(1) * (1 - done.float())) + rewards
        self.usePer = True
        loss = F.mse_loss(currentQ,formula.unsqueeze(1).float())
        self.adamOptimizer.zero_grad()
        loss.backward()
        self.adamOptimizer.step()
        self.episodeLoss.append(loss.item())


    def toTensor(self,nextState,reward,done,action,state):
        state = torch.tensor(state).to(self.device)
        nextState = torch.tensor(nextState).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)
        action = torch.tensor([action]).to(self.device)
        return nextState,reward,done,action,state

    def Loop(self):
        overallScore = 0
        for episode in range(5000):
            state = self.env.reset()
            state = torch.tensor(state).float().to(self.device)
            for timestep in count():
                self.env.render()
                action = self.getAction(state)
                state = state.cpu().detach().numpy()
                nextState, reward, done, _ = self.env.step(action)
                nextState,reward,done,action,state = self.toTensor(nextState,reward,done,action,state)
                self.experienceReplay((state,action,reward,nextState,done))
                state = nextState
                overallScore += reward
                if self.canLearn():
                    self.updateNetworks()
                    if episode > 250 and episode < 375:
                        self.scheduler.step()
                    self.lossPlot.append(sum(self.episodeLoss)/len(self.episodeLoss))
                if done:
                    break
                if timestep % 500 == 0:
                    self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
                    
o = Agent()
o.Loop()