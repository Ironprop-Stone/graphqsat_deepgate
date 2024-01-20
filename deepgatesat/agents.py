import torch
import numpy as np
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX
import torch.nn as nn
import random

class Agent(object):
    def act(self, state):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

class MiniSATAgent(Agent):
    def act(self, observation):
        # this will make GymSolver use VSIDS to make a decision
        return -1

    def __str__(self):
        return "<MiniSAT Agent>"



class CircuitAgent:
    def __init__(self, ckt_net, args):
        self.net = ckt_net
        self.device = args.device
        self.dg_emb = None

    def forward(self, graph):
        return self.net(graph)

    def act(self, hist_buffer, eps):
        graph = hist_buffer[-1]
        eps = 0     # Debug
        if np.random.random() < eps:
            acts = range(len(graph.valid_mask)*2)
            return int(np.random.choice(acts))
        else:
            qs = self.forward(graph)
            return self.choose_actions(qs)

    def choose_actions(self, qs):
        return qs.flatten().argmax().item()