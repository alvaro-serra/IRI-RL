import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import torch
from collections import OrderedDict
from src.utils import util
from .agents import BaseAgent
from src.networks.networks import NetworksFactory
from src.utils.plots import plot_estim
import numpy as np




# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_agent(BaseAgent):

    def __init__(self, opt):
        super(DQN_agent, self).__init__(opt)
        self._name = 'agent_DQN'
        
        # init input params
        self._init_set_input_params()

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt["model"]["load_epoch"] > 0:
            self.load()

        # init losses
        if self._is_train:
            self._init_losses()

        # prefetch inputs
        self._init_prefetch_inputs()

    def _init_set_input_params(self):
        self._B = self._opt[self._dataset_type]["batch_size"]               # batch
        self._S = self._opt[self._dataset_type]["image_size"]               # image size
        self._Ci = self._opt[self._dataset_type]["img_nc"]                  # num channels image
        self._Ct = self._opt[self._dataset_type]["target_nc"] * self._B     # num channels target

    def _init_create_networks(self):
        # create architecture/network/ used by the agent
        reg_type  = self._opt["networks"]["reg"]["type"]
        reg_hyper_params = self._opt["networks"]["reg"]["hyper_params"]
        self._reg = NetworksFactory.get_by_name(reg_type, **reg_hyper_params)
    
