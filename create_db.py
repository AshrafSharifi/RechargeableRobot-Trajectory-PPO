# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import json
import random
import time
from dataclasses import dataclass
from Environment import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
from variables import variables
import pandas as pd
from functions import functions
from dB import dB
import pickle


    
complete_path = dict()
model_path = 'data/Runs/decresed_weights_1026'
with open(model_path+'/complete_path.pkl', 'rb') as file:
    complete_path = pickle.load(file)
db = dB()
db.create_train_db(complete_path)


            
    