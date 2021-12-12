import torch.nn as nn
import numpy as np
from Configs import *

class Model:
    def __init__(self, N = NUM_OF_CONSECUTIVE_NORMAL_CELL):
        self.fitness = -1
        self.mark_killed = False
        self.epoch_cnt = 0
        self.age = 0
        # Init network representation
        t_network = ["input", "init_conv"]
        for i in range(3):
            t_network = t_network + [ "n_cell" for i in range(N)]
            if i < 2:
                t_network.append("r_cell")
        t_network = t_network + ["BN_Relu", "Global_AVG_Pooling", "Dense", "Softmax"]
        self.network = t_network
    def calculateFitness(self):
        self.fitness = abs(np.random.randn() + 7)