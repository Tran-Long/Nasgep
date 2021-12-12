import torch.nn as nn
import numpy as np
from Configs import *
from Utils import *

class Model:
    def __init__(self, cell_population, n = NUM_OF_CONSECUTIVE_NORMAL_CELL,
                 normal_cell=None, reduction_cell=None, for_dataset="cifar-10"):
        self.fitness = -1
        self.mark_killed = False
        self.epoch_cnt = 0
        self.age = 0
        # Select cells
        if normal_cell is None and reduction_cell is None:
            normal_cell, reduction_cell = cell_population.select_2_random_cell()
        self.all_module_block_list = nn.ModuleDict()
        # Init network representation
        current_input_channel = 3
        prev_outputs = []
        if for_dataset == "cifar-10":
            self.all_module_block_list.append(nn.ModuleList([conv_block(STEM_TERM, 3, NUM_CHANNELS)]))
            current_input_channel = 16
            prev_outputs.append(0)

            # 1st normal block => channel = 16
            for i in range(n):
                t_md = nn.ModuleDict()



    def calculateFitness(self):
        self.fitness = abs(np.random.randn() + 7)