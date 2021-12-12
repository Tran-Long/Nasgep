import torch.nn as nn
import numpy as np
from Configs import *
from Utils import *

class ADF(nn.Module):
    def __init__(self, adf_head=1, adf_tail=2, reproduction_genotype = None):
        super(ADF, self).__init__()
        if reproduction_genotype is None:
            self.genotype = self.init_data(adf_head, adf_tail)
        else:
            self.genotype = reproduction_genotype
        # self.root = build_tree_adf(self.genotype)
        self.fitness = -1
        self.is_used = False

    def init_data (self, adf_head, adf_tail):
        genotype = []
        """Normal init"""
        for i in range(adf_head):
            if np.random.rand() < 0.5:
                genotype.append(np.random.choice(CONV_TERMS))
            else:
                genotype.append("sum")
        for i in range(adf_tail):
            if np.random.rand() < 0.7:
                genotype.append(np.random.choice(CONV_TERMS))
            else:
                genotype.append(PREV_OUTPUT)
        return genotype



    def forward(self):
        self.fitness = abs(np.random.rand() + 7)

# temp = ADF()
# view_tree(temp.root)