import torch.nn as nn
import numpy as np
from Configs import *
from Utils import *


def init_data (adf_head, adf_tail):
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


class ADF:
    def __init__(self, adf_head=1, adf_tail=2, reproduction_genotype = None):
        super(ADF, self).__init__()
        if reproduction_genotype is None:
            self.genotype = init_data(adf_head, adf_tail)
        else:
            assert len(reproduction_genotype) == (adf_tail + adf_head), "Wrong pre-build adf genotype"
            self.genotype = reproduction_genotype
        self.fitness = -1
        self.is_used = False

# temp = ADF()
# view_tree(temp.root)