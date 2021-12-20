import torch.nn as nn
import numpy as np
from Configs import *
from Utils import *

class ADF:
    def __init__(self, adf_head=ADF_HEAD_LEN, adf_tail=ADF_TAIL_LEN, for_reduction=True, reproduction_genotype = None):
        super(ADF, self).__init__()
        if reproduction_genotype is None:
            self.genotype = self.init_data(adf_head, adf_tail)
        else:
            assert len(reproduction_genotype) == (adf_tail + adf_head), "Wrong pre-build adf genotype"
            self.genotype = reproduction_genotype
        self.root = build_tree_adf(self.genotype)
        if for_reduction:
            self.root = self.add_pwbr_for_reduction(self.root)
        self.fitness = -1
        self.is_used = 0

    @staticmethod
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

    def add_pwbr_for_reduction(self, root):
        if root.left is not None:
            if root.left.value == PREV_OUTPUT:
                temp = root.left
                root.left = Node(POINT_WISE_BEFORE_REDUCTION_TERM)
                root.left.left = temp
            else:
                root.left = self.add_pwbr_for_reduction(root.left)
        if root.right is not None:
            if root.right.value == PREV_OUTPUT:
                temp = root.right
                root.right = Node(POINT_WISE_BEFORE_REDUCTION_TERM)
                root.right.left = temp
            else:
                root.right = self.add_pwbr_for_reduction(root.right)
        return root

    def set_fitness(self, fitness):
        self.fitness = max(self.fitness, fitness)
# temp = ADF()
# view_tree(temp.root)