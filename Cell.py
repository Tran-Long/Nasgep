import torch.nn as nn
import numpy as np
from Configs import *
from Utils import *

class Cell(nn.Module):
    def __init__(self, cell_head, cell_tail, adf_population, reproduction_genotype=None):
        if reproduction_genotype is not None:
            self.genotype = reproduction_genotype
            self.adfs_dict = {}
            self.adfs_genotype_dict= {}
            for array_element in self.genotype:
                if ADF_FREFIX in array_element:
                    self.adfs_dict[array_element] = adf_population.adfs_dict[array_element]
                    self.adfs_genotype_dict[array_element] = adf_population.genotypes_dict[array_element]
        else:
            self.genotype, self.adfs_dict, self.adfs_genotype_dict = self.init_data(cell_head, cell_tail, adf_population)
        self.root = build_tree_cell(self.genotype, self.adfs_genotype_dict)
        self.mark_killed = False
        self.is_used = False
        self.fitness = -1

    @staticmethod
    def init_data(cell_head, cell_tail, adf_population):
        adf_population_terms = adf_population.keys_list
        genotype = []
        adfs_genotype_dict = {}
        adfs_dict = {}
        for i in range(cell_head):
            genotype.append(np.random.choice(CELL_FUNCTION))
        for i in range(cell_tail):
            adf_genotype_key = np.random.choice(adf_population_terms)
            genotype.append(adf_genotype_key)
            adfs_genotype_dict[adf_genotype_key] = adf_population.genotypes_dict[adf_genotype_key]
            adfs_dict[adf_genotype_key] = adf_population.adfs_dict[adf_genotype_key]
        return genotype, adfs_dict, adfs_genotype_dict

    def create_modules_dict(self, base_channel, nonce=0):
        module_dict = nn.ModuleDict()

        return module_dict

    def create_dict(self, root, value_dict, prev_outputs, nonce=0):
        nonce = nonce + 1
        if root.value == POINT_WISE_TERM:
            conv_key = POINT_WISE_TERM + nonce
            value_dict[conv_key] = conv_block(POINT_WISE_TERM, )

    def forward(self):
        self.fitness = abs(np.random.rand() + 7)

