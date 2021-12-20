import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from BasePopulation import *
from Cell import *
from Utils import *
from ADF import *
from ADFPopulation import *
from CellPopulation import *
from Model import *

adf_pop = ADFPopulation(1, 2, for_reduction = False)
r_adf_pop = ADFPopulation(1, 2, for_reduction = True)
r_cell_pop = CellPopulation(4, 5, 10, r_adf_pop)
n_cell_genotype = Cell(adf_pop)
r_cell_genotype = Cell(r_adf_pop)
model = Model(adf_pop, r_cell_pop)
print(model.num_params)
# c = Cell(adf_pop, 4, 5, reduction_cell = True, reproduction_genotype = cell_genotype)

# view_tree_channel(c.root)
#
# po = [0, 0, 0]
# value_dict = {}
# value_dict, nonce = c.create_dict(c.root, value_dict, po, 32, 0)
# print(value_dict.keys())
# print(value_dict)
# print(nonce)

# view_tree(c.root)

"""
    Test affecting of population object
"""
# cell_pop = CellPopulation(4, 5, 10, adf_pop)
# new_adf_genotype = ["sum", "dep_3x3", "isep_3x3"]
# print(cell_pop.terminal_set)
#
# adf_pop.add_adf(new_adf_genotype)
# print(cell_pop.terminal_set)
#
#
# adf_pop.remove_adf("adf1")
# print(cell_pop.terminal_set)
# print(c.adf_population.keys_list)

