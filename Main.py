import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from ADFPopulation import *
from CellPopulation import *
from ModelPopulation import *
from Model import *
from ADF import *
from Cell import *
from Utils import *
from Configs import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = BATCH_SIZE
DEVICE = "cpu"

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Step 1, 2
normal_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = False, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP)
reduction_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = True, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP)
reduction_cell_pop = CellPopulation(CELL_HEAD_LEN, CELL_TAIL_LEN, INIT_SIZE_CELL_POP, reduction_adf_pop)
model_pop = ModelPopulation(INIT_SIZE_MODEL_POP, NUM_OF_CONSECUTIVE_NORMAL_CELL, normal_adf_pop, reduction_cell_pop)
# Step 3
model_pop.train_population(train_loader, test_loader, model_pop.models_dict)
num_generation = 10
year = 0
while year <= 2:
    year += 1
    # Step 4
    print("*****Step 4 - kill bad gene*****")
    normal_adf_pop.kill_bad_genes()
    reduction_adf_pop.kill_bad_genes()
    # Step 5
    print("*****Step 5 - reproduction *****")
    normal_adf_pop.reproduction()
    reduction_adf_pop.reproduction()
    reduction_cell_pop.reproduction()
    model_pop.reproduction()
    # Step 6
    print("*****Step 6 - evaluate child pop")
    model_pop.evaluate_population_step_6(train_loader, test_loader, model_pop.child_models_dict)
    # Step 7
    print("*****Step 7 - survivor*****")
    model_pop.survivor_selection()
    reduction_cell_pop.remove_marked_kill_cell()
    # STep 8
    print("*****Step 8 - full training and update T-g, T-c*****")
    model_pop.evaluate_population_step_6(train_loader, test_loader, model_pop.models_dict)
    model_pop.increase_age()
    T_G = min([model.fitness for (model_id, model) in model_pop.models_dict.items()])
    T_C = 0.75*T_G
    print("\tUpdated T_G, T_C:\t", end = "")
    print("T_G = %.2f, T_C = %.2f" % T_G, T_C)
