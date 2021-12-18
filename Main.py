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

batch_size = 64

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Step 1, 2
normal_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = False, pop_size = 10, max_size = 20)
reduction_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = True, pop_size = 10, max_size = 20)
reduction_cell_pop = CellPopulation(CELL_HEAD_LEN, CELL_TAIL_LEN, 3, reduction_adf_pop)
model_pop = ModelPopulation(3, NUM_OF_CONSECUTIVE_NORMAL_CELL, normal_adf_pop, reduction_cell_pop)
# Step 3
model_pop.train_population(train_loader, test_loader, model_pop.models_dict)
num_generation = 10
year = 0
while year <= 2:
    year += 1
    # Step 4
    normal_adf_pop.kill_bad_genes()
    reduction_adf_pop.kill_bad_genes()
    # Step 5
    normal_adf_pop.reproduction()
    reduction_adf_pop.reproduction()
    reduction_cell_pop.reproduction()
    model_pop.reproduction()
    # Step 6
    model_pop.evaluate_population_step_6(train_loader, test_loader, model_pop.child_models_dict)
    # Step 7
    model_pop.survivor_selection()
    reduction_cell_pop.remove_marked_kill_cell()
    # STep 8
    model_pop.merge_dict()
    model_pop.evaluate_population_step_6(train_loader, test_loader, model_pop.models_dict)
    T_G = min([model.fitness for (model_id, model) in model_pop.models_dict.items()])
    T_C = 0.75*T_G
