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

adf_pop = ADFPopulation(1, 2, 10)
c = Cell(4, 5, adf_pop)
view_tree(c.root)
view_tree_channel(c.root)




