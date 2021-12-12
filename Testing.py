import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from BasePopulation import *




class Test():
    def __init__(self, m):
        print("Hello"+m)
    def plus(self, x, y):
        return x+y
    def minus(self, x, y):
        return x-y

class Test2(Test):
    def __init__(self, m, n):
        super(Test2, self).__init__(m)
        print(n)
        self.x = 7
        self.y = 3

t = Test2("Long", 73)
print(t.minus(t.x, t.y))
