from Cell import *
from BasePopulation import *
from Configs import *
import numpy as np

'''
    Reduction population
'''
class ReductionCellPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, pop_size, adf_population):
        super(ReductionCellPopulation, self).__init__(head_size, tail_size, pop_size)
        self.cells_dict = {}
        self.cells_usage = {}
        for i in range(self.pop_size):
            cell_id = CELL_PREFIX + str(i)
            self.cells_dict[cell_id] = Cell(head_size, tail_size, adf_population, reduction_cell = True)
            self.cells_usage[cell_id] = 0
        self.keys_list = list(self.cells_dict.keys())
        self.population = list(self.cells_dict.values())

    def select_random_reduction_cell(self):
        available_cell_terms = [key for (key, value) in self.cells_usage.items() if value == 0]
        cell_term = np.random.choice(available_cell_terms)
        self.cells_usage[cell_term] = 1
        return self.cells_dict[cell_term]