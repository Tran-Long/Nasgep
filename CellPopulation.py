from Cell import *
from BasePopulation import *
from Configs import *
import numpy as np

class CellPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, pop_size, adf_population):
        super(CellPopulation, self).__init__(head_size, tail_size, pop_size)
        self.cells_dict = {}
        self.cells_usage = {}
        for i in range(self.pop_size):
            cell_id = CELL_PREFIX + str(i)
            self.cells_dict[cell_id] = Cell(head_size, tail_size, adf_population)
            self.cells_usage[cell_id] = 0
        self.keys_list = list(self.cells_dict.keys())
        self.population = list(self.cells_dict.values())

    def select_2_random_cell(self):
        available_cell_terms = [key for (key, value) in self.cells_usage.items() if value == 0]
        cell_1_term = np.random.choice(available_cell_terms)
        available_cell_terms.remove(cell_1_term)
        cell_2_term = np.random.choice(available_cell_terms)
        self.cells_usage[cell_1_term] = 1
        self.cells_usage[cell_2_term] = 1
        return self.cells_dict[cell_1_term], self.cells_dict[cell_2_term]