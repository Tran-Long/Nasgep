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
        for i in range(self.pop_size):
            cell_id = CELL_PREFIX + str(i)
            self.cells_dict[cell_id] = Cell(head_size, tail_size, adf_population, reduction_cell = True)
        self.keys_list = list(self.cells_dict.keys())
        self.population = list(self.cells_dict.values())
        self.adf_population = adf_population
        self.function_set = CELL_FUNCTION
        self.terminal_set = adf_population.keys_list

    def select_random_reduction_cell(self):
        available_cell_terms = [cell_key for (cell_key, cell) in self.cells_dict.items() if cell.is_used == False]
        cell_term = np.random.choice(available_cell_terms)
        self.cells_dict[cell_term].is_used = True
        return self.cells_dict[cell_term]

    def add_cell(self, genotype):
        cell = Cell(self.head_size, self.tail_size, self.adf_population, reduction_cell = True, reproduction_genotype = genotype)
        cell_id = CELL_PREFIX + str(self.nonce)
        self.cells_dict[cell_id] = cell
        self.keys_list.append(cell_id)
        self.population.append(cell)

    def remove_cell(self, cell_id):
        cell_idx_in_list = self.keys_list.index(cell_id)
        self.cells_dict.pop(cell_id)
        self.keys_list.pop(cell_idx_in_list)
        self.population.pop(cell_idx_in_list)