from ObjectPopulation.Cell import *
from Base.BasePopulation import *
from Utilities.Configs import *
import numpy as np

'''
    Reduction population
'''
class CellPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, pop_size, r_adf_population):
        super(CellPopulation, self).__init__(head_size, tail_size, pop_size)
        self.cells_dict = {}
        while len(self.cells_dict) < pop_size:
            new_cell = Cell(r_adf_population, head_size, tail_size)
            cell_id = CELL_PREFIX + str(self.nonce)
            self.nonce += 1
            self.cells_dict[cell_id] = new_cell
        self.keys_list = list(self.cells_dict.keys())
        self.population = list(self.cells_dict.values())
        self.adf_population = r_adf_population
        self.function_set = CELL_FUNCTION
        self.terminal_set = r_adf_population.keys_list

    def select_random_reduction_cell(self):
        available_cell_terms = [cell_key for (cell_key, cell) in self.cells_dict.items() if cell.is_used == False]
        cell_term = np.random.choice(available_cell_terms)
        self.cells_dict[cell_term].is_used = True
        return self.cells_dict[cell_term]

    def add_cell(self, genotype):
        cell = Cell(self.adf_population, self.head_size, self.tail_size, reproduction_genotype = genotype)
        cell_id = CELL_PREFIX + str(self.nonce)
        self.nonce += 1
        self.cells_dict[cell_id] = cell
        self.keys_list.append(cell_id)
        self.child_population.append(cell)
        self.child_pop_size += 1

    def remove_cell(self, cell_id):
        cell_idx_in_list = self.keys_list.index(cell_id)
        self.cells_dict.pop(cell_id)
        self.keys_list.pop(cell_idx_in_list)
        self.population.pop(cell_idx_in_list)
        self.pop_size -= 1

    def merge_population(self):
        self.population = self.population + self.child_population
        self.pop_size += self.child_pop_size
        self.child_population = []
        self.child_pop_size = 0

    def reproduction(self):
        """ Function only for reduction cell population """
        print("\tBefore:")
        write_log("Before: ")
        print("\t\t", end = "")
        print(self.keys_list)
        write_log(self.get_info_string())
        while self.child_pop_size < self.pop_size:
            new_cell_genotype1, new_cell_genotype2 = self.reproduction_individual_genotype()
            self.add_cell(new_cell_genotype1)
            self.add_cell(new_cell_genotype2)
        self.merge_population()
        print("\tAfter:")
        write_log("After: ")
        print("\t\t", end = "")
        print(self.keys_list)
        write_log(self.get_info_string())

    def remove_marked_kill_cell(self):
        cell_id_to_kill = [cell_id for (cell_id, cell) in self.cells_dict.items() if cell.mark_killed == True]
        for cell_id in cell_id_to_kill:
            self.remove_cell(cell_id)

    def get_info_string(self):
        string = "["
        for cell_id in self.keys_list:
            string += cell_id + ", "
        string += "]"
        return string