from Cell import *
from BasePopulation import *
from Configs import *

class CellPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, pop_size, adf_population):
        super(CellPopulation, self).__init__(head_size, tail_size, pop_size)
        self.cells_dict = {}
        for i in range(self.pop_size):
            cell_id = CELL_PREFIX + str(i)
            self.cells_dict[cell_id] = Cell(head_size, tail_size, adf_population)
        self.keys_list = list(self.cells_dict.keys())
        self.population = list(self.cells_dict.values())
