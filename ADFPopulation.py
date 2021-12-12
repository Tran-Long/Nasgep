from ADF import *
from BasePopulation import *
from Configs import *

class ADFPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, pop_size):
        super(ADFPopulation, self).__init__(head_size, tail_size, pop_size)
        self.adfs_dict = {}
        self.genotypes_dict = {}
        for i in range(self.pop_size):
            adf_id = ADF_FREFIX + str(i)
            self.adfs_dict[adf_id] = ADF(1, 2)
            self.genotypes_dict[adf_id] = self.adfs_dict[adf_id].genotype
        self.keys_list = list(self.genotypes_dict.keys())
        self.population = list(self.adfs_dict.values())


# from Utils import *
# from Cell import *
# test_adf_pop = ADFPopulation(1, 2, pop_size = 10)
# test_cell = Cell(4, 5, adf_population = test_adf_pop)
# view_tree(test_cell.root)
# view_tree_channel(test_cell.root)