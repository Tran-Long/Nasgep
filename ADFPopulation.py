from ADF import *
from BasePopulation import *
from Configs import *

class ADFPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, for_reduction = True, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP):
        super(ADFPopulation, self).__init__(head_size, tail_size, pop_size)
        self.for_reduction = for_reduction
        self.adfs_dict = {}
        self.max_size = max_size
        for i in range(self.pop_size):
            adf_id = ADF_PREFIX + str(i)
            self.adfs_dict[adf_id] = ADF(1, 2, for_reduction)
        self.keys_list = list(self.adfs_dict.keys())
        self.population = list(self.adfs_dict.values())
        self.function_set = ADF_FUNCTION
        self.terminal_set = ADF_TERMINAL

    def kill_bad_genes(self):
        pass

    def add_adf(self, adf_genotype):
        adf = ADF(for_reduction = self.for_reduction, reproduction_genotype = adf_genotype)
        adf_key = ADF_PREFIX + str(self.nonce)
        self.nonce += 1
        self.adfs_dict[adf_key] = adf
        self.keys_list.append(adf_key)
        self.population.append(adf)

    def remove_adf(self, adf_key):
        adf_index_in_list = self.keys_list.index(adf_key)
        self.adfs_dict.pop(adf_key)
        self.keys_list.pop(adf_index_in_list)
        self.population.pop(adf_index_in_list)

    def reproduction(self, min_child, max_child):
        num_of_new_adf = min(self.max_size - self.pop_size, MAX_CHILD_ADF)
        assert num_of_new_adf >= MIN_CHILD_ADF, "Must create at least min = " + str(MIN_CHILD_ADF) + " child"
        # for i in range(num_of_new_adf):
        pass



# from Utils import *
# from Cell import *
# test_adf_pop = ADFPopulation(1, 2, pop_size = 10)
# test_cell = Cell(4, 5, adf_population = test_adf_pop)
# view_tree(test_cell.root)
# view_tree_channel(test_cell.root)