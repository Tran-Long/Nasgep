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
        if T_G != -1:
            adf_id_to_remove = [adf_id for (adf_id, adf) in self.adfs_dict.items() if adf.fitness != -1 and adf.is_used == 0 and adf.fitness < T_G]
            for adf_id in adf_id_to_remove:
                self.remove_adf(adf_id)

    def add_adf(self, adf_genotype):
        adf = ADF(for_reduction = self.for_reduction, reproduction_genotype = adf_genotype)
        adf_key = ADF_PREFIX + str(self.nonce)
        self.nonce += 1
        self.adfs_dict[adf_key] = adf
        self.keys_list.append(adf_key)
        self.child_population.append(adf)
        self.child_pop_size += 1

    def remove_adf(self, adf_key):
        adf_index_in_list = self.keys_list.index(adf_key)
        self.adfs_dict.pop(adf_key)
        self.keys_list.pop(adf_index_in_list)
        self.population.pop(adf_index_in_list)
        self.pop_size -= 1

    def reproduction(self):
        num_of_new_adf = min(self.max_size - self.pop_size, MAX_CHILD_ADF)
        assert num_of_new_adf >= MIN_CHILD_ADF, "Must create at least min = " + str(MIN_CHILD_ADF) + " child"
        num_of_new_adf = np.random.randint(MIN_CHILD_ADF, num_of_new_adf)
        while self.child_pop_size < num_of_new_adf:
            new_adf_genotype_1, new_adf_genotype_2 = self.reproduction_individual_genotype()
            self.add_adf(new_adf_genotype_1)
            self.add_adf(new_adf_genotype_2)
        # merge and clear child_pop
        self.population = self.population + self.child_population
        self.pop_size += self.child_pop_size
        self.child_population = []
        self.child_pop_size = 0


# from Utils import *
# from Cell import *
# test_adf_pop = ADFPopulation(1, 2, pop_size = 10)
# test_cell = Cell(4, 5, adf_population = test_adf_pop)
# view_tree(test_cell.root)
# view_tree_channel(test_cell.root)