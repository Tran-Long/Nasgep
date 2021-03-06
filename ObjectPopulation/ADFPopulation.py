from ObjectPopulation.ADF import *
from Base.BasePopulation import *
from Utilities.Configs import *

class ADFPopulation(BasePopulation):
    def __init__(self, head_size, tail_size, for_reduction = True, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP, extra = False, save_path = None):
        super(ADFPopulation, self).__init__(head_size, tail_size, pop_size)
        self.for_reduction = for_reduction
        self.adfs_dict = {}
        self.max_size = max_size
        self.extra = extra
        self.existed_genotype = []
        self.save_path = save_path
        if save_path is not None and check_file_exist(save_path):
            save_dict = load_dict_checkpoint(save_path)
            for adf_id in save_dict:
                if adf_id == "nonce":
                    self.nonce = save_dict[adf_id]
                    continue
                self.adfs_dict[adf_id] = ADF(head_size, tail_size, for_reduction, reproduction_genotype = save_dict[adf_id]["genotype"])
                self.adfs_dict[adf_id].is_used = save_dict[adf_id]["is_used"]
                self.adfs_dict[adf_id].fitness = save_dict[adf_id]["fitness"]
        else:
            while len(self.adfs_dict) < pop_size:
                new_adf = ADF(head_size, tail_size, for_reduction)
                if extra and new_adf.genotype in self.existed_genotype:
                    continue
                adf_id = ADF_PREFIX + str(self.nonce)
                self.nonce += 1
                self.adfs_dict[adf_id] = new_adf
                if extra:
                    self.existed_genotype.append(new_adf.genotype)
        self.keys_list = list(self.adfs_dict.keys())
        self.pop_size = len(self.keys_list)
        self.population = list(self.adfs_dict.values())
        self.function_set = ADF_FUNCTION
        self.terminal_set = ADF_TERMINAL

    def kill_bad_genes(self, t_g):
        if t_g != -1:
            # print("\tAdf pop consist: ")
            # write_log("Adf pop consist: ")
            # print("\t\t", end = "")
            # print({adf_id: (adf.fitness, adf.is_used) for (adf_id, adf) in self.adfs_dict.items()})
            # write_log(self.get_info_string())
            adf_id_to_remove = [adf_id for (adf_id, adf) in self.adfs_dict.items() if adf.fitness != -1 and adf.is_used == 0 and adf.fitness < t_g]
            for adf_id in adf_id_to_remove:
                self.remove_adf(adf_id)
            # print("\tAdf left: ")
            write_log("Adf left: ")
            # print("\t\t", end = "")
            # print({adf_id: (adf.fitness, adf.is_used) for (adf_id, adf) in self.adfs_dict.items()})
            write_log(self.get_info_string())

    def add_adf(self, adf_genotype):
        adf = ADF(for_reduction = self.for_reduction, reproduction_genotype = adf_genotype)
        if self.extra and adf.genotype in self.existed_genotype:
            return
        adf_key = ADF_PREFIX + str(self.nonce)
        self.nonce += 1
        self.adfs_dict[adf_key] = adf
        self.keys_list.append(adf_key)
        self.child_population.append(adf)
        self.child_pop_size += 1
        if self.extra:
            self.existed_genotype.append(adf.genotype)

    def remove_adf(self, adf_key):
        adf_index_in_list = self.keys_list.index(adf_key)
        self.adfs_dict.pop(adf_key)
        self.keys_list.pop(adf_index_in_list)
        self.population.pop(adf_index_in_list)
        self.pop_size -= 1

    def reproduction(self):
        num_of_new_adf = min(self.max_size - self.pop_size, MAX_CHILD_ADF)
        if num_of_new_adf < MIN_CHILD_ADF:
            # print("\tCreate no new adf")
            write_log("Create no new adf")
            return
        if num_of_new_adf > MIN_CHILD_ADF:
            num_of_new_adf = np.random.randint(MIN_CHILD_ADF, num_of_new_adf)
        # print("\tCreate " + str(num_of_new_adf) + " new adf")
        write_log("Create " + str(num_of_new_adf) + " new adf")
        while self.child_pop_size < num_of_new_adf:
            new_adf_genotype_1, new_adf_genotype_2 = self.reproduction_individual_genotype()
            self.add_adf(new_adf_genotype_1)
            self.add_adf(new_adf_genotype_2)
        # merge and clear child_pop
        self.population = self.population + self.child_population
        self.pop_size += self.child_pop_size
        self.child_population = []
        self.child_pop_size = 0

    def get_info_string(self):
        # string = "{"
        # for (adf_id, adf) in self.adfs_dict.items():
        #     string += adf_id + ": (" + str(adf.fitness) + ", " + str(adf.is_used) + "; "
        # string += "}"
        string = str({adf_id: (adf.fitness, adf.is_used) for (adf_id, adf) in self.adfs_dict.items()})
        return string

    def save_checkpoint(self):
        save_dict = {"nonce": self.nonce}
        for adf_id in self.adfs_dict:
            save_dict[adf_id] = {}
            save_dict[adf_id]["genotype"] = self.adfs_dict[adf_id].genotype
            save_dict[adf_id]["is_used"] = self.adfs_dict[adf_id].is_used
            save_dict[adf_id]["fitness"] = self.adfs_dict[adf_id].fitness
        save_dict_checkpoint(save_dict, self.save_path)
