from ObjectPopulation.ADFPopulation import *
from Base.BaseClass import *

class Cell(BaseClass):
    def __init__(self, adf_population, cell_head=CELL_HEAD_LEN, cell_tail=CELL_TAIL_LEN, reproduction_genotype=None, from_save_path = False):
        super(Cell, self).__init__(cell_head, cell_tail)
        self.adf_population = adf_population

        if not adf_population.for_reduction:  # Only normal cell reproduction alone by itself
            self.function_set = CELL_FUNCTION
            self.terminal_set = adf_population.keys_list

        if reproduction_genotype is not None:
            self.genotype = reproduction_genotype
            self.adfs_dict = {}
            for array_element in self.genotype:
                if ADF_PREFIX in array_element:
                    self.adfs_dict[array_element] = adf_population.adfs_dict[array_element]
        else:
            self.genotype, self.adfs_dict = self.init_data(cell_head, cell_tail, adf_population)

        if not from_save_path:
            for adf_id in self.adfs_dict:
                self.adfs_dict[adf_id].is_used += 1

        self.root = build_tree_cell(self.genotype, self.adfs_dict)
        self.mark_killed = False
        self.is_used = False
        self.fitness = -1

    @staticmethod
    def init_data (cell_head, cell_tail, adf_population):
        adf_population_terms = adf_population.keys_list
        genotype = []
        adfs_dict = {}
        for i in range(cell_head):
            genotype.append(np.random.choice(CELL_FUNCTION))
        for i in range(cell_tail):
            adf_genotype_key = np.random.choice(adf_population_terms)
            genotype.append(adf_genotype_key)
            adfs_dict[adf_genotype_key] = adf_population.adfs_dict[adf_genotype_key]
        return genotype, adfs_dict

    def create_modules_dict(self, prev_outputs, base_channel):
        value_dict, file_path_dict, _ = self.create_dict(self.root, {}, {}, prev_outputs, base_channel, nonce = 0)
        module_dict = nn.ModuleDict(value_dict)
        return module_dict, file_path_dict

    def create_dict(self, root, value_dict, file_path_dict, prev_outputs, base_channel, nonce):
        nonce = nonce + 1
        node_key = str(root.value) + str(nonce)
        if root.value == POINT_WISE_TERM:
            value_dict[node_key] = conv_block(POINT_WISE_TERM, base_channel*root.left.channel, base_channel)
            file_path_dict[node_key] = create_file_name_conv(root.value, base_channel*root.left.channel, base_channel)
            value_dict, file_path_dict, nonce = self.create_dict(root.left, value_dict, file_path_dict, prev_outputs, base_channel, nonce)
        elif root.value == POINT_WISE_BEFORE_REDUCTION_TERM:
            value_dict[node_key] = conv_block(POINT_WISE_BEFORE_REDUCTION_TERM, int(base_channel/2), base_channel)
            file_path_dict[node_key] = create_file_name_conv(root.value, int(base_channel/2), base_channel)
            value_dict, file_path_dict, nonce = self.create_dict(root.left, value_dict, file_path_dict, prev_outputs, base_channel, nonce)
        elif root.value in CONV_TERMS:
            value_dict[node_key] = conv_block(root.value, base_channel, base_channel)
            file_path_dict[node_key] = create_file_name_conv(root.value, base_channel, base_channel)
            value_dict, file_path_dict, nonce = self.create_dict(root.left, value_dict, file_path_dict, prev_outputs, base_channel, nonce)
        elif root.value == PREV_OUTPUT:
            root.value = np.random.choice(range(len(prev_outputs)))
        elif root.value == "sum" or root.value == "cat":
            value_dict_left, file_path_dict_left, nonce = self.create_dict(root.left, value_dict, file_path_dict, prev_outputs, base_channel, nonce)
            value_dict_right, file_path_dict_right, nonce = self.create_dict(root.right, value_dict, file_path_dict, prev_outputs, base_channel, nonce)
            value_dict = {**value_dict_left, **value_dict_right}
            file_path_dict = {**file_path_dict_left, **file_path_dict_right}
        return value_dict, file_path_dict, nonce

    def save_weight(self):
        pass

    def set_fitness(self, fitness):
        self.fitness = fitness
        for adf_id in self.adfs_dict:
            self.adfs_dict[adf_id].set_fitness(fitness)

    def mark_to_be_killed(self):
        self.mark_killed = True
        for adf_id in self.adfs_dict:
            self.adfs_dict[adf_id].is_used -= 1

    def reproduction(self):
        new_genotype = self.reproduction_alone()
        return Cell(self.adf_population, self.head_size, self.tail_size, new_genotype)

