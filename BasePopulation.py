import numpy as np
import copy
class BasePopulation():
    def __init__ (self, head_size, tail_size, pop_size):
        self.nonce = 0
        self.head_size = head_size
        self.tail_size = tail_size
        self.length = head_size + tail_size
        self.pop_size = pop_size
        self.population = []
        self.child_population = []

    def tournament_selection (self, k = 8):
        # select random k indices, represent objects position in population
        list_object_indices = np.array([], dtype = int)
        while len(list_object_indices) < k:
            index = np.random.randint(self.pop_size)
            if index not in list_object_indices:
                list_object_indices = np.append(list_object_indices, index)

        # fighting till 2 parents left
        np.random.shuffle(list_object_indices)
        while len(list_object_indices) > 2:
            competitor_cnt = len(list_object_indices)
            winning = np.ones(competitor_cnt)
            i = 0
            while i < len(list_object_indices):
                if i + 1 < competitor_cnt:
                    index1 = list_object_indices[i]
                    index2 = list_object_indices[i + 1]
                    if self.population[index1].fitness >= self.population[index2].fitness:
                        winning[i + 1] = 0
                    else:
                        winning[i] = 0
                i += 2
            list_object_indices = list_object_indices[winning.astype(bool)]

        return self.population[list_object_indices[0]], self.population[list_object_indices[1]]

    def replication(self, obj):
        return copy.deepcopy(obj)

    def mutation (self, obj, rate = 0.05):  # gene is an object - i.e Cell, ADF
        if np.random.rand() <= rate:
            terminal_set = self.terminal_set
            function_set = self.function_set
            mutation_pos = np.random.randint(self.length)
            if mutation_pos < self.head_size:
                # head_set = terminal_set + function_set
                head_set = function_set
                obj.genotype[mutation_pos] = np.random.choice(head_set)
            else:
                tail_set = terminal_set
                obj.genotype[mutation_pos] = np.random.choice(tail_set)

    def transposition (self, obj, rate = 0.1, is_elements_cnt = 3):
        if np.random.rand() <= rate:
            if self.head_size > 1:
                # select start of transposon
                is_start = np.random.randint(self.length)
                # select where to insert
                target_site = np.random.randint(1, self.head_size)
                # select length of transposon
                max_valid_size = min(self.head_size - target_site, self.length - is_start)
                if max_valid_size < is_elements_cnt:
                    is_lengths = [i for i in range(1, max_valid_size + 1)]
                else:
                    is_lengths = []
                    while len(is_lengths) < is_elements_cnt:
                        random_length = np.random.randint(max_valid_size) + 1
                        if random_length not in is_lengths:
                            is_lengths.append(random_length)
                transposon_length = np.random.choice(is_lengths)
                # insert 
                transposon = obj.genotype[is_start:is_start + transposon_length].copy()
                for item in reversed(transposon):
                    obj.genotype.insert(target_site, item)
                for i in range(transposon_length):
                    obj.genotype.pop(self.head_size)

    def rootTransposition (self, obj, rate = 0.1, ris_elements_cnt = 3):
        if np.random.rand() <= rate:
            # find function / start of the transposon
            ris_start = None
            random_start = np.random.randint(self.head_size)
            while random_start < self.head_size:
                if self.genotype[random_start] in self.function_set:
                    ris_start = random_start
                    break
                random_start += 1
            # if find an fuction in the head
            if ris_start is not None:
                # select length on the transposon
                ris_lengths = []
                max_valid_size = self.length - ris_start
                if max_valid_size < ris_elements_cnt:
                    ris_lengths = [i for i in range(1, max_valid_size + 1)]
                else:
                    while len(ris_lengths) < ris_elements_cnt:
                        random_length = np.random.randint(max_valid_size) + 1
                        if random_length not in ris_lengths:
                            ris_lengths.append(random_length)
                transposon_length = np.random.choice(ris_lengths)
                # insert to the head
                ris_element = obj.genotype[ris_start:ris_start + transposon_length].copy()
                for item in reversed(ris_element):
                    obj.genotype.insert(0, item)
                for i in range(transposon_length):
                    obj.genotype.pop(self.head_size)

    def onePointRecombination (self, obj_dad, obj_mom, rate = 0.5):
        if np.random.rand() <= rate:
            random_point = np.random.randint(self.length) + 1
            child1_genotype = copy.deepcopy(obj_dad.genotype)
            child2_genotype = copy.deepcopy(obj_mom.genotype)
            for i in range(random_point):
                child1_genotype[i] = obj_dad.genotype[i]
                child2_genotype[i] = obj_mom.genotype[i]
            for i in range(random_point, self.length):
                child2_genotype[i] = obj_dad.genotype[i]
                child1_genotype[i] = obj_mom.genotype[i]

            obj_dad.genotype = child1_genotype
            obj_mom.genotype = child2_genotype
        return obj_dad, obj_mom

    def twoPointRecombination (self, obj_dad, obj_mom, rate = 0.2):
        if np.random.rand() <= rate:
            random1 = np.random.randint(self.length) + 1
            random2 = np.random.randint(self.length) + 1
            while random1 == random2:
                random2 = np.random.randint(self.length) + 1
            random_point1 = min(random1, random2)
            random_point2 = max(random1, random2)

            child1_genotype = copy.deepcopy(obj_dad.genotype)
            child2_genotype = copy.deepcopy(obj_mom.genotype)
            for i in range(random_point1):
                child1_genotype[i] = obj_dad.genotype[i]
                child2_genotype[i] = obj_mom.genotype[i]
            for i in range(random_point1, random_point2):
                child2_genotype[i] = obj_dad.genotype[i]
                child1_genotype[i] = obj_mom.genotype[i]
            for i in range(random_point2, self.length):
                child1_genotype[i] = obj_dad.genotype[i]
                child2_genotype[i] = obj_mom.genotype[i]

            obj_dad.genotype = child1_genotype
            obj_mom.genotype = child2_genotype
        return obj_dad, obj_mom

