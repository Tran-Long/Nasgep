import numpy as np
import copy
from Utilities.Configs import *

class BasePopulation:
    def __init__ (self, head_size, tail_size, pop_size):
        self.nonce = 0
        self.head_size = head_size
        self.tail_size = tail_size
        self.length = head_size + tail_size
        self.pop_size = pop_size
        self.function_set = None
        self.terminal_set = None
        self.population = []
        self.child_population = []
        self.child_pop_size = 0

    def tournament_selection (self, k = TOURNAMENT_SELECTION_SIZE):
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
        obj1 = self.population[list_object_indices[0]]
        obj2 = self.population[list_object_indices[1]]
        if obj1.fitness >= obj2.fitness:
            return obj1.genotype, obj2.genotype
        return obj2.genotype, obj1.genotype

    def replication(self, obj):
        return copy.deepcopy(obj)

    def mutation (self, genotype, rate = RPD_MUTATION_RATE):  # genotype == array of elements
        if np.random.rand() <= rate:
            mutation_pos = np.random.randint(self.length)
            if mutation_pos < self.head_size:
                # head_set = terminal_set + function_set
                head_set = self.function_set
                genotype[mutation_pos] = np.random.choice(head_set)
            else:
                tail_set = self.terminal_set
                genotype[mutation_pos] = np.random.choice(tail_set)
        return genotype

    def transposition (self, genotype, rate = RPD_TRANSPOSITION_RATE, is_elements_cnt = RPD_IS_ELE_CNT):
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
                transposon = genotype[is_start:is_start + transposon_length].copy()
                for item in reversed(transposon):
                    genotype.insert(target_site, item)
                for i in range(transposon_length):
                    genotype.pop(self.head_size)
        return genotype

    def root_transposition (self, genotype, rate = RPD_ROOT_TRANSPOSITION_RATE, ris_elements_cnt = RPD_RIS_ELE_CNT):
        if np.random.rand() <= rate:
            # find function / start of the transposon
            ris_start = None
            random_start = np.random.randint(self.head_size)
            while random_start < self.head_size:
                if genotype[random_start] in self.function_set:
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
                ris_element = genotype[ris_start:ris_start + transposon_length].copy()
                for item in reversed(ris_element):
                    genotype.insert(0, item)
                for i in range(transposon_length):
                    genotype.pop(self.head_size)
        return genotype

    def one_point_recombination (self, genotype_dad, genotype_mom, rate = RPD_1_RECOMBINATION_RATE):
        if np.random.rand() <= rate:
            random_point = np.random.randint(self.length) + 1
            child1_genotype = copy.deepcopy(genotype_dad)
            child2_genotype = copy.deepcopy(genotype_mom)
            for i in range(random_point):
                child1_genotype[i] = genotype_dad[i]
                child2_genotype[i] = genotype_mom[i]
            for i in range(random_point, self.length):
                child2_genotype[i] = genotype_dad[i]
                child1_genotype[i] = genotype_mom[i]
            genotype_dad = child1_genotype
            genotype_mom = child2_genotype

        return genotype_dad, genotype_mom

    def two_point_recombination (self, genotype_dad, genotype_mom, rate = RPD_2_RECOMBINATION_RATE):
        if np.random.rand() <= rate:
            random1 = np.random.randint(self.length) + 1
            random2 = np.random.randint(self.length) + 1
            while random1 == random2:
                random2 = np.random.randint(self.length) + 1
            random_point1 = min(random1, random2)
            random_point2 = max(random1, random2)

            child1_genotype = copy.deepcopy(genotype_dad)
            child2_genotype = copy.deepcopy(genotype_mom)
            for i in range(random_point1):
                child1_genotype[i] = genotype_dad[i]
                child2_genotype[i] = genotype_mom[i]
            for i in range(random_point1, random_point2):
                child2_genotype[i] = genotype_dad[i]
                child1_genotype[i] = genotype_mom[i]
            for i in range(random_point2, self.length):
                child1_genotype[i] = genotype_dad[i]
                child2_genotype[i] = genotype_mom[i]
            genotype_dad = child1_genotype
            genotype_mom = child2_genotype

        return genotype_dad, genotype_mom

    def reproduction_individual_genotype(self):
        genotype_mom, genotype_dad = self.tournament_selection()
        genotype_mom = self.replication(genotype_mom)
        genotype_dad = self.replication(genotype_dad)
        genotype_mom = self.mutation(genotype_mom)
        genotype_dad = self.mutation(genotype_dad)
        if ENABLE_TRANSPOSITION:
            genotype_mom = self.transposition(genotype_mom)
            genotype_dad = self.transposition(genotype_dad)
            genotype_mom = self.root_transposition(genotype_mom)
            genotype_dad = self.root_transposition(genotype_dad)
        genotype_mom, genotype_dad = self.one_point_recombination(genotype_dad, genotype_mom)
        genotype_mom, genotype_dad = self.two_point_recombination(genotype_dad, genotype_mom)
        return genotype_mom, genotype_dad
