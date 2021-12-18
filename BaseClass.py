import numpy as np
import copy
from Configs import *

class BaseClass:
    def __init__(self, head, tail):
        self.head_size = head
        self.tail_size = tail
        self.length = head + tail
        self.genotype = None
        self.function_set = None
        self.terminal_set = None

    @staticmethod
    def replication(obj):
        return copy.deepcopy(obj)

    def mutation (self, genotype, rate = RPD_CELL_MUTATION_RATE):  # genotype == array of elements
        if np.random.rand() <= rate:
            mutation_pos = np.random.randint(self.length)
            if mutation_pos < self.head_size:
                genotype[mutation_pos] = np.random.choice(self.function_set)
            else:
                genotype[mutation_pos] = np.random.choice(self.terminal_set)
        return genotype

    def transposition (self, genotype, rate = RPD_CELL_TRANSPOSITION_RATE, is_elements_cnt = RPD_IS_ELE_CNT):
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

    def root_transposition (self, genotype, rate = RPD_CELL_ROOT_TRANSPOSITION_RATE, ris_elements_cnt = RPD_RIS_ELE_CNT):
        if np.random.rand() <= rate:
            # find function / start of the transposon
            ris_start = None
            random_start = np.random.randint(self.head_size)
            while random_start < self.head_size:
                if self.genotype[random_start] in self.function_set:
                    ris_start = random_start
                    break
                random_start += 1
            # if find an function in the head
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

    def reproduction_alone(self):
        genotype = self.replication(self.genotype)
        genotype = self.mutation(genotype)
        genotype = self.transposition(genotype)
        genotype = self.root_transposition(genotype)
        return genotype
