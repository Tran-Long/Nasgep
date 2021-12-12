if pop_type == "ADF":
    self.adfs_dict = {}
    self.genotypes_dict = {}
    for i in range(pop_size):
        adf_id = ADF_FREFIX + str(i)
        self.adfs_dict[adf_id] = ADF()
        self.genotypes_dict[adf_id] = self.adfs_dict[adf_id].genotype
    self.keys_list = list(self.genotypes_dict.keys())
    self.function_set = ADF_FUNCTION
    self.terminal_set = ADF_TERMINAL
elif pop_type == "CELL":
    self.cells_dict = {}
    self.cells_usage_dict = {}
    for i in range(pop_size):
        cell_id = CELL_PREFIX + str(i)
        self.cells_dict[cell_id] = Cell(head_size, tail_size, adf_population)
        self.cells_usage_dict[cell_id] = 0
    self.function_set = CELL_FUNCTION
    self.terminal_set = CELL_TERMINAL
elif pop_type == "MODEL":
    self.population = [Model() for i in range(self.pop_size)]

    def killBadGene (self):
        assert self.pop_type == "ADF"
        t_population = []
        for obj in self.population:
            if obj.mark_killed == False:
                t_population.append(obj)
        self.population = t_population

    def survivorSelection (self):
        assert self.pop_type == "CELL" or self.pop_type == "MODEL"
        if self.pop_type == "MODEL":
            max_age = max([obj.age for obj in self.population])
            list_indices_max_age = []
            for i, obj in enumerate(self.population):
                if obj.age == max_age:
                    list_indices_max_age.append(i)
            random_oldest_index = np.random.choice(list_indices_max_age)
            self.population[random_oldest_index].mark_killed = True

        new_generation = []
        for obj in self.population:
            if obj.mark_killed == False:
                new_generation.append(obj)
        self.population = new_generation
        self.population.sort(key = lambda x: x.fitness, reverse = True)
        # need fixing