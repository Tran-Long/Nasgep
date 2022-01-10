from ObjectPopulation.Model import *

# This population is kinda different from 2 previous populations
class ModelPopulation:
    def __init__(self, pop_size, n, n_adf_population, r_cell_population, for_dataset = "cifar-10", save_path = None):
        self.nonce = 0
        self.pop_size = pop_size
        self.n = n
        self.n_adf_population = n_adf_population
        self.r_cell_population = r_cell_population
        self.for_dataset = for_dataset
        self.models_dict = {}
        self.save_path = save_path
        if self.save_path is not None and check_file_exist(self.save_path):
            save_dict = load_dict_checkpoint(save_path)
            for model_id in save_dict:
                if model_id == "nonce":
                    self.nonce = save_dict["nonce"]
                    continue
                model_info = save_dict[model_id]
                t_n_cell = Cell(n_adf_population, reproduction_genotype = model_info["normal_cell"], from_save_path = True)
                t_n_cell.mark_killed = model_info["n_cell_mark_killed"]
                t_n_cell.fitness = model_info["n_cell_fitness"]
                t_r_cell = r_cell_population.cells_dict[model_info["reduction_cell"]]
                model = Model(n_adf_population, r_cell_population, n,
                              normal_cell = t_n_cell, reduction_cell = t_r_cell,
                              best_cell_genotypes = model_info["model_genotype"])
                model.reduction_cell_id = model_info["reduction_cell"]
                model.weight_path = model_info["weight_path"]
                model.load_checkpoint()
                # model.to(DEVICE)
                self.models_dict[model_id] = model
                print("=> Load " + model_id + " successfully")

        else:
            while len(self.models_dict) < pop_size:
                new_model = Model(n_adf_population, r_cell_population, n, for_dataset = for_dataset)
                while new_model.num_params > MAX_MODEL_PARAMS:
                    reduction_cell = new_model.reduction_cell
                    normal_cell = Cell(n_adf_population)
                    new_model = Model(n_adf_population, r_cell_population, normal_cell = normal_cell, reduction_cell = reduction_cell)
                model_id = MODEL_PREFIX + str(self.nonce)
                new_model.weight_path = make_path(model_id+"_weight.pth")
                self.nonce += 1
                """"""
                view_model_info(model_id, new_model)
                """"""
                self.models_dict[model_id] = new_model
                # self.models_dict[model_id].to(DEVICE)
        self.population = list(self.models_dict.values())
        self.child_models_dict = {}
        self.child_pop_size = 0
        # self.child_population = []

    def reproduction(self):
        # print("\tBefore:")
        # write_log("Before: ")
        # print("\t\t", end="")
        # print(self.models_dict.keys())
        # write_log(self.get_info_string(0))
        for (model_id, model) in self.models_dict.items():
            if not model.mark_killed:
                self.add_model(model.normal_cell)
        # print("\tAfter:")
        # write_log("After: ")
        # print("\t\t", end="")
        # print(self.models_dict.keys(), end = " ")
        # write_log(self.get_info_string(0))
        # print(self.child_models_dict.keys())
        # write_log(self.get_info_string(1))

    def get_info_string(self, mode):
        if mode == 0:
            return get_string_fr_arr(list(self.models_dict.keys()))
        else:
            return get_string_fr_arr(list(self.child_models_dict.keys()))

    """
        Used for add model to child population, input 
    """
    def add_model(self, old_normal_cell):
        reduction_cell_id, reduction_cell = self.r_cell_population.select_random_reduction_cell()
        normal_cell = old_normal_cell.reproduction()
        new_model = Model(self.n_adf_population, self.r_cell_population, self.n, normal_cell, reduction_cell, self.for_dataset)
        new_model.reduction_cell_id = reduction_cell_id
        while new_model.num_params > MAX_MODEL_PARAMS:
            normal_cell = Cell(self.n_adf_population)
            new_model = Model(self.n_adf_population, self.r_cell_population, normal_cell = normal_cell,  reduction_cell = reduction_cell)
        model_id = MODEL_PREFIX + str(self.nonce)
        new_model.weight_path = make_path(model_id + "_weight.pth")
        """"""
        view_model_info(model_id, new_model)
        """"""
        self.nonce += 1
        # new_model.to(DEVICE)
        self.child_models_dict[model_id] = new_model
        # self.child_population.append(new_model)
        self.child_pop_size += 1

    def remove_model(self, model_id):
        self.models_dict[model_id].mark_to_be_killed()
        self.models_dict.pop(model_id)
        self.pop_size -= 1

    def merge_dict(self):
        self.models_dict = {**self.models_dict, **self.child_models_dict}
        self.pop_size += self.child_pop_size
        self.child_models_dict.clear()
        self.child_pop_size = 0

    @staticmethod
    def test_model(val_loader, model_id, model):
        model.training_status = False
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # inputs, labels = data[0], data[1]
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.set_fitness(correct / total * 100)
        # print((model_id + " fitness = %.2f %%") % model.fitness)
        write_log((model_id + " fitness = %.2f %%") % model.fitness)

    def train_population(self, train_loader, val_loader, pop):
        for (model_id, model) in pop.items():
            if not model.mark_killed:
                model.epoch_cnt += 1
                if model.epoch_cnt == EPOCH_MAX:
                    model.mark_to_be_killed()
                # print("-----------------------------")
                write_log("-----------------------------")
                # print("\t\tTraining " + model_id + ".....", end = " ")
                write_log("Training " + model_id + ".....")
                model.training_status = True
                model.train()
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                    # inputs, labels = data[0], data[1]
                    model.zero_grad()
                    outputs = model(inputs)
                    loss = model.criterion(outputs, labels)

                    loss.backward()
                    model.optimizer.step()
                model.scheduler.step()
                # print("Training " + model_id + " finished")
                # write_log("Training " + model_id + " finished")
                # print("\t\tACCURACY: ", end = " ")
                # self.test_model(train_loader, model_id, model)
                # print("\t\tVALIDATION: ", end = " ")
                # write_log("VALIDATION: ")
                self.test_model(val_loader, model_id, model)
                # print("-------------------------------")
                write_log("-------------------------------")

    def evaluate_population_step_6(self, train_loader, test_loader, pop, t_c):
        self.train_population(train_loader, test_loader, pop)
        if t_c != -1:
            # print("\t\t Base score for another epoch = " + str(t_c))
            write_log("Base score for another epoch = " + str(t_c))
            extra_models = {model_id: model for (model_id, model) in pop.items() if model.fitness >= t_c}
            self.train_population(train_loader, test_loader, extra_models)

    def increase_age(self):
        for (model_id, model) in self.models_dict.items():
            model.age += 1

    def survivor_selection(self):
        self.merge_dict()
        # print("\tAll model: ")
        write_log("All model: ")
        # print("\t\t", end = "")
        # print({model_id: (model.fitness, model.age, model.mark_killed) for (model_id, model) in self.models_dict.items()})
        write_log(self.get_pop_info_string())

        all_id = list(self.models_dict.keys())
        model_id_to_preserve = []
        model_id_to_remove = []

        """Keep best fitness"""
        max_fitness = max([self.models_dict[model_id].fitness for model_id in all_id])
        max_fitness_model_id = [model_id for model_id in all_id if self.models_dict[model_id].fitness == max_fitness]
        model_id_to_preserve.extend(max_fitness_model_id)
        all_id = np.setdiff1d(all_id, max_fitness_model_id)

        if len(model_id_to_preserve) + len(all_id) > INIT_SIZE_MODEL_POP:
            """Remove oldest with lowest fitness"""
            age_list = [self.models_dict[model_id].age for model_id in all_id]
            max_age = max(age_list)
            oldest_model_id = [model_id for model_id in all_id if self.models_dict[model_id].age == max_age]
            min_fitness_old = max([self.models_dict[model_id].fitness for model_id in oldest_model_id])
            min_fitness_old_model_id = [model_id for model_id in oldest_model_id if self.models_dict[model_id].fitness == min_fitness_old]
            model_id_to_remove.extend(min_fitness_old_model_id)
            all_id = np.setdiff1d(all_id, min_fitness_old_model_id)

            """Remove mark killed"""
            mark_killed_model_id = [model_id for model_id in all_id if self.models_dict[model_id].mark_killed == True]
            model_id_to_remove.extend(mark_killed_model_id)
            all_id = np.setdiff1d(all_id, mark_killed_model_id)

        """Keeping fitness in descending order"""
        diff = len(model_id_to_preserve) + len(all_id) - INIT_SIZE_MODEL_POP
        if diff > 0:
            num_to_preserve = INIT_SIZE_MODEL_POP - len(model_id_to_preserve)
        else:
            num_to_preserve = len(all_id)
        temp_dict = {model_id: self.models_dict[model_id] for model_id in all_id}
        addition_model_id_preserve = list(dict(sorted(temp_dict.items(), key = lambda x: x[1].fitness, reverse = True)[:num_to_preserve]))
        model_id_to_preserve.extend(addition_model_id_preserve)
        all_id = np.setdiff1d(all_id, addition_model_id_preserve)

        model_id_to_remove.extend(all_id)
        assert len(set(model_id_to_preserve)) + len(set(model_id_to_remove)) == self.pop_size, "Wrong survivor"
        # print("\tModel to preserve: ")
        # write_log("Model to preserve: ")
        # print("\t\t", end = "")
        # print(model_id_to_preserve)
        # write_log(get_string_fr_arr(model_id_to_preserve))

        for model_id in model_id_to_remove:
            self.remove_model(model_id)

        # print("\tModels left: ")
        write_log("Models left: ")
        # print("\t\t", end = "")
        # print(self.models_dict.keys())
        write_log(self.get_info_string(0))

    def get_pop_info_string(self):
        string = "{"
        for (model_id, model) in self.models_dict.items():
            string += model_id + ': (' + str(model.fitness) + ", " + str(model.age) + ", " + str(model.mark_killed) + '), '
        string += "}"
        return string

    def get_best_models(self):
        max_fitness = max([self.models_dict[model_id].fitness for model_id in list(self.models_dict.keys())])
        max_fitness_model_id = [model_id for model_id in list(self.models_dict.keys()) if self.models_dict[model_id].fitness == max_fitness]
        print("There are " + str(len(max_fitness_model_id)) + " best models")
        write_log("There are " + str(len(max_fitness_model_id)) + " best models")
        best_models = []
        for model_id in max_fitness_model_id:
            best_model = self.models_dict[model_id]
            print("\n")
            best_model.show_info()
            best_models.append(best_model.get_info_to_save())
        return best_models

    def save_checkpoint(self):
        save_dict = {"nonce": self.nonce}
        for model_id in self.models_dict:
            model = self.models_dict[model_id]
            model_dict = {"model_genotype": model.get_info_to_save(),
                          "reduction_cell": model.reduction_cell_id,
                          "normal_cell": model.normal_cell.genotype,
                          "n_cell_mark_killed": model.normal_cell.mark_killed,
                          "n_cell_fitness": model.normal_cell.fitness,
                          "weight_path": model.weight_path,
                          "fitness": model.fitness,
                          "mark_killed": model.mark_killed,
                          "epoch_cnt": model.epoch_cnt,
                          "age": model.age}
            model.save_checkpoint()
            save_dict[model_id] = model_dict
        save_dict_checkpoint(save_dict, self.save_path)
