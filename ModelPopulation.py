from Model import *

# This population is kinda different from 2 previous populations
class ModelPopulation:
    def __init__(self, pop_size, n, n_adf_population, r_cell_population, for_dataset = "cifar-10"):
        self.nonce = pop_size
        self.pop_size = pop_size
        self.n = n
        self.n_adf_population = n_adf_population
        self.r_cell_population = r_cell_population
        self.for_dataset = for_dataset
        self.models_dict = {}
        for i in range(pop_size):
            model_id = MODEL_PREFIX + str(i)
            self.models_dict[model_id] = Model(n_adf_population, r_cell_population, n, for_dataset = for_dataset)
        self.population = list(self.models_dict.values())
        self.child_models_dict = {}
        self.child_pop_size = 0
        # self.child_population = []

    def reproduction(self):
        for (model_id, model) in self.models_dict.items():
            self.add_model(model.normal_cell)

    """
        Used for add model to child population, input 
    """
    def add_model(self, old_normal_cell):
        reduction_cell = self.r_cell_population.select_random_reduction_cell()
        normal_cell = old_normal_cell.reproduction()
        model_id = MODEL_PREFIX + str(self.nonce)
        self.nonce += 1
        new_model = Model(self.n_adf_population, self.r_cell_population, self.n, normal_cell, reduction_cell, self.for_dataset)
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
    def test_population(test_loader, pop):
        for (model_id, model) in pop.items():
            print("----------------------")
            print("Testing " + model_id)
            total = 0
            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("Current fitness = %.2f" % (correct/total * 100))

            # model.fitness = max(model.fitness, correct/total * 100)
            # model.set_fitness(correct/total * 100)
            print((model_id + " fitness = %.2f %%") % model.fitness)
            print("----------------------")

    @staticmethod
    def test_model(test_loader, model_id, model):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.set_fitness(correct / total * 100)
        print((model_id + " fitness = %.2f %%") % model.fitness)

    def train_population(self, train_loader, test_loader, pop):
        for (model_id, model) in pop.items():
            if not model.mark_killed:
                model.epoch_cnt += 1
                if model.epoch_cnt == EPOCH_MAX:
                    model.mark_to_be_killed()
                print("----------------------")
                print("Training " + model_id + ".....")
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    model.zero_grad()
                    outputs = model(inputs)
                    loss = model.criterion(outputs, labels)

                    loss.backward()
                    model.optimizer.step()
                model.scheduler.step()
                print("Training " + model_id + " finished")
                print("----------------------")
                print("ACCURACY: ", end = " ")
                self.test_model(train_loader, model_id, model)
                print("VALIDATION: ", end = " ")
                self.test_model(test_loader, model_id, model)
                print("----------------------")

    def evaluate_population_step_6(self, train_loader, test_loader, pop):
        self.train_population(train_loader, test_loader, pop)
        extra_models = {model_id: model for (model_id, model) in pop.items() if model.fitness >= T_C}
        self.train_population(train_loader, test_loader, extra_models)

    def increase_age(self):
        for (model_id, model) in self.models_dict.items():
            model.age += 1

    def survivor_selection(self):
        self.merge_dict()

        all_id = list(self.models_dict.keys())
        model_id_to_preserve = []
        model_id_to_remove = []

        max_fitness = max([self.models_dict[model_id].fitness for model_id in all_id])
        max_fitness_model_id = [model_id for model_id in all_id if self.models_dict[model_id].fitness == max_fitness]
        model_id_to_preserve.extend(max_fitness_model_id)
        all_id = np.setdiff1d(all_id, max_fitness_model_id)

        if len(model_id_to_preserve) + len(all_id) > INIT_SIZE_MODEL_POP:
            age_list = [self.models_dict[model_id].age for model_id in all_id]
            max_age = max(age_list)
            oldest_model_id = [model_id for model_id in all_id if self.models_dict[model_id].age == max_age]
            model_id_to_remove.extend(oldest_model_id)
            all_id = np.setdiff1d(all_id, oldest_model_id)

            mark_killed_model_id = [model_id for model_id in all_id if self.models_dict[model_id].mark_killed == True]
            model_id_to_remove.extend(mark_killed_model_id)
            all_id = np.setdiff1d(all_id, mark_killed_model_id)

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

        for model_id in model_id_to_remove:
            self.remove_model(model_id)


