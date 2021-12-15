from Model import *
from BasePopulation import *

class ModelPopulation:
    def __init__(self, pop_size, n, n_adf_population, r_cell_population, for_dataset = "cifar-10"):
        self.nonce = pop_size
        self.n = n
        self.adf_population = n_adf_population
        self.cell_population = r_cell_population
        self.models_dict = {}
        for i in range(pop_size):
            model_id = MODEL_PREFIX + str(i)
            self.models_dict[model_id] = Model(n_adf_population, r_cell_population, n, for_dataset = for_dataset)
        self.population = list(self.models_dict.values())
        self.child_models_dict = {}

    def test_population(self, test_loader, pop):
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

            model.fitness = max(model.fitness, correct/total * 100)
            print((model_id + " fitness = %.2f %%") % model.fitness)
            print("----------------------")

    def train_population(self, train_loader, test_loader, pop):
        for (model_id, model) in pop.items():
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
            print("*********ACCURACY***********")
            self.test_population(train_loader, pop)
            print("****************************")
            self.test_population(test_loader, pop)

    def evaluate_child_population(self, train_loader):
        pass