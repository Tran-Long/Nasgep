import torch
import torch.nn.functional as F
from Cell import *

class Model(nn.Module):
    def __init__(self, n_adf_population, r_cell_population, n = NUM_OF_CONSECUTIVE_NORMAL_CELL,
                 normal_cell=None, reduction_cell=None, for_dataset=DATASET):
        super(Model, self).__init__()
        self.adf_population = n_adf_population  # for normal cell making
        self.cell_population = r_cell_population  # for reduction cell making
        self.all_module_block_list = nn.ModuleList()
        self.all_cell_file_list = []  # for saving weight... contain dicts, each represent for a cell
        self.for_dataset = for_dataset
        self.n = n
        self.fitness = -1
        self.mark_killed = False
        self.epoch_cnt = 0
        self.age = 0
        # Select cells
        if normal_cell is None and reduction_cell is None:
            self.reduction_cell = r_cell_population.select_random_reduction_cell()
            self.normal_cell = Cell(4, 5, n_adf_population)
        else:
            self.normal_cell = normal_cell
            self.reduction_cell = reduction_cell
        """--------------------------------------------"""
        print("\t\t\t", end = "")
        print(self.normal_cell.genotype)
        print("\t\t\t", end = "")
        print(self.reduction_cell.genotype)
        """--------------------------------------------"""

        # Init network representation
        current_input_channel = 3
        prev_outputs = []
        if for_dataset == "cifar-10":
            self.n_cell_list = []
            self.r_cell_list = []
            self.all_module_block_list.append(nn.ModuleList([conv_block(STEM_TERM, current_input_channel, NUM_CHANNELS)]))
            current_input_channel = 16
            prev_outputs.append(0)

            # 1st normal block => channel = 16
            for i in range(n):
                self.n_cell_list.append(copy.deepcopy(self.normal_cell))
                module_dict, path_dict = self.n_cell_list[-1].create_modules_dict(prev_outputs, current_input_channel)
                self.all_module_block_list.append(module_dict)
                self.all_cell_file_list.append(path_dict)
                prev_outputs.append(0)

            for k in range(2):
                current_input_channel *= 2
                self.r_cell_list.append(copy.deepcopy(self.reduction_cell))
                module_dict, path_dict = self.r_cell_list[-1].create_modules_dict(prev_outputs, current_input_channel)
                self.all_module_block_list.append(module_dict)
                self.all_cell_file_list.append(path_dict)
                prev_outputs = [0]
                for i in range(n):
                    self.n_cell_list.append(copy.deepcopy(self.normal_cell))
                    module_dict, path_dict = self.n_cell_list[-1].create_modules_dict(prev_outputs, current_input_channel)
                    self.all_module_block_list.append(module_dict)
                    self.all_cell_file_list.append(path_dict)
                    prev_outputs.append(0)
            self.all_module_block_list.append(nn.BatchNorm2d(current_input_channel))
            self.all_module_block_list.append(nn.Linear(64, 10))
        # Init optimizer and loss
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 5)
        self.criterion = nn.CrossEntropyLoss()

    def cell_forward(self, root, prev_outputs, module_dict, nonce=0):
        nonce += 1
        node_key = str(root.value) + str(nonce)
        if root.value == POINT_WISE_TERM or root.value == POINT_WISE_BEFORE_REDUCTION_TERM or root.value in CONV_TERMS:
            output, nonce = self.cell_forward(root.left, prev_outputs, module_dict, nonce)
            output = module_dict[node_key](output)
        elif root.value == "sum":
            left_value, nonce = self.cell_forward(root.left, prev_outputs, module_dict, nonce)
            right_value, nonce = self.cell_forward(root.right, prev_outputs, module_dict, nonce)
            output = left_value + right_value
        elif root.value == "cat":
            left_value, nonce = self.cell_forward(root.left, prev_outputs, module_dict, nonce)
            right_value, nonce = self.cell_forward(root.right, prev_outputs, module_dict, nonce)
            output = torch.cat((left_value, right_value), dim = 1)
        else:
            output = prev_outputs[int(root.value)]
        return output, nonce

    def forward(self, x):
        if self.for_dataset == "cifar-10":
            blk_idx = 0
            n_cell_idx = 0
            r_cell_idx = 0
            output = x
            prev_outputs = []
            for layer in self.all_module_block_list[blk_idx]:
                output = layer(x)
                prev_outputs.append(output)
            for i in range(self.n):
                blk_idx += 1
                module_dict = self.all_module_block_list[blk_idx]
                output, _ = self.cell_forward(self.n_cell_list[n_cell_idx].root, prev_outputs, module_dict, nonce = 0)
                n_cell_idx += 1
                prev_outputs.append(output)

            for k in range(2):
                blk_idx += 1
                module_dict = self.all_module_block_list[blk_idx]
                output, _ = self.cell_forward(self.r_cell_list[r_cell_idx].root, prev_outputs, module_dict, nonce = 0)
                r_cell_idx += 1
                prev_outputs = [output]
                for i in range(self.n):
                    blk_idx += 1
                    module_dict = self.all_module_block_list[blk_idx]
                    output, _ = self.cell_forward(self.n_cell_list[n_cell_idx].root, prev_outputs, module_dict, nonce = 0)
                    n_cell_idx += 1
                    prev_outputs.append(output)

            blk_idx += 1
            output = F.relu(self.all_module_block_list[blk_idx](output))  # Batch norm + relu
            output = torch.mean(output, dim = (2, 3))
            output = torch.flatten(output, 1)
            blk_idx += 1
            output = F.relu(self.all_module_block_list[blk_idx](output))  # Fully connected
            return output

    def save_weight(self):
        pass

    def set_fitness(self, fitness):
        self.fitness = fitness
        self.normal_cell.set_fitness(fitness)
        self.reduction_cell.set_fitness(fitness)

    def mark_to_be_killed(self):
        self.mark_killed = True
        self.normal_cell.mark_to_be_killed()
        self.reduction_cell.mark_to_be_killed()
