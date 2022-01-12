import torch
import torch.nn.functional as F
from ObjectPopulation.Cell import *

class Model(nn.Module):
    def __init__(self, n_adf_population, r_cell_population, n = NUM_OF_CONSECUTIVE_NORMAL_CELL,
                 normal_cell=None, reduction_cell_id=None, for_dataset=DATASET, best_cell_genotypes = None):
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
        self.weight_path = None
        self.training_status = True
        self.drop_path_rate = DROP_PATH_RATE
        # Select cells
        if normal_cell is None and reduction_cell_id is None:
            self.reduction_cell_id, self.reduction_cell = r_cell_population.select_random_reduction_cell()
            self.normal_cell = Cell(n_adf_population)
        else:
            self.normal_cell = normal_cell
            self.reduction_cell_id = reduction_cell_id
            self.reduction_cell = r_cell_population.cells_dict[reduction_cell_id]
        """--------------------------------------------"""
        # print("\t\t\t", end = "")
        # print(self.normal_cell.genotype)
        # print("\t\t\t", end = "")
        # print(self.reduction_cell.genotype)
        """--------------------------------------------"""

        self.n_cell_roots_list = []
        self.r_cell_roots_list = []
        if best_cell_genotypes is not None:
            normal_cell_genotypes = best_cell_genotypes[0]
            for i in range(n):
                self.n_cell_roots_list.append(build_tree(normal_cell_genotypes[i]))
            reduction_cell_genotype = best_cell_genotypes[1]
            self.r_cell_roots_list.append(build_tree(reduction_cell_genotype))
            self.normal_cell.root = self.n_cell_roots_list[0]
            self.reduction_cell.root = self.r_cell_roots_list[0]

        # Init network representation
        current_input_channel = 3
        prev_outputs = []
        if for_dataset == "cifar-10":
            self.all_module_block_list.append(nn.ModuleList([conv_block(STEM_TERM, current_input_channel, NUM_CHANNELS)]))
            current_input_channel = 16
            prev_outputs.append(0)

            # 1st normal block => channel = 16
            for i in range(n):
                temp_cell = copy.deepcopy(self.normal_cell)
                module_dict, path_dict = temp_cell.create_modules_dict(prev_outputs, current_input_channel)
                if best_cell_genotypes is None:
                    self.n_cell_roots_list.append(temp_cell.root)
                self.all_module_block_list.append(module_dict)
                self.all_cell_file_list.append(path_dict)
                prev_outputs.append(0)

            for k in range(2):
                current_input_channel *= 2
                temp_cell = copy.deepcopy(self.reduction_cell)
                module_dict, path_dict = temp_cell.create_modules_dict(prev_outputs, current_input_channel)
                if len(self.r_cell_roots_list) == 0:
                    self.r_cell_roots_list.append(temp_cell.root)
                self.all_module_block_list.append(module_dict)
                self.all_cell_file_list.append(path_dict)
                prev_outputs = [0]
                for i in range(n):
                    temp_cell = copy.deepcopy(self.normal_cell)
                    module_dict, path_dict = temp_cell.create_modules_dict(prev_outputs, current_input_channel)
                    self.all_module_block_list.append(module_dict)
                    self.all_cell_file_list.append(path_dict)
                    prev_outputs.append(0)
            self.all_module_block_list.append(nn.BatchNorm2d(current_input_channel))
            self.all_module_block_list.append(nn.Linear(64, 10))
        # Init optimizer and loss
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.to(DEVICE)
        self.optimizer = torch.optim.SGD(self.parameters(), lr = LR, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
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
            if self.training_status:
                # local drop path
                drop_mask = [1, 1]
                if np.random.rand() < self.drop_path_rate:
                    drop_mask[0] = 0
                elif np.random.rand() < self.drop_path_rate:
                    drop_mask[1] = 0
                output = left_value * drop_mask[0] + right_value * drop_mask[1]
                output /= self.drop_path_rate
            else:
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
            output = x
            prev_outputs = []
            for layer in self.all_module_block_list[blk_idx]:
                output = layer(output)
            prev_outputs.append(output)
            for i in range(self.n):
                blk_idx += 1
                module_dict = self.all_module_block_list[blk_idx]
                output, _ = self.cell_forward(self.n_cell_roots_list[i], prev_outputs, module_dict, nonce = 0)
                prev_outputs.append(output)

            for k in range(2):
                blk_idx += 1
                module_dict = self.all_module_block_list[blk_idx]
                output, _ = self.cell_forward(self.r_cell_roots_list[0], prev_outputs, module_dict, nonce = 0)
                prev_outputs = [output]
                for i in range(self.n):
                    blk_idx += 1
                    module_dict = self.all_module_block_list[blk_idx]
                    output, _ = self.cell_forward(self.n_cell_roots_list[i], prev_outputs, module_dict, nonce = 0)
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
        write_log("Remove cell:" + str(self.reduction_cell_id))

    def show_info(self):
        print("\t\tModel info: ")
        write_log("Model info: ")
        print("\t\tNormal root info: ")
        write_log("Normal root info: ")
        view_tree(self.normal_cell.root)
        write_log(get_string_fr_arr(self.normal_cell.genotype))
        print("\t\tReduction root info: ")
        write_log("Reduction root info: ")
        view_tree(self.reduction_cell.root)
        write_log(get_string_fr_arr(self.reduction_cell.genotype))

    def get_info_to_save(self):
        zip_model_info = []
        normal_genotypes = []
        for root in self.n_cell_roots_list:
            normal_genotypes.append(bfs(root))
        zip_model_info.append(normal_genotypes)
        zip_model_info.append(bfs(self.r_cell_roots_list[0]))
        return zip_model_info

    def save_checkpoint(self):
        checkpoint = {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        torch.save(checkpoint, self.weight_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.weight_path)
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
