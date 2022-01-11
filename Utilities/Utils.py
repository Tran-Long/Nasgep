import os.path
import pickle

import torch.nn as nn
import torch.nn.init

from Utilities.Configs import *

def create_file_name_conv(conv_term, in_channel, out_channel):
    return conv_term + "_" + str(in_channel) + "_" + str(out_channel) + ".pth"

def get_detail_conv(conv_term):
    array = conv_term.split("_")
    return array[0], array[1]

def conv_block (conv_term, in_channel, out_channel):
    modules = [nn.ReLU()]
    conv_type, kernel = get_detail_conv(conv_term)
    kernel_size = (int(kernel[0]), int(kernel[-1]))
    if conv_type == "dep":
        assert in_channel == out_channel, "Depthwise cannot deal with different in/out"
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding = "same", groups = in_channel))
        if INIT_PARAMS == "He_normal":
            torch.nn.init.kaiming_normal_(modules[-1].weight)
        elif INIT_PARAMS == "He_uniform":
            torch.nn.init.kaiming_uniform_(modules[-1].weight)

    elif conv_type == "sep":
        modules.append(nn.Conv2d(in_channel, in_channel, kernel_size, padding = "same", groups = in_channel))
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1)))
        if INIT_PARAMS == "He_normal":
            torch.nn.init.kaiming_normal_(modules[-1].weight)
            torch.nn.init.kaiming_normal_(modules[-2].weight)
        elif INIT_PARAMS == "He_uniform":
            torch.nn.init.kaiming_uniform_(modules[-1].weight)
            torch.nn.init.kaiming_uniform_(modules[-2].weight)
    elif conv_type == "isep":
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1)))
        modules.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding = "same", groups = out_channel))
        if INIT_PARAMS == "He_normal":
            torch.nn.init.kaiming_normal_(modules[-1].weight)
            torch.nn.init.kaiming_normal_(modules[-2].weight)
        elif INIT_PARAMS == "He_uniform":
            torch.nn.init.kaiming_uniform_(modules[-1].weight)
            torch.nn.init.kaiming_uniform_(modules[-2].weight)
    elif conv_type == "1stem":
        kernel_h, kernel_w = kernel_size
        padding_h = int((kernel_h-1)/2)
        padding_w = int((kernel_w-1)/2)
        # modules.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding = (padding_h, padding_w), stride = 2))
        modules = [nn.Conv2d(in_channel, out_channel, kernel_size, padding = (padding_h, padding_w), stride = 2)]
        if INIT_PARAMS == "He_normal":
            torch.nn.init.kaiming_normal_(modules[-1].weight)
        elif INIT_PARAMS == "He_uniform":
            torch.nn.init.kaiming_uniform_(modules[-1].weight)
    elif conv_type == "pwbr":
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1), stride = int(out_channel/in_channel)))
        if INIT_PARAMS == "He_normal":
            torch.nn.init.kaiming_normal_(modules[-1].weight)
        elif INIT_PARAMS == "He_uniform":
            torch.nn.init.kaiming_uniform_(modules[-1].weight)
    elif conv_type == "point":
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1)))
        if INIT_PARAMS == "He_normal":
            torch.nn.init.kaiming_normal_(modules[-1].weight)
        elif INIT_PARAMS == "He_uniform":
            torch.nn.init.kaiming_uniform_(modules[-1].weight)
    modules.append(nn.BatchNorm2d(out_channel))
    modules = nn.Sequential(*modules)
    return modules


class Node:
    def __init__ (self, data, channel=-1):
        self.value = data
        self.left = None
        self.right = None
        self.channel = channel


def validate_tree_adf(root):
    if root.value in CONV_TERMS:
        if root.left is None:
            root.left = Node(PREV_OUTPUT)
        else:
            root.left = validate_tree_adf(root.left)
        return root
    if root.value == "sum":
        root.left = validate_tree_adf(root.left)
        root.right = validate_tree_adf(root.right)
        return root
    return root

def build_tree_adf (genotype):
    root = Node(genotype[0])
    i = 1
    this_level = [root]
    appendable = True
    while appendable and i < len(genotype):
        appendable = False
        next_level = []
        for node in this_level:
            if node.value == "sum":
                appendable = True
                if i < len(genotype):
                    node.left = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.left)
                if i < len(genotype):
                    node.right = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.right)
            elif node.value in CONV_TERMS:
                appendable = True
                if i < len(genotype):
                    node.left = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.left)
        this_level = next_level
    """Only ADF can be missing 'extra' node for convolution :V"""
    root = validate_tree_adf(root)
    return root


def calculate_channel_mul(root):
    """
    Adding extra pointwise for correct channel dimension
    :param root: root of cell tree
    :return: root of cell tree
    """
    if root.channel > 0:
        return root.channel
    if root.value == "cat":
        root.channel = calculate_channel_mul(root.left) + calculate_channel_mul(root.right)
        return root.channel
    if root.value == "sum":
        left_channel = calculate_channel_mul(root.left)
        right_channel = calculate_channel_mul(root.right)
        if left_channel != right_channel:
            if left_channel > 1:
                t_left = root.left
                root.left = Node(POINT_WISE_TERM, channel = 1)
                root.left.left = t_left
            if right_channel > 1:
                t_right = root.right
                root.right = Node(POINT_WISE_TERM, channel = 1)
                root.right.left = t_right
            root.channel = 1
        else:
            root.channel = left_channel
        return root.channel
    # This part for best model :V
    if root.value == POINT_WISE_TERM:
        calculate_channel_mul(root.left)
        root.channel = 1
        return root.channel
    # else other convolution
    root.channel = 1
    return root.channel

def expand_tree_cell(root, adfs_dict):
    if root.value == "sum" or root.value == "cat":
        root.left = expand_tree_cell(root.left, adfs_dict)
        root.right = expand_tree_cell(root.right, adfs_dict)
    elif root.value == POINT_WISE_TERM:
        root.left = expand_tree_cell(root.left, adfs_dict)
    else:
        root = adfs_dict[root.value].root
    return root

def build_tree_cell(genotype, adfs_dict):
    root = Node(genotype[0])
    i = 1
    this_level = [root]
    appendable = True
    while appendable and i < len(genotype):
        appendable = False
        next_level = []
        for node in this_level:
            if node.value == "sum" or node.value == "cat":
                appendable = True
                if i < len(genotype):
                    node.left = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.left)
                if i < len(genotype):
                    node.right = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.right)
        this_level = next_level

    calculate_channel_mul(root)
    if root.channel > 1:
        new_root_pw = Node(POINT_WISE_TERM, channel = 1)
        new_root_pw.left = root
        root = new_root_pw
    root = expand_tree_cell(root, adfs_dict)
    return root

def build_tree(genotype):
    root = Node(genotype[0])
    i = 1
    this_level = [root]
    appendable = True
    while appendable and i < len(genotype):
        appendable = False
        next_level = []
        for node in this_level:
            if node.value == "sum" or node.value == "cat":
                appendable = True
                if i < len(genotype):
                    node.left = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.left)
                if i < len(genotype):
                    node.right = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.right)
            elif node.value in CONV_TERMS or node.value == POINT_WISE_TERM or node.value == POINT_WISE_BEFORE_REDUCTION_TERM:
                appendable = True
                if i < len(genotype):
                    node.left = Node(genotype[i])
                    i = i + 1
                    next_level.append(node.left)
        this_level = next_level
    calculate_channel_mul(root)
    if root.channel > 1:
        new_root_pw = Node(POINT_WISE_TERM, channel = 1)
        new_root_pw.left = root
        root = new_root_pw
    return root

def bfs(root):
    r = root
    result = []
    queue = []
    result.append(r.value)
    queue.append(r)
    while queue:          # Creating loop to visit each node
        m = queue.pop(0)
        if m.left is not None:
            result.append(m.left.value)
            queue.append(m.left)
        if m.right is not None:
            result.append(m.right.value)
            queue.append(m.right)
    return result

def view_tree(t):
    t_level = [t]
    while len(t_level) > 0:
        n_level = []
        for i, t_node in enumerate(t_level):
            print(t_node.value, end = "  ")
            if t_node.value != "prev_output":
                if t_node.left is not None:
                    n_level.append(t_node.left)
                if t_node.right is not None:
                    n_level.append(t_node.right)
        print()
        t_level = n_level

def view_tree_channel(t):
    t_level = [t]
    while len(t_level) > 0:
        n_level = []
        for i, t_node in enumerate(t_level):
            print(t_node.channel, end = "  ")
            if t_node.value != "prev_output":
                if t_node.left is not None:
                    n_level.append(t_node.left)
                if t_node.right is not None:
                    n_level.append(t_node.right)
        print()
        t_level = n_level

def view_model_info(model_id, model):
    # print("\tModel " + model_id + " created with " + str(model.num_params) + " parameter")
    write_log("Model " + model_id + " created with " + str(model.num_params) + " parameter")
    # print("\t\t", end = "")
    # print(model.normal_cell.genotype)
    write_log(get_string_fr_arr(model.normal_cell.genotype))
    # print("\t\t", end = "")
    # print(model.reduction_cell.genotype)
    write_log(get_string_fr_arr(model.reduction_cell.genotype))

def get_string_fr_arr(arr):
    return str(arr)

def write_log(data):
    f = open(LOG_FILE, "a")
    f.write(data + "\n")
    f.close()

def create_log_file():
    f = open(LOG_FILE, "w")
    f.close()

def check_file_exist(file_path):
    return os.path.exists(file_path)

def save_dict_checkpoint(save_dict, path):
    with open(path, 'wb') as fp:
        pickle.dump(save_dict, fp)

def load_dict_checkpoint(path):
    with open(path, 'rb') as fp:
        save_dict = pickle.load(fp)
    return save_dict

def make_path(weight_file):
    # return os.path.join(CHECKPOINT_PATH, weight_file)
    return WEIGHTS_FOLDER_PATH + weight_file

def clear_folder(folder_path):
    for f in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, f))
