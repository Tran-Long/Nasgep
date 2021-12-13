
import torch.nn as nn
from Configs import *

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
    elif conv_type == "sep":
        modules.append(nn.Conv2d(in_channel, in_channel, kernel_size, padding = "same", groups = in_channel))
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1)))
    elif conv_type == "isep":
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1)))
        modules.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding = "same", groups = out_channel))
    elif conv_type == "1stem":
        kernel_h, kernel_w = kernel_size
        padding_h = int((kernel_h-1)/2)
        padding_w = int((kernel_w-1)/2)
        # modules.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding = (padding_h, padding_w), stride = 2))
        modules = [nn.Conv2d(in_channel, out_channel, kernel_size, padding = (padding_h, padding_w), stride = 2)]
    elif conv_type == "pwbr":
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1), stride = int(out_channel/in_channel)))
    elif conv_type == "point":
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1)))
    modules.append(nn.BatchNorm2d(out_channel))
    modules = nn.Sequential(*modules)
    return modules


class Node():
    def __init__ (self, data, channel=-1):
        self.value = data
        self.left = None
        self.right = None
        self.channel = channel

    def __str__ (self):
        return f'<{self.data} \n {self.left} \n {self.right}>'

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
    root.channel = 1
    return root.channel


def expand_tree_cell(root, adfs_genotype_dict):
    if root.value == "sum" or root.value == "cat":
        root.left = expand_tree_cell(root.left, adfs_genotype_dict)
        root.right = expand_tree_cell(root.right, adfs_genotype_dict)
    elif root.value == POINT_WISE_TERM:
        root.left = expand_tree_cell(root.left, adfs_genotype_dict)
    else:
        root = build_tree_adf(adfs_genotype_dict[root.value])
    return root

def build_tree_cell(genotype, adfs_genotype_dict):
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
    root = expand_tree_cell(root, adfs_genotype_dict)
    return root

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