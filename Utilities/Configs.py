import os

CONV_TERMS = ['dep_3x3', 'dep_5x5', 'dep_3x5', 'dep_5x3', 'dep_1x7', 'dep_7x1',
              'sep_3x3', 'sep_5x5', 'sep_3x5', 'sep_5x3', 'sep_1x7', 'sep_7x1',
              'isep_3x3', 'isep_5x5', 'isep_3x5', 'isep_5x3', 'isep_1x7', 'isep_7x1'
              ]

BEST_ACC_CONV = {conv: 0 for conv in CONV_TERMS}
CONV_BEST_PARAMS_LINKS = {conv: conv + ".pkl" for conv in CONV_TERMS}
LOG_FILE = os.path.join(os.path.dirname(os.getcwd()), "log.txt")

# INIT_PARAMS = "He_normal"
INIT_PARAMS = "He_uniform"

"""LEARNING RATE & REGULARIZATION"""
DROP_PATH_RATE = 0.1
LR = 0.1
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
BATCH_SIZE = 512

DATASET = "cifar-10"
DEVICE = "cuda:0"

MAX_MODEL_PARAMS = 300000

"""ADF & CELL Hyper params"""
ADF_HEAD_LEN = 1
ADF_TAIL_LEN = 2
CELL_HEAD_LEN = 4
CELL_TAIL_LEN = 5
ADF_FUNCTION = ["sum", *CONV_TERMS]
PREV_OUTPUT = "prev_output"
ADF_TERMINAL = [PREV_OUTPUT, *CONV_TERMS]
CELL_FUNCTION = ["sum", "cat"]
CELL_TERMINAL = ["ADF"]
"""END OF ADF & CELL Hyper params"""

"""CONVOLUTION BLOCK Hyper params"""
INPUT_CHANNELS = 3
#
# # IMAGENET 3->32->64->64->...
# NUM_CHANNELS = 64

# CIFAR-10 3->16->...
NUM_CHANNELS = 16

STEM_KERNEL_SIZE = (3, 3)
STEM_TERM = "1stem_3x3"
POINT_WISE_TERM = "point_1x1"  # for lowering channel
POINT_WISE_BEFORE_REDUCTION_TERM = "pwbr_1x1"  # for reduction
ADF_PREFIX = "adf"
CELL_PREFIX = "cell"
MODEL_PREFIX = "model"
NUM_OF_CONSECUTIVE_NORMAL_CELL = 3
"""END OF CONVOLUTION BLOCK Hyper params"""

"""Population Hyper params"""
EPOCH_MAX = 10
MIN_CHILD_ADF = 2
MAX_CHILD_ADF = 10
INIT_SIZE_ADF_POP = 50
MAX_SIZE_ADF_POP = 100

INIT_SIZE_CELL_POP = 10

INIT_SIZE_MODEL_POP = 10
"""END OF Population Hyper params"""


"""Reproduction Hyper params"""
TOURNAMENT_SELECTION_SIZE = 4

RPD_MUTATION_RATE = 0.3
RPD_TRANSPOSITION_RATE = 0.1
RPD_IS_ELE_CNT = 3
RPD_ROOT_TRANSPOSITION_RATE = 0.1
RPD_RIS_ELE_CNT = 3
RPD_1_RECOMBINATION_RATE = 0.5
RPD_2_RECOMBINATION_RATE = 0.2

RPD_CELL_MUTATION_RATE = 0.3
RPD_CELL_TRANSPOSITION_RATE = 0.3
RPD_CELL_ROOT_TRANSPOSITION_RATE = 0.4
"""END OF Reproduction Hyper params"""


