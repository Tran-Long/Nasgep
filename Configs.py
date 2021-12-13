CONV_TERMS = ['dep_3x3', 'dep_5x5', 'dep_3x5', 'dep_5x3', 'dep_1x7', 'dep_7x1',
              'sep_3x3', 'sep_5x5', 'sep_3x5', 'sep_5x3', 'sep_1x7', 'sep_7x1',
              'isep_3x3', 'isep_5x5', 'isep_3x5', 'isep_5x3', 'isep_1x7', 'isep_7x1'
              ]

BEST_ACC_CONV = {conv: 0 for conv in CONV_TERMS}
CONV_BEST_PARAMS_LINKS = {conv: conv + ".pkl" for conv in CONV_TERMS}

ADF_FUNCTION = ["sum", *CONV_TERMS]
PREV_OUTPUT = "prev_output"
ADF_TERMINAL = [PREV_OUTPUT, *CONV_TERMS]
CELL_FUNCTION = ["sum", "cat"]
CELL_TERMINAL = ["ADF"]

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
NUM_OF_CONSECUTIVE_NORMAL_CELL = 3

EPOCH_MAX = 10
MIN_CHILD_ADF = 2
MAX_CHILD_ADF = 10
INIT_SIZE_ADF_POP = 50
MAX_SIZE_ADF_POP = 100
