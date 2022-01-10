import torchvision.transforms as transforms
import torchvision
from ObjectPopulation.CellPopulation import *
from ObjectPopulation.ModelPopulation import *
from ObjectPopulation.Cell import *
from Utilities.Configs import *
from DataPreprocessing.Cutout import *
from DataPreprocessing.AutoAugment import *
import time

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes = 1, length = 16),
    normalize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

batch_size = BATCH_SIZE
DEVICE = "cpu"

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_dataset, _ = torch.utils.data.random_split(train_set, [45000, 5000], generator=torch.Generator().manual_seed(37))
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
_, val_dataset = torch.utils.data.random_split(train_set, [45000, 5000], generator=torch.Generator().manual_seed(37))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Step 1, 2
if check_file_exist(CONFIG_PATH):
    config_save_dict = load_dict_checkpoint()
    T_G = config_save_dict["t_g"]
    T_C = config_save_dict["t_c"]
    REMAINING_TIME = config_save_dict["remaining_time"]
    year = config_save_dict["year"]
else:
    T_G = -1
    T_C = -1
    REMAINING_TIME = 60*60*25
    year = 0
    create_log_file()
normal_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = False, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP, save_path = NORMAL_PATH)
reduction_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = True, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP, save_path = REDUCTION_PATH)
reduction_cell_pop = CellPopulation(CELL_HEAD_LEN, CELL_TAIL_LEN, INIT_SIZE_CELL_POP, reduction_adf_pop, save_path = R_CELL_PATH)
model_pop = ModelPopulation(INIT_SIZE_MODEL_POP, NUM_OF_CONSECUTIVE_NORMAL_CELL, normal_adf_pop, reduction_cell_pop, save_path = MODEL_PATH)
# Step 3
if year == 0:
    model_pop.train_population(train_loader, val_loader, model_pop.models_dict)

base_time = time.time()

while True:
    if REMAINING_TIME <= 0:
        break
    year += 1
    print("****************************")
    write_log("****************************")
    print("*********   " + str(year) + "  **************")
    write_log("*********   " + str(year) + "  **************")
    print("****************************")
    write_log("****************************")
    # Step 4
    print("*****Step 4 - kill bad gene*****")
    write_log("*****Step 4 - kill bad gene*****")
    normal_adf_pop.kill_bad_genes(T_G)
    reduction_adf_pop.kill_bad_genes(T_G)
    # Step 5
    print("*****Step 5 - reproduction *****")
    write_log("*****Step 5 - reproduction *****")
    normal_adf_pop.reproduction()
    reduction_adf_pop.reproduction()
    reduction_cell_pop.reproduction()
    model_pop.reproduction()
    # Step 6
    print("*****Step 6 - evaluate child pop")
    write_log("*****Step 6 - evaluate child pop")
    model_pop.evaluate_population_step_6(train_loader, val_loader, model_pop.child_models_dict, T_C)
    # Step 7
    print("*****Step 7 - survivor*****")
    write_log("*****Step 7 - survivor*****")
    model_pop.survivor_selection()
    reduction_cell_pop.remove_marked_kill_cell()
    # STep 8
    print("*****Step 8 - full training and update T-g, T-c*****")
    write_log("*****Step 8 - full training and update T-g, T-c*****")
    model_pop.evaluate_population_step_6(train_loader, val_loader, model_pop.models_dict, T_C)
    model_pop.increase_age()
    T_G = min([model.fitness for (model_id, model) in model_pop.models_dict.items()])
    T_C = 0.75*T_G
    print("\tUpdated T_G, T_C:\t", end = "")
    write_log("Updated T_G, T_C:")
    print("T_G = %.2f, T_C = %.2f" % (T_G, T_C))
    REMAINING_TIME -= time.time() - base_time
    if year % 2 == 0:
        normal_adf_pop.save_checkpoint()
        reduction_adf_pop.save_checkpoint()
        reduction_cell_pop.save_checkpoint()
        model_pop.save_checkpoint()
        config_dict = {
            "t_g": T_G,
            "t_c": T_C,
            "remaining_time": REMAINING_TIME,
            "year": year
        }
        save_dict_checkpoint(config_dict, CONFIG_PATH)
        print("==> Checkpoint saved")

best_model_genotypes = model_pop.get_best_models()
for model_genotype in best_model_genotypes:
    n_genotypes = model_genotype[0]
    for n_geno in n_genotypes:
        print(n_geno)
        write_log(get_string_fr_arr(n_geno))
    r_genotype = model_genotype[1]
    print(r_genotype)
    write_log(get_string_fr_arr(r_genotype))
    print("\n")
