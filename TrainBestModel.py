import torch

from ObjectPopulation.Model import *
from ObjectPopulation.CellPopulation import *
import torchvision.transforms as transforms
import torchvision
from DataPreprocessing.Cutout import *
from DataPreprocessing.AutoAugment import *
import torch.optim as optim

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4, fill = 128),
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

batch_size = 128

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 4)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

normal_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = False, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP)
reduction_adf_pop = ADFPopulation(ADF_HEAD_LEN, ADF_TAIL_LEN, for_reduction = True, pop_size = INIT_SIZE_ADF_POP, max_size = MAX_SIZE_ADF_POP)
reduction_cell_pop = CellPopulation(CELL_HEAD_LEN, CELL_TAIL_LEN, INIT_SIZE_CELL_POP, reduction_adf_pop)
n_cell1 = ["cat", "cat", "cat", "sum", "sep_7x1", "isep_1x7", "sum", "dep_5x5", "sum", 0, "sep_5x5", "dep_5x5", 0, "dep_1x7", 0, "dep_7x1", 0, 0, 0, 0]
n_cell2 = ["cat", "cat", "cat", "sum", "sep_7x1", "isep_1x7", "sum", "dep_5x5", "sum", 0, "sep_5x5", "dep_5x5", 0, "dep_1x7", 0, "dep_7x1", 1, 0, 1, 1]
n_cell3 = ["cat", "cat", "cat", "sum", "sep_7x1", "isep_1x7", "sum", "dep_5x5", "sum", 1, "sep_5x5", "dep_5x5", 1, "dep_1x7", 1, "dep_7x1", 2, 1, 2, 2]
r_cell = ["cat", "cat", "sum", "cat", "sum", "sep_1x7", "isep_1x7", "sum", "isep_3x3", "sep_1x7", "sep_1x7", "isep_5x5", "isep_7x1", "isep_3x3", "sep_3x3", "sep_5x5", "pwbr_1x1", "pwbr_1x1", "pwbr_1x1", "pwbr_1x1", "pwbr_1x1", "pwbr_1x1", "pwbr_1x1", 3, 3, 3, 3, 3, 2, 3]

# n_cell1 = ['point_1x1', 'cat', 'sum', 'sum', 'point_1x1', 'dep_3x3', 'dep_5x5', 'sum', 'cat', 'sep_5x3', 'dep_3x3', 0, 'isep_7x1', 'sum', 'sum', 0, 0, 0, 0, 'sep_7x1', 'sep_3x3', 'isep_1x7', 0, 0, 0]
# n_cell2 = ['point_1x1', 'cat', 'sum', 'sum', 'point_1x1', 'dep_3x3', 'dep_5x5', 'sum', 'cat', 'sep_5x3', 'dep_3x3', 1, 'isep_7x1', 'sum', 'sum', 0, 0, 0, 0, 'sep_7x1', 'sep_3x3', 'isep_1x7', 0, 1, 0]
# n_cell3 = ['point_1x1', 'cat', 'sum', 'sum', 'point_1x1', 'dep_3x3', 'dep_5x5', 'sum', 'cat', 'sep_5x3', 'dep_3x3', 1, 'isep_7x1', 'sum', 'sum', 1, 1, 2, 0, 'sep_7x1', 'sep_3x3', 'isep_1x7', 0, 0, 1]
# r_cell = ['point_1x1', 'cat', 'cat', 'cat', 'cat', 'isep_7x1', 'isep_5x3', 'sum', 'sum', 'sum', 'pwbr_1x1', 'dep_3x3', 'pwbr_1x1', 'dep_5x5', 'dep_7x1', 'pwbr_1x1', 'isep_1x7', 'isep_3x5', 2, 'dep_5x3', 2, 'pwbr_1x1', 'pwbr_1x1', 2, 'pwbr_1x1', 'pwbr_1x1', 'pwbr_1x1', 0, 0, 0, 1, 0]


# TODO: add models genotype [[n_cell1, n_cell2, n_cell3], r_cell]
model = Model(normal_adf_pop, reduction_cell_pop, best_cell_genotypes = [[n_cell1, n_cell2, n_cell3], r_cell])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay = 0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 301)
print(model.num_params)

if check_file_exist(BEST_MODEL_WEIGHTS_PATH):
    checkpoint = torch.load(BEST_MODEL_WEIGHTS_PATH)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
else:
    epoch = 0

correct = 0
total = 0
model.eval()
true_pos = {class_name: 0 for class_name in classes}
false_pos = {class_name: 0 for class_name in classes}
false_neg = {class_name: 0 for class_name in classes}
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to("cuda:0"), data[1].to("cuda:0")
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                true_pos[classes[prediction]] += 1
            else:
                false_pos[classes[prediction]] += 1
                false_neg[classes[label]] += 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
precision = {}
recall = {}
for class_name in classes:
    tp_fp = true_pos[class_name] + false_pos[class_name]
    tp_fn = true_pos[class_name] + false_neg[class_name]
    if tp_fp == 0:
        precision[class_name] = -1
    else:
        precision[class_name] = true_pos[class_name] / tp_fp
    if tp_fn == 0:
        recall[class_name] = -1
    else:
        recall[class_name] = true_pos[class_name] / tp_fn

print("Precision: ", precision)
print("Recall: ", recall)


running_loss = 0.0
while epoch < 300:  # loop over the dataset multiple times
    epoch += 1
    model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    # scheduler.step()
    if epoch % 5 == 0:    # print every 5 epochs
        print('[%d] loss: %.3f' %
              (epoch, running_loss / 5))
        running_loss = 0.0
        checkpoint_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(checkpoint_dict, BEST_MODEL_WEIGHTS_PATH)
    if epoch % 50 == 0:
        correct = 0
        total = 0
        model.eval()
        true_pos = {class_name: 0 for class_name in classes}
        false_pos = {class_name: 0 for class_name in classes}
        false_neg = {class_name: 0 for class_name in classes}
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to("cuda:0"), data[1].to("cuda:0")
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        true_pos[classes[prediction]] += 1
                    else:
                        false_pos[classes[prediction]] += 1
                        false_neg[classes[label]] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        precision = {}
        recall = {}
        for class_name in classes:
            tp_fp = true_pos[class_name] + false_pos[class_name]
            tp_fn = true_pos[class_name] + false_neg[class_name]
            if tp_fp == 0:
                precision[class_name] = -1
            else:
                precision[class_name] = true_pos[class_name] / tp_fp
            if tp_fn == 0:
                recall[class_name] = -1
            else:
                recall[class_name] = true_pos[class_name] / tp_fn

        print("Precision: ", precision)
        print("Recall: ", recall)
print('Finished Training')

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
#         # calculate outputs by running images through the network
#         outputs = model(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
