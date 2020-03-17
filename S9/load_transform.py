import torch
import torchvision
from torchvision import datasets, transforms

# Train phase transformations
train_transforms = transforms.Compose([
    # convert the data to torch.FloatTensor with values within the range [0.0 ,1.0]
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Validation phase transformations
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Downloading Data 
train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
val_data = datasets.CIFAR10('./data', train=False, download=True, transform=val_transforms)

SEED = 1

cuda = torch.cuda.is_available()
print('CUDA Available?', cuda)

# For reproducibility of results
torch.manual_seed(SEED)
if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64, num_workers=2)
dataloader_args_for_plot = dict(shuffle=True, batch_size=4, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
trainloader_for_plot = torch.utils.data.DataLoader(train_data, **dataloader_args_for_plot)

# validation dataloader
val_loader = torch.utils.data.DataLoader(val_data, **dataloader_args)
testloader_for_plot = torch.utils.data.DataLoader(val_data, **dataloader_args_for_plot)