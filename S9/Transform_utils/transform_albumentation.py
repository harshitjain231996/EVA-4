import torch
import torchvision
from trainalbumentation import TrainAlbumentation
from testalbumentation import TestAlbumentation

SEED = 1

class Data():

  def __init__(self):
    self.train_album = TrainAlbumentation()
    self.test_album = TestAlbumentation()

  def getTrainDataSet(self, train=True):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.train_album)
    return dataset

  def getTestDataSet(self, train=False):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.test_album)
    return dataset

  def getDataLoader(self, dataset, batches):
    # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batches, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

    
  def getGradCamDataLoader(self, dataset):
  # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=1, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=1)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader
