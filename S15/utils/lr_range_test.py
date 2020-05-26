import torch
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, StepLR
import utils.regularization  as regularization # L1 loss fxn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class LRRangeTest(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None
    ):
        self.optimizer = optimizer
        self.model = model.to(device)
        self.criterion = criterion
        self.history = {"lr": [], "train_acc": [], "val_acc": [], "train_losses": [], "val_losses": []}
        self.best_loss = None
        self.best_acc = None
        self.best_lr = None
        self.device = device

    # function to train the model on training dataset
    def train(self, train_loader, lr_scheduler, L1_loss_enable=False):
        self.model.train()
        #pbar = tqdm(train_loader)
        train_loss = 0
        correct = 0
        processed = 0
        #for batch_idx, (data, target) in enumerate(pbar):
        for batch_idx, (data, target) in enumerate(train_loader):
          # get samples
          data, target = data.to(self.device), target.to(self.device)

          # Init
          self.optimizer.zero_grad()
          # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
          # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

          # Predict
          y_pred = self.model(data)

          # Calculate loss
          loss = self.criterion(y_pred, target)
          
          if(L1_loss_enable == True):
            regloss = regularization.L1_Loss_calc(model, 0.0005)
            regloss /= len(data) # by batch size
            loss += regloss

          train_loss += loss.item()

          # Backpropagation
          loss.backward()
          self.optimizer.step()

          curLR = lr_scheduler.get_last_lr()[0]
          if(lr_scheduler != None): # this is for batchwise lr update
            lr_scheduler.step()
            curLR = lr_scheduler.get_last_lr()[0]

          # Update pbar-tqdm

          pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
          processed += len(data)

          #pbar.set_description(desc= f'Loss={loss.item():0.6f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
          #pbar.set_description(desc= f'Loss={train_loss/(batch_idx+1):0.6f} Batch_id={batch_idx+1} Accuracy={100*correct/processed:0.2f}')
          #pbar.set_description(desc= f'Loss={train_loss/(batch_idx+1):0.6f} Batch_id={batch_idx+1} Accuracy={100*correct/processed:0.2f} lr={curLR}')

        train_loss /= len(train_loader)
        acc = 100. * correct/len(train_loader.dataset) #processed # 
        return np.round(acc,2), np.round(train_loss,6)

    # function to test the model on testing dataset
    def test(self, test_loader, L1_loss_enable=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                #test_loss += criterion(output, target, reduction='sum').item()   # sum up batch loss # criterion = F.nll_loss
                test_loss += self.criterion(output, target).item()                     # sum up batch loss # criterion = nn.CrossEntropyLoss()
                
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            #test_loss /= len(test_loader.dataset)  # criterion = F.nll_loss
            test_loss /= len(test_loader)           # criterion = nn.CrossEntropyLoss()

            if(L1_loss_enable == True):
              regloss = regularization.L1_Loss_calc(model, 0.0005)
              regloss /= len(test_loader.dataset) # by batch size which is here total test dataset size
              test_loss += regloss

        #print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        #    test_loss, correct, len(test_loader.dataset),
        #    100. * correct / len(test_loader.dataset)))
        
        acc = 100. * correct / len(test_loader.dataset)
        return np.round(acc,2), test_loss

    def range_test(self, trainloader, testloader, start_lr, end_lr, epochs):

        step_size_up = epochs*len(trainloader)   # stepsize for LR cycle policy

        # CyclicLR- use one cycle policy: MAX-LR at end of last epoch, Triangular policy
        lr_scheduler = CyclicLR(self.optimizer, base_lr=start_lr, max_lr=end_lr, step_size_up=step_size_up, last_epoch=-1)

        print("Running LR Range test")
        for epoch in range(1,epochs+1):

          cur_lr1 = self.optimizer.state_dict()["param_groups"][0]["lr"]
      
          train_acc, train_loss = self.train(trainloader, lr_scheduler=lr_scheduler)
           
          test_acc, test_loss = self.test(testloader)

          cur_lr2 = self.optimizer.state_dict()["param_groups"][0]["lr"]

          # store the epoch train and test results
          self.addToHistory(train_acc, train_loss, test_acc, test_loss, cur_lr2)
          print("Epoch={} Accuracy={} lr={} ==> {}".format(epoch, test_acc, cur_lr1, cur_lr2))

    def addToHistory(self, train_acc, train_loss, val_acc, val_loss, lr):
        self.history["train_acc"].append(train_acc)
        self.history["train_losses"].append(train_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_losses"].append(val_loss)
        self.history["lr"].append(lr)
        return

    def plot(self, title="LR vs Accuracy", save_filename="lr_rangetest_plot"):

        fig = plt.figure(figsize=(20,5))

        ydata = self.history["val_acc"]
        xdata = self.history["lr"]

        plt.plot(xdata, ydata)

        plt.title(title)
        plt.xlabel('LR')
        plt.ylabel("Accuracy")

        plt.show()
        fig.savefig("./images/{}.png".format(save_filename))

    def getLrForMaxAccuracy(self):
        values = self.history["val_acc"]
        index = values.index(np.max(values))
        return self.history["lr"][index]
