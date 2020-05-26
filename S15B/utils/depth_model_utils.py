import torch
import torchvision
import torch.nn as nn
import gc
import time
import numpy as np
import utils.regularization  as regularization # L1 loss fxn
from utils.model_history import ModelHistory
import utils.plot_utils as plot_utils
from tqdm import tqdm

from utils.TimeMemoryProfile import MemoryUsage, TimeProfile
from torch.utils.tensorboard import SummaryWriter

class DepthModelUtils():
  def __init__(self, ds_means_stds=None, saved_model_dir=None, saved_results_dir=None, tb_log_dir=None, saved_results_freq=1000, tqdm_status=False):
      self.ds_means_stds = ds_means_stds
      self.saved_model_dir = saved_model_dir
      self.saved_results_dir = saved_results_dir
      self.saved_results_freq = saved_results_freq
      self.tqdm_status = tqdm_status
      self.tb_writer = SummaryWriter(f'{tb_log_dir}/train_model')
      self.time_profile = TimeProfile()
      self.memory_usage = MemoryUsage()

  '''
  function to train the model on training dataset
  '''
  def train(self, model, epoch, device, train_loader, criterion, optimizer, batch_lr_update, lr_scheduler, L1_loss_enable=False):
      model.train()
      total_loss = 0
      total_loss_m = 0
      total_loss_d = 0
      correct = 0
      processed = 0
      tb_freq = 100 # tensor board log frequency
      tb_pos = (epoch-1)*(len(train_loader)//tb_freq) # every set of batches

      pbar = train_loader
      if self.tqdm_status:
        pbar = tqdm(train_loader)

      for batch_idx, data in enumerate(pbar):
        # get samples
        load_time = time.time()
        data['bg'], data['fgbg'], data['mask'], data['depth'] = data['bg'].to(device), data['fgbg'].to(device), data['mask'].to(device), data['depth'].to(device)
        load_time = time.time() - load_time

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        fwd_time = time.time()
        # combine both mask and depth channels - 3+3=6
        inp = torch.cat([data['bg'], data['fgbg']], dim=1)
        output = model(inp) 
        fwd_time = time.time() - fwd_time

        # Calculate loss
        loss_time = time.time()
        loss, loss_m, loss_d = criterion(output[0], data['mask'], output[1], data['depth'])
        loss_time = time.time() - loss_time
              
        total_loss_m += loss_m.item()
        total_loss_d += loss_d.item()
        total_loss += loss.item()

        # Backpropagation
        backprop_time = time.time()
        loss.backward()
        optimizer.step()
        backprop_time = time.time() - backprop_time

        curLR = optimizer.state_dict()["param_groups"][0]["lr"]
        if(lr_scheduler != None) and (batch_lr_update == True): # this is for batchwise lr update
          lr_scheduler.step()
          curLR = lr_scheduler.get_last_lr()[0]

        running_avg_loss = total_loss/(batch_idx+1)
        running_avg_loss_m = total_loss_m/(batch_idx+1)
        running_avg_loss_d = total_loss_d/(batch_idx+1)
        
        # save the best_model
        if (batch_idx+1) % self.saved_results_freq == 0:
          # save the model
          self.save_model(model, epoch, batch_idx+1, optimizer, criterion, running_avg_loss, "train")
          #memory status
          print(f'\nEpoch: {epoch} ==> Memory usage => {self.memory_usage.show()}')
          # show and save prediction result images
          self.save_plot_results(output, data, epoch, batch_idx+1)

        # write to tensor board..after evry few batches.. ensure this does not slow the training so adjust log frequent accordingly
        if (batch_idx+1)%tb_freq == 0:
          tb_losses = {"loss": running_avg_loss, "loss_m": running_avg_loss_m, "loss_d": running_avg_loss_d}
          self.tb_writer.add_scalars("training_loss", tb_losses,tb_pos)
          
          self.tb_writer.add_scalars("training_memory", self.memory_usage.show(),tb_pos)

          timelog = {"load": load_time, "forward": fwd_time, "loss": loss_time, "backprop": backprop_time}
          self.tb_writer.add_scalars("training_time", timelog, tb_pos)
          tb_pos += 1
        
        # Update pbar-tqdm
        #pbar.set_description(desc= f'Loss={train_loss/(batch_idx+1):0.9f} Batch_id={batch_idx+1} Accuracy={100*correct/processed:0.2f} lr={curLR}')
        if self.tqdm_status:
          pbar.set_description(desc= f'Epoch: {epoch} ==> Loss={running_avg_loss:0.12f}, Loss_M={running_avg_loss_m:0.12f}, Loss_D={running_avg_loss_d:0.12f}, lr={curLR:0.12f}')
        else:
          # display how much iteration is over
          logfreq = len(train_loader)//20 # every 5% is over
          if ((batch_idx+1) % logfreq == 0):
            print(f'Epoch: {epoch} ==> Loss={running_avg_loss:0.12f}, Loss_M={running_avg_loss_m:0.12f}, Loss_D={running_avg_loss_d:0.12f}, lr={curLR:0.12f} [Batch={batch_idx+1}/{len(train_loader)}, {(100.*(batch_idx+1)/len(train_loader)):0.0f}%]')
      
      total_loss /= len(train_loader)
      return np.round(0,2), np.round(total_loss,6)
      
  '''
  function to test the model on testing dataset
  '''
  def test(self, model, epoch, device, test_loader, criterion, L1_loss_enable=False):
      model.eval()
      loss_m = 0 # mask
      loss_d = 0 # depth
      loss = 0   # total
      correct = 0
      with torch.no_grad():
          pbar = test_loader
          if self.tqdm_status:
            pbar = tqdm(test_loader)
          for batch_idx, data in enumerate(pbar):
              data['bg'], data['fgbg'], data['mask'], data['depth'] = data['bg'].to(device), data['fgbg'].to(device), data['mask'].to(device), data['depth'].to(device)
              
              # combine both mask and depth channels - 3+3=6
              inp = torch.cat([data['bg'], data['fgbg']], dim=1)
              output = model(inp) 
              
              loss_t, loss_m, loss_d = criterion(output[0], data['mask'], output[1], data['depth'])
              loss += loss_t.item()
              avg_loss = loss/(batch_idx+1)

              if self.tqdm_status:
                pbar.set_description(desc= f'Epoch: {epoch} ==> Test Loss={avg_loss:0.6f}')
              else:
                # display how much iteration is over
                logfreq = len(test_loader)//20 # every 5% is over
                if ((batch_idx+1) % logfreq == 0):
                   print(f'Epoch: {epoch} Test ==> Loss={avg_loss:0.12f}[Batch={batch_idx+1}/{len(test_loader)},{(100.*(batch_idx+1)/len(test_loader)):0.0f}%]')

          loss /= len(test_loader)

      print(f'Test set: loss: {loss:.6f}\n')
      return np.round(0,2), np.round(loss,9)
      
  '''
  build and train the model. epoch based result is store in Modelhistory and returned.
  LR is update epoch wise
  '''
  def build_model(self, model, device, trainloader, testloader, start_epoch, epochs, criterion, optimizer, 
          lr_scheduler=None, reduceLr_scheduler=None, batch_lr_update=False, L1_loss_enable=False):

      # object to store model hsitory such as training and test accuracy and losses, epoch wise lr values etc
      history = ModelHistory(epochs)
      for epoch in range(start_epoch,epochs+1):
        # read the current epoch Lr values
        cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print("EPOCH-{}: learning rate is: {}".format(epoch, cur_lr))
    
        train_acc, train_loss = self.train(model, epoch, device, trainloader, criterion, optimizer, batch_lr_update, lr_scheduler=lr_scheduler, L1_loss_enable=L1_loss_enable)

        if(lr_scheduler != None) and (batch_lr_update == False):
          lr_scheduler.step()
          
        test_acc, test_loss = self.test(model, epoch, device, testloader, criterion, L1_loss_enable=L1_loss_enable)

        if(reduceLr_scheduler != None):
          reduceLr_scheduler.step(test_loss)
        
        self.save_model(model, epoch, 0, optimizer, criterion, test_loss, "test")

        # write to tensor board
        ep_results = {"train_loss": train_loss, "test_loss": test_loss, "cur_lr": cur_lr}
        self.tb_writer.add_scalars("epoch_results", ep_results, epoch)

        # store the epoch train and test results
        history.append_epoch_result(train_acc, train_loss, test_acc, test_loss, cur_lr)
        gc.collect()
      return history

  def save_plot_results(self, output, data, epoch, bidx):
      if self.saved_results_dir:
        plot_utils.visualize_data(output[0], figsize=(10,10), filename=self.saved_results_dir/f'ep{epoch}_b{bidx}_mask_pred.jpg', show=True, nrow=4)
        plot_utils.visualize_data(output[1], figsize=(10,10), filename=self.saved_results_dir/f'ep{epoch}_b{bidx}_depth_pred.jpg', show=True, nrow=4)
        plot_utils.visualize_data_norm(data["bg"], self.ds_means_stds['bg_means'], self.ds_means_stds['bg_stds'], figsize=(10,10), filename=self.saved_results_dir/f'ep{epoch}_b{bidx}_bg.jpg', show=False, nrow=4)
        plot_utils.visualize_data_norm(data["fgbg"], self.ds_means_stds['fg_bg_means'], self.ds_means_stds['fg_bg_stds'], figsize=(10,10), filename=self.saved_results_dir/f'ep{epoch}_b{bidx}_fgbg.jpg', show=False, nrow=4)
        plot_utils.visualize_data(data["mask"], figsize=(10,10), filename=self.saved_results_dir/f'ep{epoch}_b{bidx}_mask_gt.jpg', show=False, nrow=4)
        plot_utils.visualize_data(data["depth"], figsize=(10,10), filename=self.saved_results_dir/f'ep{epoch}_b{bidx}_depth_gt.jpg', show=False, nrow=4)

  def save_model_dict(self, model, epoch, bidx, loss, tag):
      if self.saved_model_dir:
        torch.save(model.state_dict(), self.saved_model_dir/f'model_ep{epoch}_b{bidx}_{tag}loss_{loss:0.12f}.pth')
  
  def save_model(self, model, epoch, bidx, optimizer, losslogger, loss, tag):
      if self.saved_model_dir:
        filename = self.saved_model_dir/f'model_ep{epoch}_b{bidx}_{tag}loss_{loss:0.9f}.pth'
        if tag=='test':
          filename = self.saved_model_dir/f'model_ep{epoch}_{tag}loss_{loss:0.9f}.pth'
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'losslogger': losslogger, }
        torch.save(state, filename)
  
  