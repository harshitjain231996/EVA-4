import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import zipfile
from io import BytesIO
import numpy as np
import gc
from tqdm import tqdm

'''
1. This is dataset to read image from ZIP file
2. reading is done once during ojject creation and store inot RAM. this help to avoid reading during getitem and speedup training time.
3. as this load all images into RAM, this class is suitable for limited dataset size that fit into RAM
Creating custom dataset class by inherting torch.utils.data.Dataset and overide following methods:
1. __len__ : returns the size of the dataset
2. __getitem__: to support indexing such that dataset[i] can be used to get the sample

referenced material: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
class DepthMapDatasetZip(Dataset):
      """ Depth Map dataset"""
      def __init__(self, zf_dict, im_files, bg_transform=None, fgbg_transform=None, mask_transform=None, depth_transform=None):
          """
          Args:
          zf_dict (dict): zip file resources for all dataset with items names : "bg", "fgbg", "mask" and "depth"
          im_files: list of file to be loaded into memory (file name format: fg001_bg001_10.jpg)
          data_transform (callable, optional): optional transform to be applied on bg and fgbg images 
          target_transform (callable, optional): optional transform to be applied on mask and depth images. these are ground truth

          NOTE: zip file resources shall be open and closed by the user
          """
          self.zf_dict =  zf_dict

          self.bg_transform = bg_transform
          self.fgbg_transform = fgbg_transform
          self.mask_transform = mask_transform
          self.depth_transform = depth_transform

          # fgbg, mask and depth retain the same file names
          self.im_files = im_files

          '''
          keeping all the data loaded and applied data transformation as well. 
          doing so mean data is already ready and __getitem__ fxn will be fast during training 
          '''
          self.fgbg_data = self.load_data(zf_dict["fgbg"], self.fgbg_transform, kind="fgbg")
          self.mask_data = self.load_data(zf_dict["mask"], self.mask_transform, kind="mask")
          self.depth_data = self.load_data(zf_dict["depth"], self.depth_transform, kind="depth")
          self.bg_data = self.load_bg_data(zf_dict["bg"], self.bg_transform, kind="bg")

      def __len__(self):
          return len(self.im_files)

      def __getitem__(self, idx):

          if(torch.is_tensor(idx)):
            idx = idx.tolist()
          
          # format fgxxx_bgxxx_xx.jpg
          filename = self.im_files[idx]

          # for fgbg, mask and depth all its image data are loaded on same index location
          im_fgbg = self.fgbg_data[idx]
          im_mask = self.mask_data[idx]
          im_depth = self.depth_data[idx]
          
          # load bg data
          bg_idx = np.uint8(filename.split("_")[1][2:]) # get bg num from the file name
          im_bg = self.bg_data[bg_idx-1]

          sample = {"bg": im_bg, "fgbg": im_fgbg, "mask": im_mask, "depth": im_depth}
          return sample

      def read_image_from_zip(self, zf, filename):
          data = zf.read(filename)
          dataEnc = BytesIO(data)
          img = Image.open(dataEnc)
          del data
          del dataEnc
          return img

      def load_data(self, zf, transform, kind):
          load_images = []
          pbar = tqdm(self.im_files)
          for file in pbar:
            im = self.read_image_from_zip(zf, file)
            if transform is not None:
              im = transform(im)
            load_images.append(im)
            pbar.set_description(desc= f'Loading {kind} Data: ')
          gc.collect() # free unused memory
          return load_images

      '''
      gb have files with names as img_xxx.jpg(001 to 100)
      currently all the 100 bg images ae loaded so that we can directly access bg image throuhg its index number(1-100)
      later we can optimize it to load only relevent number of images..
      right max over head is instead of loading 100 images we are loading 200 (100 each for train and test data set, just to reduce time complexcity)
      '''
      def load_bg_data(self, zf, transform, kind):
          load_images = []
          pbar = tqdm(np.arange(1,101))
          for idx in pbar:
            filename = f'img_{idx:03d}.jpg'
            im = self.read_image_from_zip(zf, filename)
            if transform is not None:
              im = transform(im)
            load_images.append(im)
            pbar.set_description(desc= f'Loading {kind} Data: ')
          return load_images

      def get_count(self):
          ds_cnt = {"bg": len(self.bg_data),            
                    "fg_bg": len(self.fgbg_data),
                    "fg_bg_mask": len(self.mask_data),
                    "fg_bg_depth": len(self.depth_data)}
          return ds_cnt

'''
1. Dataset to read images from folder
2. this will slow the training as every time getitem is called image is read from file system
3. as all images are not loaded at once in RAM, so suitable to training large dataset

Creating custom dataset class by inherting torch.utils.data.Dataset and overide following methods:
1. __len__ : returns the size of the dataset
2. __getitem__: to support indexing such that dataset[i] can be used to get the sample

referenced material: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
class DepthMapDatasetFolder(Dataset):
      """ Depth Map dataset"""
      def __init__(self, ds_folder_dict, im_files, bg_transform=None, fgbg_transform=None, mask_transform=None, depth_transform=None):
          """
          Args:
          img_folders (dict): items names : "bg", "fgbg", "mask" and "depth"
          im_files: image filename list for the dataset(file name format: fg001_bg001_10.jpg)
          xxxx_transform (callable, optional): optional transform to be applied on respective image kind 
          """
          self.ds_folder_dict =  ds_folder_dict

          self.bg_transform = bg_transform
          self.fgbg_transform = fgbg_transform
          self.mask_transform = mask_transform
          self.depth_transform = depth_transform

          #fgbg, mask and depth retain the same file names. bg file name to be retrieve from this image file name
          self.im_files = im_files 

      def __len__(self):
          return len(self.im_files)

      def __getitem__(self, idx):

          if(torch.is_tensor(idx)):
            idx = idx.tolist()

          im_fgbg = self.load_data(f'{self.ds_folder_dict["fgbg"]}/{self.im_files[idx]}', self.fgbg_transform)
          im_mask = self.load_data(f'{self.ds_folder_dict["mask"]}/{self.im_files[idx]}', self.mask_transform)
          im_depth = self.load_data(f'{self.ds_folder_dict["depth"]}/{self.im_files[idx]}', self.depth_transform)
          
          # load bg data: read gb num from : format fgxxx_bgxxx_xx.jpg
          #bg_num = np.uint8(self.im_files[idx].split("_")[1][2:]) # get bg num from the file name
          #im_bg = self.load_data(f'{self.ds_folder_dict["bg"]}/img_{bg_num:03d}.jpg', self.depth_transform)
          
          bg_num = self.im_files[idx].split("_")[1][2:] # get bg num from the file name
          im_bg = self.load_data(f'{self.ds_folder_dict["bg"]}/img_{bg_num}.jpg', self.bg_transform)

          sample = {"bg": im_bg, "fgbg": im_fgbg, "mask": im_mask, "depth": im_depth}
          return sample

      def load_data(self, filename, transform):
          im = Image.open(filename)
          if transform is not None:
              im = transform(im)
          return im
          
class DepthMapDatasetFxn():
      def __init__(self):
          self.dum = 0

      def get_random_filelist(self, im_files):
          idx_list = np.arange(len(im_files))
          np.random.shuffle(idx_list)
          shuffled_im_files = [im_files[idx] for idx in idx_list]
          return shuffled_im_files

      def train_test_split(self, im_files, test_size=0.3):
          idx_list = np.arange(len(im_files))
          np.random.shuffle(idx_list)
          shuffle_im_files = [im_files[idx] for idx in idx_list]
          n_train = int(np.round((1-test_size)*len(im_files)))
          return shuffle_im_files[:n_train], shuffle_im_files[n_train:]

      def save_img_filenames(self, img_name_list, filename):
          # open file and write the content
          with open(filename, 'w') as filehandle:
            filehandle.writelines([f'{name}\n' for name in img_name_list])

      def read_img_filenames(self, filename):
          img_name_list = []
          # open file and read the content in a list
          with open(filename, 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                img_name = line[:-1] # remove linebreak which is the last character of the string
                img_name_list.append(img_name) # add item to the list
          return img_name_list