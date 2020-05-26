import time
import numpy as np
import shutil
from tqdm import tqdm
import pandas as pd
import os

'''
    self.col_names=['src_type', 'image', "path", 'class_id', 'class_num', 'class_name', 'x','y','h','w']

    src_type: whether image is used from train or val folder
    image: image file name
    path: where the image is residing in source folder
    class_id: class-id as per wnids.txt for this image
    class_num: which num(0~199) for this image
    class_name: class category as per words.txt
    'x','y','h','w': box attributes
'''

class TinyImagenetDataset():
    def __init__(self, base_path):
        self.base_path = base_path
        self.id_dict = self.get_id_dictionary()
        self.class_name = self.get_class_to_id_dict()      
        self.class_name_list = self.get_classes()
        self.col_names=['src_type', 'image', "path", 'class_id', 'class_num', 'class_name', 'x','y','h','w']
    
    # Mapping class ids to class numbers from 0~199
    def get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open(self.base_path + '/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict

    # reading class name for the class-ids
    def get_class_to_id_dict(self):
        id_dict = self.get_id_dictionary()
        all_classes = {}
        result = {}
        for i, line in enumerate(open(self.base_path + '/words.txt', 'r')):
            line = line.replace('\n', '')
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in self.id_dict.items():
            result[value] = (key, all_classes[key])      
        return result
    
    def get_classes(self):
        class_list = []
        class_list += [value[1].split(',')[0] for key, value in self.class_name.items()]
        return class_list

    def shuffle_dataset(self, dataset):
        size = len(dataset)
        idx = np.arange(size)
        np.random.shuffle(idx)
        return dataset[idx]
    
    def split_dataset(self, dataset, split_size=0.70):
        df_dataset = pd.DataFrame(dataset,columns=self.col_names)
        train_idx = []
        val_idx = []

        # to balance the data split is done for each classess
        for key, value in self.id_dict.items():
            idx = df_dataset[df_dataset.class_id==key].index
            train_size = (np.uint32)(len(idx)*split_size) 
            train_idx.extend(idx[:train_size])
            val_idx.extend(idx[train_size:])

        train_data = df_dataset.values[train_idx]
        val_data = df_dataset.values[val_idx]
        return train_data, val_data
    
    # combining 100,000 inages of train and 10,000 image of val dataset from tiny-imagenet-200 folder
    # reading the box attributes for each images from their box definition files
    def prepare_dataset(self):
        train_dir = os.path.join(self.base_path, "train") 
        dataset = []
        
        # collecting image data information from train folder
        #key is class_id and value is number
        for key, value in self.id_dict.items():
            class_dir = os.path.join(train_dir, key)
            box_file = os.path.join(class_dir, "{}_boxes.txt".format(key))
            for line in open(box_file):
                line = line.replace('\n', '')
                img_name,x,y,h,w = line.split('\t')
                img_path = "train/{}/images/{}".format(key,img_name)
                # store as per self.cocol_names
                dataset.append(["train",img_name,img_path,key,value, self.class_name[value][1], x,y,h,w])
        
        # collecting image data information from val folder
        val_dir = os.path.join(self.base_path, "val") 
        val_annotation_file = os.path.join(val_dir,"val_annotations.txt")
        for line in open(val_dir + '/val_annotations.txt'):
            line = line.replace('\n', '')
            img_name,class_id,x,y,h,w = line.split('\t')
            img_path = "val/images/{}".format(img_name)
            # store as per self.cocol_names
            dataset.append(["val",img_name,img_path,class_id,self.id_dict[class_id],self.class_name[value][1], x,y,h,w])
                
        return np.array(dataset)
    
    def create_imagefolder_old(self, target_base_dir):
        
        if os.path.exists(target_base_dir):
            print("Dataset is already ready and existing..")
            return
        
        # collect and combine entire data from train and val source folder
        dataset = self.prepare_dataset()
        
        # shiffle and split dataset into train and val (70:30)
        train, val = self.split_dataset(dataset, split_size=0.70)

        # persist train and val data information so that we no need to creare this dataset again and again in multiple notebook session            
        target_train_dir = os.path.join(target_base_dir,"train")
        target_val_dir = os.path.join(target_base_dir,"val")
              
        # create train and val target directories
        os.mkdir(target_base_dir)
        os.mkdir(target_train_dir)
        os.mkdir(target_val_dir)

        # save train and val dataset records for future use
        df_train = pd.DataFrame(train,columns=self.col_names)
        df_val = pd.DataFrame(val,columns=self.col_names)
        df_train.to_csv(os.path.join(target_base_dir,"train_data.csv"))
        df_val.to_csv(os.path.join(target_base_dir,"val_data.csv"))
        
        startTime = time.time()
        train_pbar = tqdm(train[:1000])
        for index, item in enumerate(train_pbar):
            # make sure respective class folder exists in dst
            dst = os.path.join(target_train_dir,item[4])
            if not os.path.exists(dst):
                os.mkdir(dst)
                
            src = os.path.join(self.base_path, item[2])
            shutil.copy(src,dst)
            #train_pbar.set_description(desc= f'Creating/Copying training dataset: {index+1}/{len(train)}')
            train_pbar.set_description(desc= f'Creating/Copying training dataset')
                    
        val_pbar = tqdm(val[:300])
        for item in val_pbar:
            # make sure respective class folder exists in dst
            dst = os.path.join(target_val_dir,item[4])
            if not os.path.exists(dst):
                os.mkdir(dst)
                
            src = os.path.join(self.base_path, item[2])
            shutil.copy(src,dst)
            #val_pbar.set_description(desc= f'Creating/Copying validation dataset: {index+1}/{len(val)}')
            val_pbar.set_description(desc= f'Creating/Copying validation dataset')
            
        endTime = time.time()
        copy_dur = endTime - startTime;
        print("Execution time: %0.2f minutes" %(copy_dur/60))
        
        print("Total dataset: {}, Training size: {}, Validation size: {}".format(len(dataset),len(train), len(val)))
    
    def create_imagefolder(self, target_base_dir):
        
        if os.path.exists(target_base_dir):
            print("Dataset is already ready and existing..")
            return

        # collect and combine entire data from train and val source folder
        dataset = self.prepare_dataset()
        
        # split dataset into train and val (70:30)
        train, val = self.split_dataset(dataset, split_size=0.70)

        # persist train and val data information so that we no need to creare this dataset again and again in multiple notebook session            
        target_train_dir = os.path.join(target_base_dir,"train")
        target_val_dir = os.path.join(target_base_dir,"val")
              
        # create train and val target directories
        os.mkdir(target_base_dir)
        os.mkdir(target_train_dir)
        os.mkdir(target_val_dir)

        # save train and val dataset records for future use
        df_train = pd.DataFrame(train,columns=self.col_names)
        df_val = pd.DataFrame(val,columns=self.col_names)
        df_train.to_csv(os.path.join(target_base_dir,"train_data.csv"))
        df_val.to_csv(os.path.join(target_base_dir,"val_data.csv"))
        
        startTime = time.time()
        train_pbar = tqdm(self.id_dict.items())
        for dx, (key, value) in enumerate(train_pbar):
            df_class = df_train[df_train.class_id==key]
            
            # make sure respective class folder exists in dst
            dst = os.path.join(target_train_dir,"{}".format(value))
            if not os.path.exists(dst):
                os.mkdir(dst)      
                            
            for item in df_class.values:
                src = os.path.join(self.base_path, item[2])
                shutil.copy(src,dst)
            
            #train_pbar.set_description(desc= f'Creating/Copying training dataset: {index+1}/{len(train)}')
            train_pbar.set_description(desc= f'Creating/Copying training dataset for class({key},{value})')
            
        
        val_pbar = tqdm(self.id_dict.items())
        for idx, (key, value) in enumerate(val_pbar):
            df_class = df_val[df_val.class_id==key]
            
            # make sure respective class folder exists in dst
            dst = os.path.join(target_val_dir,"{}".format(value))
            if not os.path.exists(dst):
                os.mkdir(dst)      
                            
            for item in df_class.values:
                src = os.path.join(self.base_path, item[2])
                shutil.copy(src,dst)
            
            val_pbar.set_description(desc= f'Creating/Copying validation dataset for class({key},{value})')
            
        endTime = time.time()
        copy_dur = endTime - startTime;
        print("Execution time: %0.2f minutes" %(copy_dur/60))
        
        print("Total dataset: {}, Training size: {}, Validation size: {}".format(len(dataset),len(train), len(val)))