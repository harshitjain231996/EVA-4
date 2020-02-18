# Session 5 - MNIST 99.4% Test Accuracy with less than 10,000 Parameters

The desired **test accuracy** expected in this assignment is **99.4%** on the MNIST test dataset with a model having following constraints/conditions:

- The given accuracy should take less than or equal to 15 epochs
- The total number of parameters to be used should be less than 10,000

The desired target should be achieved in a minimum of 5 steps.

### Step 1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17ZJeAAxgsAL1q3mP2v3NzE3HtiQyt5UD)

#### Target

- Get the set-up right
- Visualize dataset statistics and samples
- Set the data transforms
- Set train and test data and create the data loader
- Create an initial working model architecture
- Set training and test loop

#### Result

- Parameters: 15,530
- Best Training Accuracy: 99.23
- Best Test Accuracy: 99.12

### Step 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hrti39bpOgm5zArhid2FOI4QgIFX5jhQ)

#### Target

- Decreasing the number of model parameters

  - Reduce the number of kernels
  - Using 1x1 kernel before GAP
  - Adding Global Average Pooling (GAP)

- Decreasing Batch Size  
  Small batch size helps the model to escape any local minima

#### Result

- Parameters: 8,962
- Best Training Accuracy: 98.94
- Best Test Accuracy: 98.83

### Step 3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r0_0TkW_InmFO5bNhKo7BtURofH8hbQM)

#### Target

- Apply Batch Normalization to increase model accuracy

#### Result

- Parameters: 9,142
- Best Training Accuracy: 99.34
- Best Test Accuracy: 99.37

### Step 4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18sBCg8A11FIEygZqQ-2mN6Z7eN-pkId5)

#### Target

- Appling LR Scheduler  

#### Result

- Parameters: 9,142
- Best Training Accuracy: 99.40
- Best Test Accuracy: 99.38

### Step 5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dB_x8RvgsLiHT30qti2l-fxm-4q00DIh)

#### Target

- Apply Dropout
- Add Image Augmentation

#### Result

- Parameters: 9,142
- Best Training Accuracy: 99.14
- Best Test Accuracy: 99.45

### Parameters and Hyperparameters

- Kernel Size: 3x3
- Loss Function: Negative Log Likelihood
- Optimizer: SGD
- Dropout Rate: 0.01
- Batch Size: 64
- Learning Rate: 0.01

The model reached the test accuracy of **99.45%** after **10 epochs**.

## Project Setup

### On Local System

Install the required packages  
 `$ pip install -r requirements.txt`

### On Google Colab

Select Python 3 as the runtime type and GPU as the harware accelerator.

## Group Members

Harshit Jain
Sanjeev V Raichur
Prakash Upadhyay
