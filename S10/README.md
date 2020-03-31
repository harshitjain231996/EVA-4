# Session 10 - Learning Rates

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Awbpu4hVinMWWrDlkSDL5ojYSHvCvxGE)

The model reaches a maximum accuracy of **91.52%** on CIFAR-10 dataset using **ResNet-18** model.

**LR Finder and Reduce LR on Plateau** was implemented for model training.

The model uses the library **TensorNet** to train the model. The library can be installed by running the following command  
`pip install torch-tensornet==0.0.7`  

### Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss (combination of `nn.LogSoftmax` and `nn.NLLLoss`)
- LR Finder
  - Start LR: 1e-7
  - End LR: 5
  - Number of iterations: 400
- Optimizer: SGD
  - Momentum: 0.9
  - Learning Rate: 0.013 (Obtained from LR Finder)
- Reduce LR on Plateau
  - Decay factor: 0.1
  - Patience: 2
  - Min LR: 1e-4
- Batch Size: 64
- Epochs: 50

### Data Augmentation

The following data augmentation techniques were applied to the dataset during training:

- Horizontal Flip
- Rotation
- CutOut

## GradCAM

Some of the examples of GradCAM on misclassified images is shown below:

![grad_cam](3s1.png)

## Change in Training and Validation Accuracy

<img src="s10.png" width="450px">

