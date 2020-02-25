# Session 6 - Regularization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_w2xDoNwCd-b1tFScs-0RamsgUP7Lmfz)

The goal of this assignment is to apply L1 and L2 regularization on the final model from the [previous](../S5/) session and plot the changes in validation loss and accuracy obtained during model training in the following scenarios:

1. Without L1 and L2 regularization
2. With L1 regularization
3. With L2 regularization
4. With L1 and L2 regularization

After model training, we need to display 25 misclassified images for L1 and L2 models.

### Parameters and Hyperparameters

- Kernel Size: 3x3
- Loss Function: Negative Log Likelihood
- Optimizer: SGD
- Dropout Rate: 0.01
- Batch Size: 64
- Learning Rate: 0.01
- **L1 Factor:** 0.001
- **L2 Factor:** 0.0001

## Results

### Change in Validation Loss and Accuracy

<img src="i2.png" width="600px">
<img src="i3.png" width="600px">

## Misclassified Images

### Without L1 and L2 Regularization

![plain](i4.png)

### With L1 Regularization

![plain](i5.png)

### With L2 Regularization

![plain](i6.png)

### With L1 and L2 Regularization

![plain](i7.png)


