import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import Image, display

def gradcamof(net, img, classes):
  netx = Res18(net)
  netx.eval()
  
  fig, axes = plt.subplots(nrows=1, ncols=3)
  # get the most likely prediction of the model
  pred = netx(img.cuda())
  from torchvision.utils import save_image
  imx = img[0]
  save_image(imx, 'img1.png')
  class_pred = int(np.array(pred.cpu().argmax(dim=1)))
  imshow(torchvision.utils.make_grid(img),axes[0])
  print(classes[class_pred])

  # draw the heatmap
  heatmap = getheatmap(pred, class_pred, netx, img)
  axes[1].matshow(heatmap.squeeze())

  imx = cv2.imread("./img1.png")
  imx = cv2.cvtColor(imx, cv2.COLOR_BGR2RGB)
 # plt.imshow(imx, cmap='gray', interpolation='bicubic')
  superposeimage(heatmap, imx)

  imx = cv2.imread('./map.jpg')
  imx = cv2.cvtColor(imx, cv2.COLOR_BGR2RGB)
  axes[2].imshow(imx, cmap='gray', interpolation='bicubic')