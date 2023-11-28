import os
import sys
import argparse
import matplotlib.pyplot as plt
import cv2 


fold = 1
src_path = f'object-detection/benchmarking/datasets/NuCLS/folds/fold_{fold}'
img_path = f'object-detection/benchmarking/datasets/NuCLS/images'
dst_path = f'object-detection/benchmarking/datasets/NuCLS/folds/fold_{fold}/rpn'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)


from torch import nn
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np


images = os.listdir(img_path)
try:
    images.remove('.DS_Store')
except:
    pass

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True)
