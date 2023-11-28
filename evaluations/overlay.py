import os 
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

base_path = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/converted'
# img = 'JP.1_#_E_#_TCGA-A1-A0SP-01Z-00-DX1_id-5e83b16eddda5f83987d646e_left-8066_top-55982_bottom-56288_right-8365'
img = 'JP.1_#_E_#_TCGA-A1-A0SP-01Z-00-DX1_id-5e83b16eddda5f83987d646e_left-9067_top-55759_bottom-56058_right-9369'

gt_txt_path = f'{base_path}/ground_truth/annotations/{img}.txt'
pred_txt_path = f'{base_path}/faster_rcnn_R_50_C4_1x_fold_1/{img}.txt'
img_path = f'{base_path}/ground_truth/images/{img}.png'

# read txt files
gt = pd.read_csv(gt_txt_path, sep=' ', header=None)
gt.columns = ['class', 'x1', 'y1', 'x2', 'y2', ]
pred = pd.read_csv(pred_txt_path, sep=' ', header=None)
pred.columns = ['class', 'score', 'x1', 'y1', 'x2', 'y2', ]
print(pred)

# read image
im = cv2.imread(img_path)
# scaling image down to 0.8
im = cv2.resize(im, (0, 0), fx=0.8, fy=0.8)


# plt.figure(figsize=(10, 10))
plt.imshow(im)
# gt
for i in range(len(gt)):
    x1 = gt.iloc[i]['x1']
    y1 = gt.iloc[i]['y1']
    x2 = gt.iloc[i]['x2']
    y2 = gt.iloc[i]['y2']
    # convert to int
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
# pred
for i in range(len(pred)):
    # pred was scaled to 0.8
    x1 = pred.iloc[i]['x1']
    y1 = pred.iloc[i]['y1']
    x2 = pred.iloc[i]['x2']
    y2 = pred.iloc[i]['y2']
    # convert to int
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
plt.imshow(im)
plt.show()


# save image
cv2.imwrite('test.png', im)
