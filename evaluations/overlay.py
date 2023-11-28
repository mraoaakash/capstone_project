import os 
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

gt_txt_path = ''
pred_txt_path = ''
img_path = ''

# read txt files
gt = pd.read_csv(gt_txt_path, sep=' ', header=None)
gt.columns = ['class', 'x1', 'y1', 'x2', 'y2', ]
pred = pd.read_csv(pred_txt_path, sep=' ', header=None)
pred.columns = ['class', 'score', 'x1', 'y1', 'x2', 'y2', ]

# read image
im = cv2.imread(img_path)

# plot ground truth
for i in range(len(gt)):
    x1, y1, x2, y2 = gt.loc[i, ['x1', 'y1', 'x2', 'y2']]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(im, gt.loc[i, 'class'], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# plot prediction
for i in range(len(pred)):
    x1, y1, x2, y2 = pred.loc[i, ['x1', 'y1', 'x2', 'y2']]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(im, pred.loc[i, 'class'], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# save image
cv2.imwrite('test.png', im)
