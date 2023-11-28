import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer


main_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/outputs'
outpath = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/converted'

data_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds'

if not os.path.exists(outpath):
    os.makedirs(outpath)

def data_test():
        data = np.load(os.path.join(data_path,"final_test",f'test.npy'), allow_pickle=True)
        data = list(data)
        print(len(data))
        return data
DatasetCatalog.register(f'test', data_test)
data = DatasetCatalog.get(f'test')
MetadataCatalog.get(f'test').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
MetadataCatalog.get(f'test').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]

models = os.listdir(main_path)

for model in models:
    prediction = np.load(os.path.join(main_path, model, 'predictions.npy'), allow_pickle=True)
    pred_save_path = os.path.join(outpath, model)
    if not os.path.exists(pred_save_path):
         os.makedirs(pred_save_path)

    i = 0
    for d in data_test():
        print(d['image_id'])
        im = cv2.imread(d["file_name"])
        boxes = prediction[i]['instances'].pred_boxes
        classes = prediction[i]['instances'].pred_classes.cpu().numpy()
        scores = prediction[i]['instances'].scores.cpu().numpy()
        boxes_np = []
        for i in boxes.__iter__():
            box = i.cpu().numpy()
            print(box)
            boxes_np.append(box)
        boxes_np = np.array(boxes_np)
        print(boxes_np.shape)
        print(scores)
        print(classes)
        
        
        i+=1
        break
    break