import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import cv2
import shutil

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
def run_pred_level():
    for model in models:
        if 'rpn' in model:
            continue
        prediction = np.load(os.path.join(main_path, model, 'predictions.npy'), allow_pickle=True)
        pred_save_path = os.path.join(outpath, model)
        if not os.path.exists(pred_save_path):
            os.makedirs(pred_save_path)

        i = 0
        for d in data_test():
            boxes_np = []
            boxes = []
            classes = []
            scpres = []
            box = []
            
            print(d['image_id'])
            print(model)
            im = cv2.imread(d["file_name"])
            boxes = prediction[i]['instances'].pred_boxes
            classes = prediction[i]['instances'].pred_classes.cpu().numpy()
            scores = prediction[i]['instances'].scores.cpu().numpy()
            for j in boxes.__iter__():
                box = j.cpu().numpy()
                boxes_np.append(box)
            boxes_np = np.array(boxes_np).astype(np.uint8)
            print(boxes_np.shape)
            print(scores.shape)
            print(classes.shape)
            with open(os.path.join(pred_save_path, f'{d["image_id"]}.txt'), 'w+') as f:
                for k in range(len(boxes_np)):
                    f.write(f'{classes[k]} {scores[k]} {boxes_np[k][0]} {boxes_np[k][1]} {boxes_np[k][2]} {boxes_np[k][3]}\n')
            i+=1
        
test_annnot_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/final_test/test.npy'
test_gt_save_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/converted/ground_truth'
ground_truth_img = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/converted/ground_truth/images'
ground_truth_annot = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/converted/ground_truth/annotations'

if not os.path.isdir(ground_truth_annot):
    os.makedirs(ground_truth_annot)
if not os.path.isdir(ground_truth_img):
    os.makedirs(ground_truth_img)

if not os.path.exists(test_gt_save_path):
    os.makedirs(test_gt_save_path)
gt = np.load(test_annnot_path, allow_pickle=True)
for annot in gt:
    image_id = annot['image_id']
    file_name = annot['file_name']
    print(image_id)
    annotations = annot['annotations']
    classes = []
    cofidences = []
    boxes = []
    for annotation in annotations:
        # print(annotation)
        class_id = annotation['category_id']
        confidence = 1.0
        box = annotation['bbox']
        boxes.append(box)
        classes.append(class_id)
        cofidences.append(confidence)
    boxes = np.array(boxes).astype(np.uint8)
    classes = np.array(classes)
    cofidences = np.array(cofidences)
    print(boxes.shape)
    print(classes.shape)
    print(cofidences.shape)
    im_width, im_height = cv2.imread(file_name).shape[:2]
    shutil.copy(file_name, os.path.join(ground_truth_img, f'{image_id}.png'))
    with open(os.path.join(ground_truth_annot, f'{image_id}.txt'), 'w+') as f:
        for k in range(len(boxes)):
            x1 = boxes[k][0]
            y1 = boxes[k][1]
            x2 = boxes[k][2]
            y2 = boxes[k][3]
            # computing the center of the box
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            # computing the width and height of the box
            w = x2 - x1
            h = y2 - y1

            # adjusting the coordinates based on the size of the image
            x_center = x_center 
            y_center = y_center 
            w = w 
            h = h 

            f.write(f'{classes[k]} {x_center} {y_center} {w} {h}\n')

# run_pred_level()