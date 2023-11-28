from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
import numpy as np

import os 
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

data_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds'

def data_test():
    data = np.load(os.path.join(data_path,"final_test", f'test.npy' ), allow_pickle=True))
    data = list(data)
    print(len(data))
    return data

DatasetCatalog.register(f'test', data_test)
data = DatasetCatalog.get(f'test')
MetadataCatalog.get(f'test').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
MetadataCatalog.get(f'test').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]

MODEL_PATH = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/outputs/nucls_faster_rcnn_R_50_FPN_3x'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(MODEL_PATH, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.DATASETS.TEST = (f'test',)
predictor = DefaultPredictor(cfg)
