from turtle import colormode
import pandas as pd
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import argparse

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model



# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer

# args
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, required=True, default=1)
parser.add_argument('--version', type=str, required=True, default='detectron')
parser.add_argument('--max_iters', type=int, required=True, default=100)
parser.add_argument('--batch_size', type=int, default=8, required=False)
parser.add_argument('--lr', type=float, default=0.00025, required=False)
parser.add_argument('--gpu', type=str, default='0', required=False)
parser.add_argument('--num_workers', type=int, required=False, default=4)
parser.add_argument('--log', type=str, required=False, default=True)
parser.add_argument('--save', type=str, required=False, default=True)
parser.add_argument('--project', type=str, required=True, default='capstone')
parser.add_argument('--name', type=str, required=True, default='experiment')
parser.add_argument('--data_path', type=str, default=os.curdir, required=True)
parser.add_argument('--image_dir', type=str, default='images', required=True)
parser.add_argument('--api_key', type=int, required=False, default=None)
parser.add_argument('--config_info', type=str, required=True, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

parse = parser.parse_args()


fold = parse.fold
version = parse.version
max_iters = parse.max_iters
batch_size = parse.batch_size
lr = parse.lr
gpu = parse.gpu
num_workers = parse.num_workers
log = parse.log
save = parse.save
project = parse.project
name = parse.name
data_path = parse.data_path
image_dir = parse.image_dir
api = parse.api_key
config_info = parse.config_info
'''
example run in multiline
python detectron_train_v1.py \
--fold 1 \
--version detectron \
--max_iters 100 \
--batch_size 8 \
--lr 0.00025 \
--gpu 0 \
--num_workers 4 \
--log True \
--save True \
--project capstone \
--name faster_rcnn_R_50_FPN_3x \

single line
python detectron_train_v1.py --fold 1 --version detectron  --max_iters 100 --batch_size 8 --lr 0.00025 --gpu 0 --num_workers 4 --log True --save True --project capstone-project --name faster_rcnn_R_50_FPN_3x --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/EvalSet/detectron/master/npsave --image_dir /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/EvalSet/detectron/master/images --api_key 0 --config_info COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
'''

# print summary
print('Fold: ', fold)
print('Version: ', version)
print('max_iters: ', max_iters)
print('Batch Size: ', batch_size)
print('Learning Rate: ', lr)
print('GPU: ', gpu)
print('Num Workers: ', num_workers)
print('Log: ', log)
print('Save: ', save)
print('Project: ', project)
print('Name: ', name)

# experiment = Experiment(
#     workspace="mraoaakash",
#     project_name=project,
# )
# experiment.log_parameters({
#     'fold': fold,
#     'version': version,
#     'model': model,
#     'max_iters': max_iters,
#     'batch_size': batch_size,
#     'gpu': gpu,
#     'num_workers': num_workers,
#     'log': log,
#     'save': save,
#     'project': project,
#     'name': name,
#     'data_path': data_path,
#     'image_dir': image_dir,
# })

def data_train():
    data = np.load(os.path.join(data_path,f'fold_{fold}_train.npy'), allow_pickle=True)
    data = list(data)
    print(len(data))
    return data

def data_val():
    data = np.load(os.path.join(data_path,f'fold_{fold}_val.npy'), allow_pickle=True)
    data = list(data)
    print(len(data))
    return data

def data_test():
    data = np.load(os.path.join(data_path,f'test.npy'), allow_pickle=True)
    data = list(data)
    print(len(data))
    return data



DatasetCatalog.register(f'fold_{fold}_train', data_train)
data = DatasetCatalog.get(f'fold_{fold}_train')
MetadataCatalog.get(f'fold_{fold}_train').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
MetadataCatalog.get(f'fold_{fold}_train').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]


DatasetCatalog.register(f'fold_{fold}_val', data_val)
data = DatasetCatalog.get(f'fold_{fold}_val')
MetadataCatalog.get(f'fold_{fold}_val').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
MetadataCatalog.get(f'fold_{fold}_val').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]

DatasetCatalog.register(f'test', data_test)
data = DatasetCatalog.get(f'test')
MetadataCatalog.get(f'test').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
MetadataCatalog.get(f'test').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_info))
cfg.DATASETS.TRAIN = (f'fold_{fold}_train',)
cfg.DATASETS.TEST = (f'fold_{fold}_val',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_info)  
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = max_iters
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
cfg.OUTPUT_DIR = os.path.join(data_path, f'fold_{fold}_{name}_{version}')


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = (f'test',)
predictor = DefaultPredictor(cfg)
predictions = []
pred_save_path = os.path.join(cfg.OUTPUT_DIR, 'predictions')
for d in data_test():
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    predictions.append(outputs)
    v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get(f'test'), 
                    scale=0.8,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.savefig(os.path.join(pred_save_path, d['file_name'].split('/')[-1]), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()
    break
print('Predictions: ', predictions)



# experiment.end()