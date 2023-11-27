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

import sys



# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer

fold = 1

# data_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/fold_1/'
# config_info = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# max_iters = 1500
# name  = 'exp_1_iters_1'
# project = 'capstone-project' 
# version = '1'


def train_detectron2(data_path, config_info, max_iters, name, project, fold, version):
    def data_train():
        data = np.load(os.path.join(data_path, f'fold_{fold}', 'train.npy'), allow_pickle=True)
        data = list(data)
        print(len(data))
        return data

    def data_val():
        data = np.load(os.path.join(data_path, f'fold_{fold}', f'test.npy'), allow_pickle=True)
        data = list(data)
        print(len(data))
        return data

    def data_test():
        data = np.load(os.path.join(data_path,"final_test",f'test.npy'), allow_pickle=True)
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

    dataset_dicts = data_train()
    metadata = MetadataCatalog.get(f'fold_{fold}_train')

    i = 0
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        print(img.shape)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        im = vis.get_image()[:, :, ::-1]
        print(im.shape)
        plt.imshow(im)
        plt.savefig(f"/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/outputs/train_imgs/train_{i}.png",dpi=300)
        plt.close()
        # cv2.waitKey(0)
        # break
        i+=1

    plt.show()

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
    cfg.OUTPUT_DIR = os.path.join(data_path, f'outputs/{name}')


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
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
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
    print('Predictions: ', predictions)
    predictions = np.array(predictions)
    np.save(os.path.join(cfg.OUTPUT_DIR, 'predictions.npy'), predictions)


    # experiment.end()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--data_path', type=str, default=None, help='path to data')
    argparse.add_argument('--config_info', type=str, default=None, help='config info')
    argparse.add_argument('--max_iters', type=int, default=None, help='max iters')
    argparse.add_argument('--name', type=str, default=None, help='name')
    argparse.add_argument('--project', type=str, default=None, help='project')
    argparse.add_argument('--fold', type=str, default=None, help='version')
    argparse.add_argument('--version', type=str, default=None, help='version')
    args = argparse.parse_args()
    train_detectron2(args.data_path, args.config_info, args.max_iters, args.name, args.project, args.fold, args.version)
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_3x_fold_3 --project capstone-project --fold 3 --version 1

    # fast_rcnn_R_50_FPN_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml" --max_iters 2000 --name fast_rcnn_R_50_FPN_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml" --max_iters 2000 --name fast_rcnn_R_50_FPN_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml" --max_iters 2000 --name fast_rcnn_R_50_FPN_1x_fold_3 --project capstone-project --fold 3 --version 1

    # faster_rcnn_R_101_C4_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_C4_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_C4_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds --config_info "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_C4_3x_fold_3 --project capstone-project --fold 3 --version 1


    # faster_rcnn_R_101_DC5_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_DC5_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_DC5_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_DC5_3x_fold_3 --project capstone-project --fold 3 --version 1
    
    # faster_rcnn_R_101_FPN_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_FPN_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_FPN_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_101_FPN_3x_fold_3 --project capstone-project --fold 3 --version 1


    # faster_rcnn_R_50_C4_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_C4_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_C4_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_C4_1x_fold_3 --project capstone-project --fold 3 --version 1


    # faster_rcnn_R_50_C4_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_C4_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_C4_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_C4_3x_fold_3 --project capstone-project --fold 3 --version 1
    
    
    # faster_rcnn_R_50_DC5_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_DC5_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_DC5_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_DC5_1x_fold_3 --project capstone-project --fold 3 --version 1
    

    # faster_rcnn_R_50_DC5_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_DC5_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_DC5_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_DC5_3x_fold_3 --project capstone-project --fold 3 --version 1
    

    # faster_rcnn_R_50_FPN_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_1x_fold_3 --project capstone-project --fold 3 --version 1
    


    # faster_rcnn_R_50_FPN_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_R_50_FPN_3x_fold_3 --project capstone-project --fold 3 --version 1
    


    # faster_rcnn_X_101_32x8d_FPN_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_X_101_32x8d_FPN_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_X_101_32x8d_FPN_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" --max_iters 2000 --name faster_rcnn_X_101_32x8d_FPN_3x_fold_3 --project capstone-project --fold 3 --version 1
    


    # fcos_R_50_FPN_1x
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/fcos_R_50_FPN_1x.yaml" --max_iters 2000 --name fcos_R_50_FPN_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/fcos_R_50_FPN_1x.yaml" --max_iters 2000 --name fcos_R_50_FPN_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/fcos_R_50_FPN_1x.yaml" --max_iters 2000 --name fcos_R_50_FPN_1x_fold_3 --project capstone-project --fold 3 --version 1
    


    # retinanet_R_101_FPN_3x
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_101_FPN_3x.yaml" --max_iters 2000 --name retinanet_R_101_FPN_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_101_FPN_3x.yaml" --max_iters 2000 --name retinanet_R_101_FPN_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_101_FPN_3x.yaml" --max_iters 2000 --name retinanet_R_101_FPN_3x_fold_3 --project capstone-project --fold 3 --version 1
    
 

    # retinanet_R_50_FPN_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_50_FPN_1x.yaml" --max_iters 2000 --name retinanet_R_50_FPN_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_50_FPN_1x.yaml" --max_iters 2000 --name retinanet_R_50_FPN_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_50_FPN_1x.yaml" --max_iters 2000 --name retinanet_R_50_FPN_1x_fold_3 --project capstone-project --fold 3 --version 1
    


    # retinanet_R_50_FPN_3x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_50_FPN_3x.yaml" --max_iters 2000 --name retinanet_R_50_FPN_3x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_50_FPN_3x.yaml" --max_iters 2000 --name retinanet_R_50_FPN_3x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/retinanet_R_50_FPN_3x.yaml" --max_iters 2000 --name retinanet_R_50_FPN_3x_fold_3 --project capstone-project --fold 3 --version 1
    

    # rpn_R_50_C4_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/rpn_R_50_C4_1x.yaml" --max_iters 2000 --name rpn_R_50_C4_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/rpn_R_50_C4_1x.yaml" --max_iters 2000 --name rpn_R_50_C4_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/rpn_R_50_C4_1x.yaml" --max_iters 2000 --name rpn_R_50_C4_1x_fold_3 --project capstone-project --fold 3 --version 1
    


    # rpn_R_50_FPN_1x 
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/rpn_R_50_FPN_1x.yaml" --max_iters 2000 --name rpn_R_50_FPN_1x_fold_1 --project capstone-project --fold 1 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/rpn_R_50_FPN_1x.yaml" --max_iters 2000 --name rpn_R_50_FPN_1x_fold_2 --project capstone-project --fold 2 --version 1
    # python3 train_detectron2.py --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds  --config_info "COCO-Detection/rpn_R_50_FPN_1x.yaml" --max_iters 2000 --name rpn_R_50_FPN_1x_fold_3 --project capstone-project --fold 3 --version 1