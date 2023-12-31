import os 
import cv2
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer

import matplotlib.pyplot as plt

fold = 1
name = f'faster_rcnn_R_50_C4_1x_fold_{fold}'
config_info = f'COCO-Detection/{name[:-7]}.yaml'

def bb_intersection_over_union(a,b):
    # compute IoU
    # a = gt
    # b = model_pred
    # print(a)
    # print(b)
    # print(a[0], a[1], a[2], a[3])
    # print(b[0], b[1], b[2], b[3])
    # print('---')
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    # print(xA, yA, xB, yB)
    # print('---')
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # print(interArea)
    # print('---')
    boxAArea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    # print(boxAArea)
    # print('---')
    boxBArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # print(boxBArea)
    # print('---')
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # print(iou)
    return iou

def get_boxes(d):
    boxes = []
    classes = []
    for i in range(len(d['annotations'])):
        boxes.append(d['annotations'][i]['bbox'])
        classes.append(d['annotations'][i]['category_id'])


    return boxes, classes
for fold in [1,2,3]:
    for model_info in ['faster_rcnn_R_50_C4_1x','faster_rcnn_R_50_C4_3x','faster_rcnn_R_50_DC5_1x','faster_rcnn_R_50_DC5_3x', 'faster_rcnn_R_101_C4_3x','faster_rcnn_R_101_DC5_3x']:
        basepath = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/outputs'
        gt_overlay_save = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/outputs/evaluations'
        model_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/outputs/{model_info}_fold_{fold}'
        # model_preds = '{basepath}/faster_rcnn_R_50_C4_1x_fold_1'


        model_preds = '{basepath}/{model_info}_fold_{fold}'
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_info))
        cfg.DATASETS.TRAIN = (f'fold_{fold}_train',)
        cfg.DATASETS.TEST = (f'fold_{fold}_val',)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_info)
        cfg.MODEL.LOAD_PROPOSALS = False
        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.STEPS = []        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
        cfg.OUTPUT_DIR = os.path.join(gt_overlay_save, f'{name}')
        cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
        cfg.DATASETS.TEST = (f'test',)

        predictor = DefaultPredictor(cfg)
        predictions = []
        pred_save_path = os.path.join(cfg.OUTPUT_DIR, 'predictions')
        if not os.path.exists(pred_save_path):
            os.makedirs(pred_save_path)

        data_path = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds'
        def data_test():
            data = np.load(os.path.join(data_path,"final_test",f'test.npy'), allow_pickle=True)
            data = list(data)
            print(len(data))
            return data

        for d in data_test():
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            print(outputs)
            predictions.append(outputs)
            v = Visualizer(im[:, :, ::-1],
                            metadata=MetadataCatalog.get(f'test'), 
                            scale=1.0,
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out = v.draw_dataset_dict(d)

            plt.imshow(out.get_image()[:, :, ::-1])
            plt.axis('off')
            # plt.show()
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()
            classes = outputs["instances"].pred_classes.cpu().numpy()
            print(boxes)
            print(scores)
            print(classes)

            gt_boxes, gt_classes = get_boxes(d)
            print(gt_boxes)
            print(gt_classes)

            boxes = np.array(boxes)
            scores = np.array(scores)
            classes = np.array(classes)
            gt_boxes = np.array(gt_boxes)
            gt_classes = np.array(gt_classes)

            # saving all the files
            image_level_save_path = os.path.join(pred_save_path, d['file_name'].split('/')[-1][:-4])
            if not os.path.exists(image_level_save_path):
                os.makedirs(image_level_save_path)
            np.save(os.path.join(image_level_save_path, d['file_name'].split('/')[-1][:-4] + '_boxes.npy'), boxes)
            np.save(os.path.join(image_level_save_path, d['file_name'].split('/')[-1][:-4] + '_scores.npy'), scores)
            np.save(os.path.join(image_level_save_path, d['file_name'].split('/')[-1][:-4] + '_classes.npy'), classes)
            np.save(os.path.join(image_level_save_path, d['file_name'].split('/')[-1][:-4] + '_gt_boxes.npy'), gt_boxes)
            np.save(os.path.join(image_level_save_path, d['file_name'].split('/')[-1][:-4] + '_gt_classes.npy'), gt_classes)
            plt.savefig(os.path.join(image_level_save_path,  d['file_name'].split('/')[-1]), bbox_inches='tight', pad_inches=0, dpi=300)



            break