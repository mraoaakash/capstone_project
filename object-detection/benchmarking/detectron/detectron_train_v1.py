import pandas as pd
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt

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


experiment = Experiment(
  api_key="AKIafKnSIJd2sEZkr8pUN3fnv",
  project_name="faster-rcnn",
  workspace="mraoaakash"
)

path_before_benchmark = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection'
def consep_v1_train():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_test_v1.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # changing type of "annotations" to list of dictionaries
    df["annotations"] = df["annotations"].apply(lambda x: eval(x))
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print datatypes of each column
    # print(df)
    # print(type(df["annotations"][0]))
    # print(df_list[0])
    return df_list

def consep_v1_test():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_train_v1.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list


DatasetCatalog.register("consep_v1_train", consep_v1_train)
MetadataCatalog.get("consep_v1_train").set(thing_classes=['other','inflammatory','healthy epithelial','dysplastic/malignant epithelial','fibroblast','muscle','endothelial'])

DatasetCatalog.register("consep_v1_test", consep_v1_test)
MetadataCatalog.get("consep_v1_test").set(thing_classes=['other','inflammatory','healthy epithelial','dysplastic/malignant epithelial','fibroblast','muscle','endothelial'])

my_dataset_train_metadata = MetadataCatalog.get("consep_v1_train")
dataset_dicts = DatasetCatalog.get("consep_v1_train")





print(len(dataset_dicts))
if len(os.listdir(f'{path_before_benchmark}/benchmarking/report_figures/detectron/train_batches/') ) != 0:
    print("Emptying train batch folder")
    os.system(f'rm {path_before_benchmark}/benchmarking/report_figures/detectron/train_batches/*')
    print("Done emptying train batch folder")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    print(img.shape)
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow(vis.get_image()[:, :, ::-1])
    # save the image
    plt.imsave(f'{path_before_benchmark}/benchmarking/report_figures/detectron/train_batches/{str(d["file_name"]).split("/")[-1]}', vis.get_image()[:, :, ::-1])
    # cv2.imwrite(f'{path_before_benchmark}/benchmarking/report_figures/detectron/train_batches/{d["file_name"]}.png', vis.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("consep_v1_train",)
cfg.DATASETS.TEST = ("consep_v1_test",)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# cfg to dict
hyper_params = cfg.dump()
experiment.log_parameters(hyper_params)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



log_model(experiment, trainer, model_name="faster_rcnn_X_101_32x8d_FPN_3x")