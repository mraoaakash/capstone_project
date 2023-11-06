import pandas as pd
import numpy as np
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
import cv2


path_before_benchmark = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection'
def consep_v1_train():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v1_train.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v1_test():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v1_test.csv'
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


for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow(vis.get_image()[:, :, ::-1])
    # save the image
    cv2.imwrite(f'{path_before_benchmark}/report_figures/detectron/train_batches/{d["file_name"]}.png', vis.get_image()[:, :, ::-1])