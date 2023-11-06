import pandas as pd
import numpy as np
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt


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
