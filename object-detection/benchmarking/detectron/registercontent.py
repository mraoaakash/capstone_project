import pandas as pd
import numpy as np
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

path_before_benchmark = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection'
def consep_v1_train():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_V1_train.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v1_test():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_V1_test.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v2_train():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v2_train.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v2_test():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v2_test.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v3_train():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v3_train.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v3_test():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v3_test.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v4_train():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v4_train.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list

def consep_v4_test():
    data_path = f'{path_before_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/detectron_df_v4_test.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # make df into a list of dictionaries
    df_list = df.to_dict('records')
    # print(df_list[0])
    return df_list



def register_consep_v1_train():
    DatasetCatalog.register("consep_v1_train", consep_v1_train)
    MetadataCatalog.get("consep_v1_train").set(thing_classes=['other','inflammatory','healthy epithelial','dysplastic/malignant epithelial','fibroblast','muscle','endothelial'])

def register_consep_v1_test():
    DatasetCatalog.register("consep_v1_test", consep_v1_test)
    MetadataCatalog.get("consep_v1_test").set(thing_classes=['other','inflammatory','healthy epithelial','dysplastic/malignant epithelial','fibroblast','muscle','endothelial'])

def register_consep_v2_train():
    DatasetCatalog.register("consep_v2_train", consep_v2_train)
    MetadataCatalog.get("consep_v2_train").set(thing_classes=['other','inflammatory','healthy epithelial','dysplastic/malignant epithelial','fibroblast','muscle','endothelial'])

def register_consep_v2_test():
    DatasetCatalog.register("consep_v2_test", consep_v2_test)
    MetadataCatalog.get("consep_v2_test").set(thing_classes=['other','inflammatory','healthy epithelial','dysplastic/malignant epithelial','fibroblast','muscle','endothelial'])

def register_consep_v3_train():
    DatasetCatalog.register("consep_v3_train", consep_v3_train)
    MetadataCatalog.get("consep_v3_train").set(thing_classes=['cell'])

def register_consep_v3_test():
    DatasetCatalog.register("consep_v3_test", consep_v3_test)
    MetadataCatalog.get("consep_v3_test").set(thing_classes=['cell'])

def register_consep_v4_train():
    DatasetCatalog.register("consep_v4_train", consep_v4_train)
    MetadataCatalog.get("consep_v4_train").set(thing_classes=['cell'])

def register_consep_v4_test():
    DatasetCatalog.register("consep_v4_test", consep_v4_test)
    MetadataCatalog.get("consep_v4_test").set(thing_classes=['cell'])
    

register_consep_v1_train()
register_consep_v1_test()
register_consep_v2_train()
register_consep_v2_test()
register_consep_v3_train()
register_consep_v3_test()
register_consep_v4_train()
register_consep_v4_test()

# import torch
# print(torch.__version__)