import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import cv2
import re

def cm():
    path = f'benchmarking/yolov5/runs/train/experiment 2/'

    outpath = f'report_figures/yolov5/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    experiments = os.listdir(path)
    experiments = [experiment for experiment in experiments if experiment != '.DS_Store']
    # experiments = [f'{path}{experiment}' for experiment in experiments]
    print(experiments)

    for experiment in experiments:
        results = pd.read_csv(f'{path}{experiment}/results.csv')
        confusion_matrix = pd.read_json(f'{path}{experiment}/confusion_matrix.json')
        matrix = np.array(confusion_matrix['matrix'].values.tolist()).astype(int)
        labels = confusion_matrix['labels'].values.tolist()
        labels = np.array([label.title() for label in labels])
        # finding the index of "Dysplastic"
    
        
        if 'v1' in experiment or 'v2' in experiment:
            dysplastic_index = np.where(labels == 'Dysplastic/Malignant Epithelial')[0][0]
            labels[dysplastic_index] = 'Dysplastic Epithelial'
            print(dysplastic_index)
            plt.figure(figsize=(10.5, 10))
            sns.heatmap(matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
            plt.xlabel('Predicted', fontsize=16,fontweight='bold')
            plt.ylabel('True', fontsize=16, fontweight='bold')
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xticks(fontsize=10, rotation=45, ha='right')
            plt.yticks(fontsize=10)
            plt.tight_layout()
        else:
            plt.figure(figsize=(5.5, 5))
            sns.heatmap(matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
            plt.xlabel('Predicted', fontsize=14,fontweight='bold')
            plt.ylabel('True', fontsize=14, fontweight='bold')
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xticks(fontsize=10, rotation=45, ha='right')
            plt.yticks(fontsize=10)
            plt.tight_layout()
        # plt.show()
        plt.savefig(f'{outpath}{experiment}_confusion_matrix.png', dpi=300)
        pass


def hist():
    path = f'benchmarking/yolov5/runs/train/experiment 2/'
    outpath = f'report_figures/yolov5/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    epcount = '50'
    experiments = os.listdir(path)
    experiments = [experiment for experiment in experiments if experiment != '.DS_Store']
    experiments = [experiment for experiment in experiments if experiment[0] == epcount[0]]
    # experiments = [f'{path}{experiment}' for experiment in experiments]
    
    print(experiments)

    col_names = {'               epoch':'epochs', '      train/box_loss':'train_box_loss', '      train/obj_loss':'trai_obj_loss',
       '      train/cls_loss':'train_cls_loss', '   metrics/precision':'precision', '      metrics/recall':'recall',
       '     metrics/mAP_0.5':'ap50', 'metrics/mAP_0.5:0.95':'ap95', '        val/box_loss':'val_box_loss',
       '        val/obj_loss':'val_obj_loss', '        val/cls_loss':'va;_class_loss', '               x/lr0':'lr_0',
       '               x/lr1':'lr_1', '               x/lr2':'lr_2'}

    train_box_loss = pd.DataFrame(columns=experiments)
    train_obj_loss = pd.DataFrame(columns=experiments)
    train_cls_loss = pd.DataFrame(columns=experiments)
    precision = pd.DataFrame(columns=experiments)
    recall = pd.DataFrame(columns=experiments)
    ap50 = pd.DataFrame(columns=experiments)
    ap95 = pd.DataFrame(columns=experiments)
    val_box_loss = pd.DataFrame(columns=experiments)
    val_obj_loss = pd.DataFrame(columns=experiments)
    val_cls_loss = pd.DataFrame(columns=experiments)
    for experiment in experiments:
        results = pd.read_csv(f'{path}{experiment}/results.csv')
        results.rename(columns = {'               epoch':'epochs', '      train/box_loss':'train_box_loss', '      train/obj_loss':'train_obj_loss','      train/cls_loss':'train_cls_loss', '   metrics/precision':'precision', '      metrics/recall':'recall','     metrics/mAP_0.5':'ap50', 'metrics/mAP_0.5:0.95':'ap95', '        val/box_loss':'val_box_loss','        val/obj_loss':'val_obj_loss', '        val/cls_loss':'val_cls_loss', '               x/lr0':'lr_0','               x/lr1':'lr_1', '               x/lr2':'lr_2'}, inplace = True)
        print(results.columns)
        precision[experiment] = results['precision'].tolist()
        recall[experiment] = results['recall'].tolist()
        ap50[experiment] = results['ap50'].tolist()
        ap95[experiment] = results['ap95'].tolist()
        train_box_loss[experiment] = results['train_box_loss'].tolist()
        train_cls_loss[experiment] = results['train_cls_loss'].tolist()
        train_obj_loss[experiment] = results['train_obj_loss'].tolist()
        val_box_loss[experiment] = results['val_box_loss'].tolist()
        val_cls_loss[experiment] = results['val_cls_loss'].tolist()
        val_obj_loss[experiment] = results['val_obj_loss'].tolist()
    print(precision)
    title_n_axis =  {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10,
        }

    precision = precision[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(precision, label = precision.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Precision",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Precision",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}precision_{epcount}.jpg', dpi=300)
    
    recall = recall[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(recall, label = recall.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Recall",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Recall",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}recall_{epcount}.jpg', dpi=300)
    
    ap95 = ap95[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(ap95, label = ap95.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model mAP:0.95",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("mAP:0.95",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}ap95_{epcount}.jpg', dpi=300)
    
    ap50 = ap50[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(ap50, label = ap50.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model mAP:0.5",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("mAP:0.5",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}ap50_{epcount}.jpg', dpi=300)
    
    val_box_loss = val_box_loss[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(val_box_loss, label = val_box_loss.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Box Loss (Val)",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Box Loss (Val)",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}val_box_loss_{epcount}.jpg', dpi=300)
    
    val_cls_loss = val_cls_loss[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(val_cls_loss, label = val_cls_loss.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Class Loss (Val)",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Class Loss (Val)",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}val_cls_loss_{epcount}.jpg', dpi=300)
    
    val_obj_loss = val_obj_loss[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(val_obj_loss, label = val_obj_loss.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Obj. Loss (Val)",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Obj. Loss (Val)",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}val_obj_loss_{epcount}.jpg', dpi=300)
    
    train_box_loss = train_box_loss[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(train_box_loss, label = train_box_loss.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Box Loss (Train)",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Box Loss (Train)",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}train_box_loss_{epcount}.jpg', dpi=300)
    
    train_cls_loss = train_cls_loss[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(train_cls_loss, label = train_cls_loss.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Class Loss (Train)",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Class Loss (Train)",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}train_cls_losss_{epcount}.jpg', dpi=300)
    
    train_obj_loss = train_obj_loss[[f'{epcount}_epoch_v1',f'{epcount}_epoch_v2',f'{epcount}_epoch_v3',f'{epcount}_epoch_v4']]
    plt.figure(figsize=(4,3))
    plt.plot(train_obj_loss, label = train_obj_loss.columns)
    plt.legend(['V1','V2','V3','V4'], bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.title("Model Obj. Loss (Train)",fontdict=title_n_axis)
    plt.xlabel("Epochs",fontdict=title_n_axis)
    plt.ylabel("Obj. Loss (Train)",fontdict=title_n_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{outpath}train_obj_loss_{epcount}.jpg', dpi=300)
    
    

hist()