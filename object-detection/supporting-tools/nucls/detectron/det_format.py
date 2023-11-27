import os 
import json
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import cv2


path_at_main = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project'

main_data = f'dataset/NuCLS/data/EvaluationSet'
main_save = f'object-detection/benchmarking/datasets/NuCLS'
if not os.path.exists(main_save):
    os.makedirs(main_save)

folds = 3
im_save = os.path.join(main_save, 'images')
if not os.path.exists(im_save):
    os.makedirs(im_save)

image = {
    "file_name": "",
    "height": 0,
    "width": 0,
    "image_id": 0,
    "annotations": [],
}

annotation_dict =  {
    "bbox": [],
    "bbox_mode": "BoxMode.XYWH_ABS",
    "category_id": 0,
}

classes = np.array(['nonTIL_stromal','sTIL','tumor_any','other_nucleus'])

im_data = os.path.join(main_data, 'rgb')

im_list = os.listdir(im_data)
try:
    im_list.remove('.DS_Store')
except:
    pass

master_imgs = []

for raw_img in im_list[4:]:
    try:
        img = cv2.imread(os.path.join(im_data, raw_img))
        mask = pd.read_csv(os.path.join(main_data, 'csv', raw_img[:-4] + '.csv'))
        print(f'Processing {raw_img}')

        im_out = os.path.join(im_save, raw_img)
        shutil.copy(os.path.join(im_data, raw_img), im_out)

        image_cp = image.copy()
        image_cp['file_name'] = os.path.join(path_at_main,raw_img)
        image_cp['image_id'] = raw_img[:-4]
        image_cp['height'] = img.shape[0]
        image_cp['width'] = img.shape[1]

        master_annots = []
        for index, row in mask.iterrows():
            annot_cp = annotation_dict.copy()
            if row['super_classification'] == 'unlabeled' or row['super_classification'] == 'AMBIGUOUS':
                continue
            class_id = np.where(classes == row['super_classification'])[0][0]
            annot_cp['category_id'] = class_id
            
            xmin = int(row['xmin'])
            xmax = int(row['xmax'])
            ymin = int(row['ymin'])
            ymax = int(row['ymax'])
            # print(xmin, xmax, ymin, ymax)

            annot_cp['bbox'] = [xmin, ymin, xmax, ymax]
            master_annots.append(annot_cp)
        image_cp['annotations'] = master_annots
        master_imgs.append(image_cp)
    except:
        continue

master = np.array(master_imgs)
print(master)

random.seed(42)
random.shuffle(master)
train = master[:int(len(master)*0.8)]
test = master[int(len(master)*0.8):]

fold_1 = train[:int(len(train)/folds)]
fold_2 = train[int(len(train)/folds):int(len(train)/folds)*2]
fold_3 = train[int(len(train)/folds)*2:]

train_fold_1 = np.concatenate((fold_1, fold_2), axis=0)
train_fold_2 = np.concatenate((fold_2, fold_3), axis=0)
train_fold_3 = np.concatenate((fold_1, fold_3), axis=0)

fold_save = os.path.join(main_save, 'folds',f'fold_1')
if not os.path.exists(fold_save):
    os.makedirs(fold_save)

# npy save path
np.save(os.path.join(fold_save, 'train.npy'), train_fold_1)
np.save(os.path.join(fold_save, 'test.npy'), test)

fold_save = os.path.join(main_save, 'folds',f'fold_2')
if not os.path.exists(fold_save):
    os.makedirs(fold_save)

# npy save path
np.save(os.path.join(fold_save, 'train.npy'), train_fold_2)
np.save(os.path.join(fold_save, 'test.npy'), test)


fold_save = os.path.join(main_save, 'folds',f'fold_3')
if not os.path.exists(fold_save):
    os.makedirs(fold_save)

# npy save path
np.save(os.path.join(fold_save, 'train.npy'), train_fold_3)
np.save(os.path.join(fold_save, 'test.npy'), test)

# final test save
fold_save = os.path.join(main_save, 'folds',f'final_test')
if not os.path.exists(fold_save):
    os.makedirs(fold_save)

# npy save path
np.save(os.path.join(fold_save, 'train.npy'), train)
np.save(os.path.join(fold_save, 'test.npy'), test)
