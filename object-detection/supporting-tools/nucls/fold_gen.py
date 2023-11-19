import os
import shutil
import pandas as pd
import numpy as np
import cv2
import random

for folder in ['NuCLSCBootstrapControl','NuCLSEvalSet','NuCLSPBC','NuCLSQC']:
    for version in ['v1', 'v2']:
        src_path = f'object-detection/benchmarking/datasets/{folder}'
        save_path = f'object-detection/benchmarking/datasets/{folder}'
        src_path = os.path.join(src_path, version, 'master')
        save_path = os.path.join(save_path, version)

        ims = os.listdir(os.path.join(src_path, 'images'))
        masks = os.listdir(os.path.join(src_path, 'labels'))

        try:
            ims.remove('.DS_Store')
        except:
            pass
        try:
            masks.remove('.DS_Store')
        except:
            pass


        ims.sort()
        masks.sort()


        orig_len = len(ims)//5
        random_seed = 42
        random.seed(random_seed)
        random.shuffle(ims)
        random.seed(random_seed)
        random.shuffle(masks)

        ex_test_im = ims[0:orig_len]
        ex_test_mask = masks[0:orig_len]
        ex_train = ims[orig_len:]
        ex_train_mask = masks[orig_len:]

        orig_len = len(ex_train)//5
        random.seed(random_seed)
        random.shuffle(ex_train)
        random.seed(random_seed)
        random.shuffle(ex_train_mask)
        for i in range(5):
            fold_val = ex_train[orig_len*i:orig_len*(i+1)]
            fold_val_mask = ex_train_mask[orig_len*i:orig_len*(i+1)]

            fold_train = ex_train[0:orig_len*i] + ex_train[orig_len*(i+1):]
            fold_train_mask = ex_train_mask[0:orig_len*i] + ex_train_mask[orig_len*(i+1):]

            fold = f'fold_{i+1}'
            fold_path = os.path.join(save_path, fold)
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            fold_im_path = os.path.join(fold_path, 'val', 'images')
            fold_mask_path = os.path.join(fold_path, 'val', 'labels')
            if not os.path.exists(fold_im_path):
                os.makedirs(fold_im_path)
            if not os.path.exists(fold_mask_path):
                os.makedirs(fold_mask_path)
            for im, mask in zip(fold_val, fold_val_mask):
                shutil.copy(os.path.join(src_path, 'images', im), os.path.join(fold_im_path, im))
                shutil.copy(os.path.join(src_path,  'labels', mask), os.path.join(fold_mask_path, mask))

            fold_im_path = os.path.join(fold_path, 'train', 'images')
            fold_mask_path = os.path.join(fold_path, 'train','labels')
            if not os.path.exists(fold_im_path):
                os.makedirs(fold_im_path)
            if not os.path.exists(fold_mask_path):
                os.makedirs(fold_mask_path)
            for im, mask in zip(fold_train, fold_train_mask):
                shutil.copy(os.path.join(src_path, 'images', im), os.path.join(fold_im_path, im))
                shutil.copy(os.path.join(src_path,  'labels', mask), os.path.join(fold_mask_path, mask))


            val_path = os.path.join(fold_path, 'test')
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            val_im_path = os.path.join(val_path, 'images')
            val_mask_path = os.path.join(val_path, 'labels')
            if not os.path.exists(val_im_path):
                os.makedirs(val_im_path)
            if not os.path.exists(val_mask_path):
                os.makedirs(val_mask_path)
            for im, mask in zip(ex_test_im, ex_test_mask):
                shutil.copy(os.path.join(src_path, 'images', im), os.path.join(val_im_path, im))
                shutil.copy(os.path.join(src_path,  'labels', mask), os.path.join(val_mask_path, mask))
