import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import json



# read colors json
with open('object-detection/benchmarking/datasets/Custom.json') as f:
    colors = json.load(f)

colors = [(166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44), (251, 154, 153), (227, 26, 28), (253, 191, 111), (255, 127, 0), (202, 178, 214), (106, 61, 154), (255, 255, 153), (177, 89, 40), (141, 211, 199)]


for version in ['v2']:
    src_path = f'dataset/NuCLS/data/EvaluationSet'
    save_path = f'object-detection/benchmarking/datasets/NuCLSEvalSet'
    save_path = os.path.join(save_path, version, 'master')

    im_save_path = os.path.join(save_path, 'images')
    mask_save_path = os.path.join(save_path, 'labels')
    overlay_save_path = os.path.join(save_path, 'overlay')
    overlay_contour_save_path = os.path.join(save_path, 'overlay_contour')

    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    if not os.path.exists(overlay_save_path):
        os.makedirs(overlay_save_path)
    if not os.path.exists(overlay_contour_save_path):
        os.makedirs(overlay_contour_save_path)

    # classes = np.array(['apoptotic_body', 'ductal_epithelium', 'eosinophil', 'fibroblast', 'lymphocyte', 'macrophage', 'mitotic_figure', 'myoepithelium', 'neutrophil', 'plasma_cell', 'tumor', 'vascular_endothelium','unlabeled'])
    classes = np.array(['nonTIL_stromal','sTIL','tumor_any','other_nucleus'])



    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_path = os.path.join(src_path, 'rgb')
    mask_path = os.path.join(src_path, 'csv')

    img_list = os.listdir(img_path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass
    mask_list = os.listdir(mask_path)
    try:
        mask_list.remove('.DS_Store')
    except:
        pass

    img_list.sort()
    mask_list.sort()

    class_array = []
    width_array = []
    height_array = []
    for image, mask in zip(img_list, mask_list): #zip(img_list[0:1], mask_list[0:1]):
        img = cv2.imread(os.path.join(img_path, image))
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imcpy = img.copy()
        contour_copy = img.copy()
        try:
            contour_info = pd.read_csv(os.path.join(mask_path, image[:-4] + '.csv'))
        except:
            continue
        # filter to remove unlabelled super_classification
        shutil.copy(os.path.join(img_path, image), os.path.join(im_save_path, image))
        print(f'Processing {image}')
        print(f'Processing {image[:-4]}.csv')
        # iterating over each contour
        label_file =os.path.join(mask_save_path, f'{image[:-4]}.txt')
        

        if os.path.exists(label_file):
            os.remove(label_file)
        for index, row in contour_info.iterrows():
            if row['super_classification'] not in classes:
                continue
            class_array.append(row['super_classification'])
            coords_x = row['coords_x'][1:-1].split(',')[1:-1]
            print(coords_x)
            coords_y = row['coords_y'][1:-1].split(',')[1:-1]
            coords_x = [int(x) for x in coords_x]
            coords_y = [int(y) for y in coords_y]
            x_min = row['xmin']
            x_max = row['xmax']
            y_min = row['ymin']
            y_max = row['ymax']        

            width = x_max - x_min
            height = y_max - y_min

            center_x = x_min + width/2
            center_y = y_min + height/2

            norm_x = center_x/img.shape[1]
            norm_y = center_y/img.shape[0]
            norm_width = width/img.shape[1]
            norm_height = height/img.shape[0]
            print(row['super_classification'])
            class_info = np.where(classes == row['super_classification'])[0][0]
            cv2.rectangle(imcpy, (x_min, y_min), (x_max, y_max), colors[class_info], 2)
            # drawing contours
            if row['type'] == 'polyline':
                cv2.drawContours(contour_copy, [np.array(list(zip(coords_x, coords_y)))], 0, colors[class_info], 2)
            else:
                cv2.rectangle(contour_copy, (x_min, y_min), (x_max, y_max), colors[class_info], 2)
            

            with open(label_file, 'a') as f:
                f.write(f'{class_info} {norm_x} {norm_y} {norm_width} {norm_height}\n')
        cv2.imwrite(os.path.join(overlay_save_path, image), imcpy)
        cv2.imwrite(os.path.join(overlay_contour_save_path, image), contour_copy)

