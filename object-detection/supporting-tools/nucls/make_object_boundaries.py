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

colors = [(166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44), (251, 154, 153), (227, 26, 28), (253, 191, 111), (255, 127, 0), (202, 178, 214), (106, 61, 154), (255, 255, 153), (177, 89, 40)]


for version in ['v1', 'v2']:
    src_path = f'dataset/NuCLS/data/BootstrapControl'
    save_path = f'object-detection/benchmarking/datasets/NuCLSCBootstrapControl'
    save_path = os.path.join(save_path, version, 'master')

    im_save_path = os.path.join(save_path, 'images')
    mask_save_path = os.path.join(save_path, 'labels')
    overlay_save_path = os.path.join(save_path, 'overlay')

    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    if not os.path.exists(overlay_save_path):
        os.makedirs(overlay_save_path)

    classes = np.array(['apoptotic_body', 'ductal_epithelium', 'eosinophil', 'fibroblast', 'lymphocyte', 'macrophage', 'mitotic_figure', 'myoepithelium', 'neutrophil', 'plasma_cell', 'tumor', 'vascular_endothelium'])


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
    for image, mask in zip(img_list, mask_list): #zip(img_list[0:1], mask_list[0:1]):
        print(f'Processing {image}')
        img = cv2.imread(os.path.join(img_path, image))
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shutil.copy(os.path.join(img_path, image), os.path.join(im_save_path, image))
        imcpy = img.copy()
        contour_info = pd.read_csv(os.path.join(mask_path, mask))
        # filter to remove unlabelled raw_classification
        contour_info = contour_info[contour_info['raw_classification'] != 'unlabeled']
        # iterating over each contour
        label_file =os.path.join(mask_save_path, f'{image[:-4]}.txt')
        

        if os.path.exists(label_file):
            os.remove(label_file)
        for index, row in contour_info.iterrows():
            class_array.append(row['raw_classification'])

            x_min = row['xmin']
            x_max = row['xmax']
            y_min = row['ymin']
            y_max = row['ymax']        

            # computing width height and center
            width = x_max - x_min
            height = y_max - y_min
            # normalizig the values to be between 0 and 1
            width = width/imcpy.shape[1]
            height = height/imcpy.shape[0]
            center_x = (x_min + width/2)/imcpy.shape[1]
            center_y = (y_min + height/2)/imcpy.shape[0]
            class_raw = row['raw_classification']
            class_index = np.where(classes == class_raw)[0][0]
            # drawing the bounding box with color based on class_index
            # cv2.rectangle(imcpy, (x_min, y_min), (x_max, y_max), colors[class_index], 1)
            # writing the label file
            with open(label_file, 'a') as f:
                f.write(f'{class_index if version=="v1" else 0} {center_x} {center_y} {width} {height}\n')

        # plt.imshow(imcpy)
        # plt.axis('off')
        # plt.tight_layout(pad=0)
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # # plt.show()
        # plt.savefig(os.path.join(overlay_save_path, f'{image}'), dpi=300)

    class_array = np.array(class_array)
    print(np.unique(class_array))