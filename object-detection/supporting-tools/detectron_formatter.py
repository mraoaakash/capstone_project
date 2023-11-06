import cv2
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from PIL import Image
from multiprocessing import Pool
import multiprocessing
import os
import pandas as pd

basepath = os.path.dirname(os.path.abspath(__file__))
print(basepath)

base_dict = {
    "bbox":[],
    "bbox_mode":1,
    "category_id":0
}
detectron_df = pd.DataFrame(columns=['file_name','height', 'width','image_id','annotations'])

path_till_benchmark = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection'
def overlay(no_arr, im_class='train',version='v2'):
    try:
        det_copy = detectron_df.copy()
        im_class = im_class[1:]
        labelfile = f'{path_till_benchmark}/benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels_mat/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.mat'
        txt_file = f'{path_till_benchmark}/benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.txt'
        image_file = f'{path_till_benchmark}/benchmarking/datasets/CoNSeP/{version}/t{im_class}/images/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.png'

        # removing overlapping files if they exist
        if os.path.exists(txt_file):
            os.remove(txt_file)

        mat = scipy.io.loadmat(labelfile)
        cells = np.unique(mat['inst_map'])
        class_map = np.array(mat['type_map'])
        
        # only keeping values that are 1
        img = mat['inst_map']
        img = img.astype('uint8')
        output = cv2.imread(image_file)
        im_width = output.shape[1]
        im_height = output.shape[0]

        box_list = []
        cols = [(102,102,0),(14,17,19),(74,2,59),(109,53,1),(100,100,145),(149,4,9),(255,0,0)]
        for i in cells:
            img_copy = img.copy()
            img_copy[img != i] = 0
            contours = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(contours) > 0:
                for contour in contours:
                    cont_inf = base_dict.copy()
                    # finding the class wirhin that contour
                    class_cont = np.array([])
                    for point in contour:
                        class_cont = np.append(class_cont, class_map[point[0][1]][point[0][0]])
                    # print(class_cont)

                    x, y, w, h = cv2.boundingRect(contour)


                    # finding predominant class
                    class_cont = class_cont[class_cont != 0]
                    class_cont = np.unique(class_cont)
                    class_cont = class_cont.astype('int')
                    class_cont = class_cont.tolist()
                    class_cont = class_cont[0] - 1

                    # x_norm, y_norm, w_norm, h_norm = x/im_width, y/im_height, w/im_width, h/im_height
                    bbox = [x, y, w, h]
                    
                    # print(bbox)
                    # print(class_cont)
                    
                    cont_inf['bbox'] = bbox
                    cont_inf['category_id'] = class_cont
                    # print(cont_inf)
                    box_list.append(cont_inf)

        # box_list = np.array(box_list)

        print(f'Done with {no_arr}')
        det_copy['file_name'] = [image_file]
        det_copy['height'] = [im_height]
        det_copy['width'] = [im_width]
        det_copy['image_id'] = f't{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}'
        det_copy['annotations'] = [box_list]

        # print(det_copy)
        return det_copy
    except:
        # print(f'Error in {labelfile}')
        pass

if __name__ == '__main__':
    # making all possible permutations of three lower and upper bounds
    master_df = detectron_df.copy()
    main_arr = []
    im_class = 'train'
    version = 'v1'


    for i in range(1,28):
        for j in range(0,4):
            for k in range(0,4):
                no_arr = [i,j,k]
                main_arr.append(no_arr)
                df = overlay(no_arr,im_class=im_class,version=version)
                master_df = pd.concat([master_df, df], ignore_index=True)
    
    print(master_df)
    det_path = f'{path_till_benchmark}/benchmarking/datasets/CoNSeP/detectron_format/'
    if not os.path.exists(det_path):
        os.makedirs(det_path)
    master_df.to_csv(det_path + f'detectron_df_{version}_{im_class}.csv', index=False)

