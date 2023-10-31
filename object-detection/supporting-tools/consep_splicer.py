import cv2
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from PIL import Image
from multiprocessing import Pool
import multiprocessing
import os

basepath = os.path.dirname(os.path.abspath(__file__))
print(basepath)

def splicer(no, im_class='train',version='v4'):
    try:
        im_class = im_class[1:]
        labelfile = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Labels/mat/t{im_class}_{no}.mat'
        imagefile = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Images/t{im_class}_{no}.png'
        outpath_lab = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels_mat'
        outpath_img = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/images'
        txt_file = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Labels/yolo/t{im_class}_{no}.txt'
        overlay_file = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Labels/overlay/t{im_class}_{no}.png'
        image_file = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Images/t{im_class}_{no}.png'

        if not os.path.exists(outpath_lab):
            os.makedirs(outpath_lab)
        if not os.path.exists(outpath_img):
            os.makedirs(outpath_img)

        # image = image[0:250,0:250]
        label = scipy.io.loadmat(labelfile)
        image = cv2.imread(imagefile)
        label_segmap = np.array(label['inst_map']).astype('uint8')
        label_classmap = np.array(label['type_map']).astype('uint8')
        
        if version == 'v1' or version == 'v3':
            limit = 1000
            loopend = 1
        elif version == 'v2' or version == 'v4':
            limit = 250
            loopend = 4
        




        for i in range(0,loopend):
            for j in range(0,loopend):
                print(f'[{limit*i}:{limit*(i+1)},{limit*j}:{limit*(j+1)}]')
                image_x = image[limit*i:limit*(i+1),limit*j:limit*(j+1)]
                label_segmap_x = label_segmap[limit*i:limit*(i+1),limit*j:limit*(j+1)]
                label_classmap_x = label_classmap[limit*i:limit*(i+1),limit*j:limit*(j+1)]
                print(i,j)
                print(label_segmap_x.shape)
                print(label_classmap_x.shape)
                print(image_x.shape)
                cv2.imwrite(f'{outpath_img}/t{im_class}_{no}_{i}_{j}.png', image_x)
                scipy.io.savemat(f'{outpath_lab}/t{im_class}_{no}_{i}_{j}.mat', {'inst_map': label_segmap_x, 'type_map': label_classmap_x})



        # label_segmap = np.array(label_segmap).astype('uint8')[0:250,0:250]
        # label_classmap = np.array(label_classmap).astype('uint8')[0:250,0:250]

        # plt.imshow(label_classmap)
        # plt.show()
    except:
        print(f'Error in {no}')
        pass


# splicer(1, im_class='train',version='v1')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    pool = Pool(multiprocessing.cpu_count())
    pool.map(splicer, range(1, 28))
    # pool.map(splicer, range(1, 15))
    pool.close()
    pool.join()
