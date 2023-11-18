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

def overlay(no_arr, im_class='train',version='v1'):
    try:
        im_class = im_class[1:]
        labelfile = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels_mat/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.mat'
        outpath_lab = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels'
        outpath_img = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/images'
        outpath_overlay = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/overlay'
        outpath_classoverlay = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/overlay_classmap'
        outpath_cell_overlay = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/overlay_cellmap'
        txt_file = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.txt'
        overlay_file = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/overlay/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.png'
        image_file = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/images/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.png'

        if not os.path.exists(outpath_lab):
            os.makedirs(outpath_lab)
        if not os.path.exists(outpath_img):
            os.makedirs(outpath_img)
        if not os.path.exists(outpath_overlay):
            os.makedirs(outpath_overlay)
        if not os.path.exists(outpath_classoverlay):
            os.makedirs(outpath_classoverlay)
        if not os.path.exists(outpath_cell_overlay):
            os.makedirs(outpath_cell_overlay)

        # removing overlapping files if they exist
        if os.path.exists(txt_file):
            os.remove(txt_file)
        if os.path.exists(overlay_file):
            os.remove(overlay_file)

        mat = scipy.io.loadmat(labelfile)
        class_map = np.array(mat['type_map'])
        plt.imshow(class_map)
        plt.axis('off')
        outpath_classoverlay = f'{outpath_classoverlay}/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.png'
        plt.savefig(outpath_classoverlay, bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show()
        # plt.close()
        # print(mat['inst_map'])
        # finding unique values in mat['inst_map']
        # print(np.unique(mat['inst_map']))
        cells = np.unique(mat['inst_map'])
        plt.imshow(class_map)
        plt.axis('off')
        outpath_cell_overlay = f'{outpath_cell_overlay}/t{im_class}_{no_arr[0]}_{no_arr[1]}_{no_arr[2]}.png'
        plt.savefig(outpath_cell_overlay, bbox_inches='tight', pad_inches=0, dpi=300)
        # only keeping values that are 1
        img = mat['inst_map']
        img = img.astype('uint8')
        output = cv2.imread(image_file)
        global_contours = np.array([])
        cols = [(102,102,0),(14,17,19),(74,2,59),(109,53,1),(100,100,145),(149,4,9),(255,0,0)]
        for i in cells:
            img_copy = img.copy()
            img_copy[img != i] = 0
            contours = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(contours) > 0:
                for contour in contours:
                    # finding the class wirhin that contour
                    class_cont = np.array([])
                    for point in contour:
                        class_cont = np.append(class_cont, class_map[point[0][1]][point[0][0]])
                    # print(class_cont)
                    # print(np.unique(class_cont)[0].astype('int'))
                    x, y, w, h = cv2.boundingRect(contour)
                    # adding rectianges of yellow color
                    cv2.rectangle(output, (x,y), (x+w, y+h), cols[np.unique(class_cont)[0].astype('int')], 2, cv2.LINE_AA, 0)
                    # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
                    # print(f'{np.unique(class_cont)[0].astype('int'))} {x+w//2} {y+h//2} \n')
                    center_y = (y + h/2)/img.shape[0]
                    center_x = (x + w/2)/img.shape[1]
                    norm_height = h/img.shape[0]
                    norm_width = w/img.shape[1]
                    # writing to txt file
                    class_counter = (np.unique(class_cont)[0].astype("int") if np.unique(class_cont)[0].astype("int") else 0 ) if version in ['v1','v2'] else 0
                    # print(f'{class_counter} {center_x} {center_y} {norm_width} {norm_height}\n')
                    with open(txt_file, 'a') as f:
                        f.write(f'{class_counter} {center_x} {center_y} {norm_width} {norm_height}\n')

        # Display the output image
        plt.imshow(output)
        plt.axis('off')
        # plt.show()
        plt.savefig(overlay_file, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f'Done with {no_arr}')
    except:
        # print(f'Error in {labelfile}')
        pass
if __name__ == '__main__':
    # making all possible permutations of three lower and upper bounds
    main_arr = []
    for i in range(1,28):
        for j in range(0,4):
            for k in range(0,4):
                no_arr = [i,j,k]
                main_arr.append(no_arr)
    # print(main_arr)

    overlay(main_arr[0], im_class='train',version='v1')
    overlay(main_arr[0], im_class='train',version='v2')
    overlay(main_arr[0], im_class='train',version='v3')
    overlay(main_arr[0], im_class='train',version='v4')
    # with Pool(multiprocessing.cpu_count()) as p:
    #     p.map(overlay, main_arr)
    #     p.close()
    #     p.join()
    # print('Done')