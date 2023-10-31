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

def overlay(no, im_class='train',version='v2'):
    im_class = im_class[1:]
    labelfile = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Labels/mat/t{im_class}_{no}.mat'
    imagefile = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Images/t{im_class}_{no}.png'
    outpath_lab = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/labels'
    outpath_img = f'benchmarking/datasets/CoNSeP/{version}/t{im_class}/images'
    txt_file = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Labels/yolo/t{im_class}_{no}.txt'
    overlay_file = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Labels/overlay/t{im_class}_{no}.png'
    image_file = f'benchmarking/datasets/CoNSeP_master/T{im_class}/Images/t{im_class}_{no}.png'



    if not os.path.exists(outpath_lab):
        os.makedirs(outpath_lab)
    if not os.path.exists(outpath_img):
        os.makedirs(outpath_img)

    # removing overlapping files if they exist
    if os.path.exists(txt_file):
        os.remove(txt_file)
    if os.path.exists(overlay_file):
        os.remove(overlay_file)

    # img = cv2.imread(labelfile)
    mat = scipy.io.loadmat(labelfile)
    class_map = mat['type_map']
    plt.imshow(class_map)
    plt.axis(drop=True)
    plt.savefig
    plt.close()
    # print(mat['inst_map'])
    # finding unique values in mat['inst_map']
    # print(np.unique(mat['inst_map']))
    cells = np.unique(mat['inst_map'])
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
                # cv2.rectangle(output, (x,y), (x+w, y+h), cols[np.unique(class_cont)[0].astype('int'))], 2, cv2.LINE_AA, 0)
                # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
                # print(f'{np.unique(class_cont)[0].astype('int'))} {x+w//2} {y+h//2} \n')
                center_y = (y + h/2)/img.shape[0]
                center_x = (x + w/2)/img.shape[1]
                norm_height = h/img.shape[0]
                norm_width = w/img.shape[1]
                # writing to txt file
                print(f'{np.unique(class_cont)[0].astype("int") if np.unique(class_cont)[0].astype("int") else 0 } {center_x} {center_y} {norm_width} {norm_height}\n')
                with open(txt_file, 'a') as f:
                    f.write(f'{np.unique(class_cont)[0].astype("int") if np.unique(class_cont)[0].astype("int") else 0 } {center_x} {center_y} {norm_width} {norm_height}\n')
        

    # Display the output image
    plt.imshow(output)
    plt.axis('off')
    # plt.show()
    # plt.savefig(overlay_file, bbox_inches='tight', pad_inches=0, dpi=300)


overlay(1, im_class='test',version='v1')

# if __name__ == '__main__':
#     arr = np.arange(1, 28)
#     num_cores = multiprocessing.cpu_count()
#     pool = Pool(num_cores)
#     pool.map(overlay, arr)
#     pool.close()
