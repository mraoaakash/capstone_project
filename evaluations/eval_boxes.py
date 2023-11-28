import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


basepath = f'evaluations/data/faster_rcnn_R_50_C4_1x_fold_1/predictions'
impath = '/Users/mraoaakash/Documents/research/capstone_project/object-detection/benchmarking/datasets/NuCLS/images'
thresh_IOU = 0.5

conf_matrix = np.zeros((4,4))

count = 0
for i in os.listdir(basepath):
    if '.DS_Store' in i:
        continue
    print(i)
    level_path = os.path.join(basepath, i)
    gt_boxes = os.path.join(level_path,f'{i}_gt_boxes.npy')
    gt_boxes = np.load(gt_boxes, allow_pickle=False).astype(int)
    print(gt_boxes.shape)
    gt_classes = os.path.join(level_path,f'{i}_gt_classes.npy')
    gt_classes = np.load(gt_classes, allow_pickle=False)
    print(gt_classes.shape)
    boxes = os.path.join(level_path,f'{i}_boxes.npy')
    boxes = np.load(boxes, allow_pickle=False).astype(int)
    print(boxes.shape)
    scores = os.path.join(level_path,f'{i}_scores.npy')
    scores = np.load(scores, allow_pickle=False).astype(int)
    print(scores.shape)
    classes = os.path.join(level_path,f'{i}_classes.npy')
    classes = np.load(classes, allow_pickle=False)
    print(classes.shape)
    print('---')
    image = cv2.imread(os.path.join(impath, f'{i}.png'))
    print(image.shape)

    for j in range(gt_boxes.shape[0]):
        for k in range(boxes.shape[0]):
            IoU1 = get_iou(gt_boxes[j], boxes[k])
            IoU2 = get_iou(boxes[k], gt_boxes[j])
            IoU = max(IoU1, IoU2)
            if IoU >= thresh_IOU:
                print(gt_boxes[j])
                print(boxes[k])
                print(IoU)
                print('---')
                cv2.rectangle(image, (gt_boxes[j][0], gt_boxes[j][1]), (gt_boxes[j][2], gt_boxes[j][3]), (0, 255, 0), 2)
                cv2.rectangle(image, (boxes[k][0], boxes[k][1]), (boxes[k][2], boxes[k][3]), (0, 0, 255), 2)
    # cv2.imshow('image', image)
    # plt.show()
    cv2.imwrite(level_path + f'/{i}_comparison.png', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    count += 1
    # if count == 5:
    #     break