import numpy as np, os, pandas, shutil


path = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/EvalSet/detectron/master/images/'
master = np.load('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/master.npy', allow_pickle=True)
for i in range(master.shape[0]):
    master[i]['file_name'] = path + master[i]['file_name']
    # print(master[i])

train_split = 0.8
train = master[:int(train_split*master.shape[0])]
val = master[int(train_split*master.shape[0]):]

folds = 3
part_1 = train[:int(1/3 * train.shape[0])]
part_2 = train[int(1/3 * train.shape[0]) : int(2/3 * train.shape[0])]
part_3 = train[int(2/3 * train.shape[0]):]

print(part_1.shape, part_2.shape, part_3.shape)

fold_1_train = np.concatenate((part_1, part_2), axis=0)
fold_1_val = part_3

fold_2_train = np.concatenate((part_1, part_3), axis=0)
fold_2_val = part_2

fold_3_train = np.concatenate((part_2, part_3), axis=0)
fold_3_val = part_1

print(fold_1_train.shape, fold_1_val.shape)
print(fold_2_train.shape, fold_2_val.shape)
print(fold_3_train.shape, fold_3_val.shape)

np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/fold_1_train.npy', fold_1_train)
np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/fold_1_val.npy', fold_1_val)

np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/fold_2_train.npy', fold_2_train)
np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/fold_2_val.npy', fold_2_val)

np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/fold_3_train.npy', fold_3_train)
np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/fold_3_val.npy', fold_3_val)

np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/val.npy', val)
np.save('object-detection/benchmarking/datasets/NuCLSEvalSet/detectron/master/npsave/train.npy', train)
