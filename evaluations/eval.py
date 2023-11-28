import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

main_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/outputs'
outpath = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/converted'
if not os.path.exists(outpath):
    os.makedirs(outpath)

models = os.listdir(main_path)

for model in models:
    prediction = np.load(os.path.join(main_path, model, 'predictions.npy'), allow_pickle=True)
    print(prediction)