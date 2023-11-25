import comet_ml
import os
import random
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor, hooks
from comet_trainer import CometDefaultTrainer
from utils import format_predictions, get_balloon_dicts
import argparse
import comet_ml


from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model





def setup(fold, config_info, outputdir='output'):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_info))
    cfg.DATASETS.TRAIN = (f'fold_{fold}_train',)
    cfg.DATASETS.TEST = (f'fold_{fold}_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_info)  
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
    cfg.OUTPUT_DIR = outputdir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def log_predictions(predictor, dataset_dicts, experiment):
    """Log Model Predictions to Comet for analysis.

    Args:
        predictor (DefaultPredictor): Predictor Object for Detectron Model
        dataset_dicts (dict): Dataset Dictionary contaning samples of data and annotations
        experiment (comet_ml.Experiment): Comet Experiment Object
    """
    predictions_data = {}
    for d in random.sample(dataset_dicts, 3):
        file_name = str(d["file_name"])

        im = cv2.imread(file_name)
        annotations = d["annotations"]

        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        formatted_data = format_predictions(outputs, annotations)
        predictions_data[file_name] = formatted_data
        experiment.log_image(file_name, name=file_name)

    experiment.log_asset_data(predictions_data, name="predictions-data.json")




# experiment.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, default=1)
    parser.add_argument('--version', type=str, required=True, default='detectron')
    parser.add_argument('--iters', type=int, required=True, default=100)
    parser.add_argument('--batch_size', type=int, default=8, required=False)
    parser.add_argument('--lr', type=float, default=0.00025, required=False)
    parser.add_argument('--gpu', type=str, default='0', required=False)
    parser.add_argument('--num_workers', type=int, required=False, default=4)
    parser.add_argument('--log', type=str, required=False, default=True)
    parser.add_argument('--save', type=str, required=False, default=True)
    parser.add_argument('--project', type=str, required=True, default='capstone')
    parser.add_argument('--name', type=str, required=True, default='experiment')
    parser.add_argument('--data_path', type=str, default=os.curdir, required=True)
    parser.add_argument('--image_dir', type=str, default='images', required=True)
    parser.add_argument('--api_key', type=int, required=False, default=None)
    parser.add_argument('--config_info', type=str, required=True, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    parse = parser.parse_args()


    fold = parse.fold
    version = parse.version
    iters = parse.iters
    batch_size = parse.batch_size
    lr = parse.lr
    gpu = parse.gpu
    num_workers = parse.num_workers
    log = parse.log
    save = parse.save
    project = parse.project
    name = parse.name
    data_path = parse.data_path
    image_dir = parse.image_dir
    api = parse.api_key
    config_info = parse.config_info
    '''
    example run in multiline
    python detectron_train_v1.py \
    --fold 1 \
    --version detectron \
    --model mask_rcnn_R_50_FPN_3x \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.00025 \
    --gpu 0 \
    --num_workers 4 \
    --log True \
    --save True \
    --project capstone \
    --name mask_rcnn_R_50_FPN_3x \

    single line
    python detectron_train_v1.py --fold 1 --version detectron --iters 100 --batch_size 8 --lr 0.00025 --gpu 0 --num_workers 4 --log True --save True --project capstone-project --name mask_rcnn_R_50_FPN_3x --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/EvalSet/detectron/master/npsave --image_dir /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/EvalSet/detectron/master/images --config_info COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
    '''

    # print summary
    print('Fold: ', fold)
    print('Version: ', version)
    print('iters: ', iters)
    print('Batch Size: ', batch_size)
    print('Learning Rate: ', lr)
    print('GPU: ', gpu)
    print('Num Workers: ', num_workers)
    print('Log: ', log)
    print('Save: ', save)
    print('Project: ', project)
    print('Name: ', name)

    def data_train():
        data = np.load(os.path.join(data_path,f'fold_{fold}_train.npy'), allow_pickle=True)
        data = list(data)
        print(len(data))
        return data

    def data_val():
        data = np.load(os.path.join(data_path,f'fold_{fold}_val.npy'), allow_pickle=True)
        data = list(data)
        print(len(data))
        return data

    def data_test():
        data = np.load(os.path.join(data_path,f'test.npy'), allow_pickle=True)
        data = list(data)
        return data

    DatasetCatalog.register(f'fold_{fold}_train', data_train)
    data = DatasetCatalog.get(f'fold_{fold}_train')
    MetadataCatalog.get(f'fold_{fold}_train').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
    MetadataCatalog.get(f'fold_{fold}_train').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]


    DatasetCatalog.register(f'fold_{fold}_val', data_val)
    data = DatasetCatalog.get(f'fold_{fold}_val')
    MetadataCatalog.get(f'fold_{fold}_val').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
    MetadataCatalog.get(f'fold_{fold}_val').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]

    DatasetCatalog.register(f'test', data_test)
    data = DatasetCatalog.get(f'test')
    MetadataCatalog.get(f'test').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other_nucleus']
    MetadataCatalog.get(f'test').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]





    # cfg = setup(fold, config_info, outputdir = os.path.join(project, name))
    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    experiment = comet_ml.Experiment()
    cfg = setup(fold, config_info, outputdir = os.path.join(project, name))
    trainer = CometDefaultTrainer(cfg, experiment)
    trainer.resume_or_load(resume=False)
    # Register Hook to compute metrics using an Evaluator Object
    trainer.register_hooks(
        [hooks.EvalHook(10, lambda: trainer.evaluate_metrics(cfg, trainer.model))]
    )

    # Register Hook to compute eval loss
    trainer.register_hooks(
        [hooks.EvalHook(10, lambda: trainer.evaluate_loss(cfg, trainer.model))]
    )
    trainer.train()

    # Evaluate Model Predictions
    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_final.pth"
    )  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    log_predictions(predictor, get_balloon_dicts("balloon/val"), experiment)