from model import SegModel
from config import Config
from dataset import get_train_dataloader, get_test_datalaoder
import os
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
import evaluate
from mask_utils import get_body_mask, upper_body_mask, lower_body_mask

def category_metric(preds):
    metric_body =  evaluate.load("mean_iou")
    metric_upper_body =  evaluate.load("mean_iou")
    metric_lower_body =  evaluate.load("mean_iou")
    metric_category =  evaluate.load("mean_iou")
    for i in range(len(preds)):
        gt = list(preds[i]['gt'])
        output = preds[i]['preds']
        metric_body.add_batch(references=[get_body_mask(i) for i in gt], predictions=[get_body_mask(i.numpy()) for i in output])
        metric_upper_body.add_batch(references=[upper_body_mask(i) for i in gt], predictions=[upper_body_mask(i.numpy()) for i in output])
        metric_lower_body.add_batch(references=[lower_body_mask(i) for i in gt], predictions=[lower_body_mask(i.numpy()) for i in output])
        metric_category.add_batch(references=gt, predictions=output)


    print(metric_body.compute(num_labels=2, ignore_index = 0)['mean_iou'])
    print(metric_upper_body.compute(num_labels=2, ignore_index = 0)['mean_iou'])
    print(metric_lower_body.compute(num_labels=2, ignore_index = 0)['mean_iou'])
    print(metric_category.compute(num_labels=len(Config.id2label), ignore_index = 0)['per_category_iou'])


def get_parser():
    parser = argparse.ArgumentParser(description='Parse training data arguments')

    parser.add_argument('main_dir', type=str, help='Main dataset dir')
    parser.add_argument(
        'weights_path',
        type=str,
        help='path for checkpoint to procced training from'
    )
    parser.add_argument(
        '--log',
        type=bool,
        default=True,
        help='log to wandb logger'
    )
    args = parser.parse_args()
    return args

def validate(args):
    
    model = SegModel(Config)    
    
    #get data for training
    val_dataloader = None
    if 'validation_ids' in dir(Config):
        validation_ids = Config.validation_ids
        val_dataloader = get_test_datalaoder(validation_ids)
    if val_dataloader is None:
        raise FileNotFoundError('No validation ids provided!')

    callbacks = []
    if args.log:
        wandb_logger = WandbLogger(project=Config.wandb_project)
        callbacks.append(wandb_logger)
    model = SegModel()
    trainer = pl.Trainer(logger=wandb_logger, 
                        gpus=1,
                        precision=16,
                        log_every_n_steps=1,
                        min_epochs=1,
                        check_val_every_n_epoch=2,
                        )
    
    preds = trainer.predict(model, val_dataloader, ckpt_path=args.weights_path)
    

if __name__ == '__main__':
    wandb.login()
    args = get_parser()
    preds = validate(args)
    category_metric(preds=preds)