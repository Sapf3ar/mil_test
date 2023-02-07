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

def train(args):
    
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
    
    trainer.fit(model, ckpt_path=args.weights_path, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)


if __name__ == '__main__':
    wandb.login()
    args = get_parser()
    train(args)