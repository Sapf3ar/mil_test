from typing import Dict
import numpy as np

def parse_id2label(self, path:str) -> Dict[str, str]:
        with open(path) as file:
            labels = file.read().split('\n')
        id2label = dict()
        for entry in labels:
            id_, label = entry.split(':')
            id2label[int(id_)] = label.strip()
        return id2label

class Config:
    height:int = 512
    width:int = 512

    trains_ids:str = 'train_id.txt'
    validation_ids:str = 'val_id.txt'
    
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    train_batch_size=12
    validation_batch_size=8

    lr:float = 5e-5
    warmup_steps:int = 200
    max_epochs:int = 20
    lrdecay_factor = 0.5

    train_batches = train_batch_size*max_epochs

    warmup_batches = int(train_batches*0.1)

    lrdecay_threshold = 0.01
    lrdecay_monitor = 'train_loss'    
    lrdecay_patience = 2

    path_id2label:str = 'classes.txt'
    id2label = parse_id2label(path_id2label)

