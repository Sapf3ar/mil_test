import os
import torch
import pandas as pd
import cv2
import numpy as np
from config import Config
from transformers import MaskFormerImageProcessor

preprocessor = MaskFormerImageProcessor(ignore_index=0, 
                                        reduce_labels=False, 
                                        do_resize=False, 
                                        do_rescale=False, 
                                        do_normalize=False)

class ImageSegmentationDataset(torch.utils.data.Dataset):
    """Image segmentation dataset."""

    def __init__(self, transform, id_path:str, images_path:str='JPEGImages', masks_path:str='gt_masks' ):
        """
        Args:
            dataset
        """
        self.data_map = self.get_data_map(id_path, images_path, masks_path)
        
        self.transform = transform
        
        
    def __len__(self):
        return len(self.data_map)

    def get_data_map(self, id_path:str, images_path:str, masks_path:str) -> pd.DataFrame:
        with open(id_path) as file:
            ids = file.read().split('\n')
        data = {'mask_path':[], 'image_path':[]}
        blacklist = set('2009_003650') # there are masks small enough, 
                                       # so they dissapear through resizing and other transforms

        im_dir = os.listdir(images_path)

        img_ext = '.'+im_dir[0].split('.')[1]
        if len(im_dir) ==0:
          raise ()
        assert len() !=0, "Empty image directory"
        
        images = set([i.split('.')[0] for i in im_dir])
        mask_dir = os.listdir(masks_path)
        mask_ext = '.'+mask_dir[0].split('.')[1]
        assert len(mask_dir) !=0, "Empty masks directory"
        masks = set([i.split('.')[0] for i in mask_dir])

        for id_ in ids:
            if id_ in images and id_ in masks:
                if id_ in blacklist:
                    pass
                else:
                    data['mask_path'].append(os.path.join(masks_path, id_+mask_ext))
                    data['image_path'].append(os.path.join(images_path, id_+img_ext))

        return pd.DataFrame(data)

    def __getitem__(self, idx):
      
        row = self.data_map.iloc[idx]
        path_to_img = row['image_path']

        path_to_mask = row['mask_path']
        original_image= cv2.imread(path_to_img)
        original_segmentation_map = np.load(path_to_mask)
        
        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']      

        # convert to C, H, W
        image = image.transpose(2,0,1)
        return image, segmentation_map, original_image, original_segmentation_map


def collate_fn(batch):
  inputs = list(zip(*batch))
  
  images = inputs[0]
  segmentation_maps = inputs[1]

  batch = preprocessor(
    images,
    segmentation_maps=segmentation_maps,
    return_tensors="pt",
  )
  batch["original_images"] = inputs[2]        
  batch['original_segmentation_maps'] = inputs[3]
  return batch




def get_dataloader(id_path:str, transform, batch_size:int) -> torch.utils.data.Dataloader:
    current_dataset = ImageSegmentationDataset(id_path=id_path, transform=transform)
    current_dataloader = torch.utils.data.DataLoader(current_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return current_dataloader

def get_train_dataloader()