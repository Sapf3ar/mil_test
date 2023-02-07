# mil_test
Hierarchical semantic segmentation on pascal part dataset

### MaskFormer(base and small) fine-tuned on Pascal Part dataset


   Hierarchical metric (IoU) calculated 


### Training logs: 


[wandb logs](https://wandb.ai/sapsapfear/mil_test/overview?workspace=user-sapsapfear)

#### Achieved metrics:


Body-level mIoU: 0.499


Upper body-level: 0.438


Lower body-level: 0.439



Per category level: 0.644 0.752 0.656 0.926 0.614 0.608


 * (1) low_hand : 0.644
 
 
 * (6) up_hand : 0.752
 
 
 * (2) torso : 0.656
 
 
 * (4) head : 0.926
 
 
 * (3) low_leg : 0.614
 
 
 * (5) up_leg : 0.608



### Ideas
------------------------------------------------
##### Fine-tuning by hierarchy level


   Depends on what level metrics are most important, it is possible that fine-tuning model on part of a dataset (approx. 50%+) for body semantic
   
   segmentation, then fine-tuning on a bigger subset, but for upper and lower body parts mask. And only after that, fine-tune on all categories on all samples.
   
   
##### Data


   Filter samples with small mask area (relatively to the original image shape).
   Fine-tune (or train) model on other body parts dataset
