import torch
import pytorch_lightning as pl
import torch
from config import Config
import evaluate
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor

class SegModel(pl.LightningModule):
    def __init__(self, Config):
        super(SegModel, self).__init__()                             
        self.metric=evaluate.load("mean_iou")
        self.args = Config
        self.lr = Config.lr
        self.id2label = self.args.id2label
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-ade",
                                                                id2label=self.id2label,
                                                                ignore_mismatched_sizes=True, mask_feature_size=512)  
        self.processor  = MaskFormerImageProcessor(ignore_index=0, 
                                        reduce_labels=False, 
                                        do_resize=False, 
                                        do_rescale=False, 
                                        do_normalize=False)
                                                                      
        self.train_batches = Config.train_batch_size*Config.max_epochs
        self.warmup_batches = int(self.train_batches*0.3)

    def forward(self, *args):
        return self.model(args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        #can't stack and move whole batch to device, because 
        #we need original images and original seg. maps (inconsistent size inside batch)
        #also, preprocessor object is required fot matching model outputs with predictions
        return batch

    def create_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
            
    def lr_warmup_config(self):

        def warmup(step):
            """
            This method will be called for ceil(warmup_batches/accum_grad_batches) times,
            warmup_steps has been adjusted accordingly
            """
            if self.args.warmup_steps <= 0:
                factor = 1
            else:
                factor = min(step / self.args.warmup_steps, 1)
            return factor

        opt1 = self.create_optimizer()
        return {
            'frequency': self.warmup_batches,
            'optimizer': opt1,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt1, warmup),
                'interval': 'step',
                'frequency': 1,
                'name': 'lr/warmup'
            },
        }


    def lr_decay_config(self):
        opt2 = self.create_optimizer()
        return {
            'frequency': self.train_batches - self.warmup_batches,
            'optimizer': opt2,
            'lr_scheduler': {

                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    opt2, 'min', 
                                    factor=self.args.lrdecay_factor, 
                                    patience=self.args.lrdecay_patience,
                                    threshold=self.args.lrdecay_threshold, 
                                    threshold_mode='rel',  verbose=False,
                                    
                                   ),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': self.args.lrdecay_monitor,
                'strict': False,
                'name': 'lr/reduce_on_plateau',
            }
        }


    def configure_optimizers(self):
        return (
            self.lr_warmup_config(),
            self.lr_decay_config()
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        outputs = self.model(
        pixel_values=batch["pixel_values"].to(self.device),
                mask_labels=[labels.to(self.device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(self.device) for labels in batch["class_labels"]],)
        loss = outputs.loss        
        self.log("train_loss", loss, logger=True, on_epoch=True)          
        return loss

    def match_seg_maps(self, batch, outputs):
        original_images = batch["original_images"]

        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]

        predicted_segmentation_maps = self.preprocessor.post_process_semantic_segmentation(outputs,
                                                                                    target_sizes=target_sizes)
        return predicted_segmentation_maps
    
    def get_metrics(self, batch, outputs):
        predicted_segmentation_maps = self.match_seg_maps(batch, outputs)

        ground_truth_segmentation_maps = batch["original_segmentation_maps"]
        metrics =self.metric.compute(references=ground_truth_segmentation_maps, 
                                    predictions=predicted_segmentation_maps,
                                    num_labels=self.args.num_labels, 
                                    ignore_index=self.arg.ignore_index)
        return metrics


    def validation_step(self, batch, batch_idx):

        self.log('step', batch_idx, prog_bar=True)

        outputs = self.model(
            pixel_values=batch["pixel_values"].to(self.device),
        )     
        metrics = self.get_metrics(outputs=outputs, batch=batch)
        iou = metrics['mean_iou']
        acc = metrics['mean_accuracy']
        self.log('val_iou', iou,logger=True,prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_acc', acc,logger=True )
    
    def predict_step(self, batch, batch_idx):

        self.log('step', batch_idx, prog_bar=True)

        outputs = self.model(
            pixel_values=batch["pixel_values"].to(self.device),
        )     
        predicted_segmentation_maps = self.match_seg_maps(batch, outputs)
        return predicted_segmentation_maps
        
    