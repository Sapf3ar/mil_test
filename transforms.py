import albumentations as A
from config import Config

def get_train_transforms(use_optional:bool = False, Config=Config):

    train_transform = A.Compose(
        [       A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.4),
                A.Resize(width=Config.width, height=Config.height),
        
                A.Normalize(mean=Config.ADE_MEAN, std=Config.ADE_STD),
        
        ]
    )

    #optional transforms:
    optional_transforms = A.Compose(
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=(-8, 8), #15
                                        shift_limit=0.1, p=1, border_mode=0, value=0.0),
                    A.IAAAdditiveGaussianNoise(p=0.2),
                    A.IAAPerspective(p=0.5),
                    A.OneOf(
                        [
                            A.CLAHE(p=1),
                            A.RandomBrightnessContrast(p=1),
                            A.RandomGamma(p=1),
                        ],
                        p=0.9,
                    ),
                    A.OneOf(
                        [
                            A.IAASharpen(p=1),
                            A.Blur(blur_limit=3, p=1),
                            A.MotionBlur(blur_limit=3, p=1),
                        ],
                        p=0.9,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1),
                            A.HueSaturationValue(p=1),
                        ],
                        p=0.9,
                    ) 
                    )
def get_test_transforms(Config=Config):
    
    test_transform = A.Compose([
        A.Resize(width=Config.width, height=Config.height),
        A.Normalize(mean=Config.ADE_MEAN, std=Config.ADE_STD),
    ])

    return test_transform