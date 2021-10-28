import albumentations as A
from albumentations.pytorch import ToTensorV2
#The mean and std I use are the values from the ImageNet dataset
#The augmentations are used to make training harder and more robust to novel situations.
#We don't use augment on the validation set other than normalization to try and estimate the real power of the model in the wild.
Augmentations = {
    'train':
        A.Compose(
            [
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0,),
                ToTensorV2()
            ],
        ),
    'valid':
        A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0,),
                ToTensorV2()
            ],
        ),
    'test':
        A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0,),
                ToTensorV2()
            ],
        )
}
