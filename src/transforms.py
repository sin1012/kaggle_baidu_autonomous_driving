import albumentations as A
def get_train_transforms():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(),
        ], p=0.2),
        A.CLAHE(clip_limit=(1,4), p=0.2),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
    ])