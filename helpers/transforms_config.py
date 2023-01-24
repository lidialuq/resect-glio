import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import monai.transforms as trans
from transforms import ConvertBrainpowerToMultiChannel, ConvertBratsToMultiChannel, ConvertSailorToMultiChannel, RandResampled


def get_transforms(dataset, split, label=True, resample=True, prob=0.15, size=(128,128,128)):
    """ Get the transform for the dataset and split.
    dataset: str, name of the dataset: 'brainpower', 'brats', 'sailor' or 'embrace'
    split: str, name of the split: 'train', 'val' or 'infer'
    label: bool, whether to transform the label
    resample: bool, whether to resample the image
    """

    # check you're not doing anything stupid
    if split == 'val': assert label, "Validation must have a label"

    # label vs no label 
    if label == True:
        keys = ["image", "label"]
        mode_tri = ('trilinear', 'nearest')
        mode_bi = ('bilinear', 'nearest')
    else:
        keys = ["image"]
        mode_tri = ('trilinear')
        mode_bi = ('bilinear')
    

    transform = []

    if split == 'train':
        transform = [
            trans.CropForegroundd(keys=keys, source_key="image", margin=3, return_coords=False),
            # spatial transforms
            trans.RandZoomd(keys=keys, prob=prob, min_zoom=0.9, max_zoom=1.1, mode=mode_tri),
            trans.RandRotated( keys=keys, range_x=0.2, range_y=0.2, range_z=0.2, prob=prob, padding_mode='zeros', mode=mode_bi),
            trans.RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            trans.RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 2)),
            trans.RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(1, 2)),
            trans.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            trans.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            trans.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            # Intensity transforms
            trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=prob),
            trans.RandHistogramShiftd(keys=["image"], num_control_points=(3, 10), prob=prob),
            trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            trans.RandScaleIntensityd(keys=["image"], factors=0.2, prob=prob),
            # pad if needed and crop
            trans.SpatialPadd(keys=keys, spatial_size=size, mode='constant'),
            trans.RandSpatialCropd(keys=keys, roi_size=size, random_center=True, random_size=False),
            trans.EnsureTyped(keys=keys, data_type='tensor', dtype=torch.float16),
            ]
    elif split == 'val':
        transform = [
            trans.CropForegroundd(keys=keys, source_key="image", margin=3, return_coords=False),   
            trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
            trans.SpatialPadd(keys=keys, spatial_size=size, mode='constant'),
            trans.RandSpatialCropd(keys=keys, roi_size=size, random_center=True, random_size=False),
            trans.EnsureTyped(keys=keys, data_type='tensor', dtype=torch.float16),
            ]
    elif split == 'infer':
        transform = [
            trans.CropForegroundd(keys=keys, source_key="image", margin=3, return_coords=False),   
            trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
            trans.SpatialPadd(keys=keys, spatial_size=size, mode='constant'),
            trans.EnsureTyped(keys=["image"], data_type='tensor', dtype=torch.float16),
            ]

    if resample == True and split == 'train':
        transform.insert(1, RandResampled(keys=["image"], prob=prob, mode=('area')))

    if label == True:
        if dataset == "brainpower":
            transform.insert(0, ConvertBrainpowerToMultiChannel(keys="label"))
        elif dataset == "brats":
            transform.insert(0, ConvertBratsToMultiChannel(keys="label"))
        elif dataset == "sailor" or dataset == "embrace":
            transform.insert(0, ConvertSailorToMultiChannel(keys="label"))
        elif dataset == "wmh":
            pass
    
    # compose transforms
    transform = trans.Compose(transform)
    
    return transform






# GRAVEYARD

# transform_train_resample = trans.Compose([
#     ConvertBrainpowerToMultiChannel(keys="label"),
#     trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),
#     RandResampled(keys=["image"], prob=prob, mode=('area')),
#     # spatial transforms
#     trans.RandZoomd(keys=["image", "label"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('trilinear', 'nearest')),
#     trans.RandRotated( keys=['image', 'label'], range_x=0.2, range_y=0.2, range_z=0.2, prob=prob, padding_mode='zeros', mode=("bilinear", 'nearest')),
#     trans.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
#     trans.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
#     trans.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
#     trans.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#     trans.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#     trans.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#     # Intensity transforms
#     trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=prob),
#     trans.RandHistogramShiftd(keys=["image"], num_control_points=(3, 10), prob=prob),
#     trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#     trans.RandScaleIntensityd(keys=["image"], factors=0.2, prob=prob),
#     # pad if needed and crop
#     trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
#     trans.RandSpatialCropd(keys=["image", "label"], roi_size=size, random_center=True, random_size=False),
#     trans.EnsureTyped(keys=["image", "label"], data_type='tensor', dtype=torch.float16),
    
#     ])

# transform_train = trans.Compose([
#     ConvertBrainpowerToMultiChannel(keys="label"),
#     trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),
#     # spatial transforms
#     trans.RandZoomd(keys=["image", "label"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('trilinear', 'nearest')),
#     trans.RandRotated( keys=['image', 'label'], range_x=0.2, range_y=0.2, range_z=0.2, prob=prob, padding_mode='zeros', mode=("bilinear", 'nearest')),
#     trans.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
#     trans.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
#     trans.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
#     trans.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#     trans.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#     trans.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#     # Intensity transforms
#     trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=prob),
#     trans.RandHistogramShiftd(keys=["image"], num_control_points=(3, 10), prob=prob),
#     trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#     trans.RandScaleIntensityd(keys=["image"], factors=0.2, prob=prob),
#     # pad if needed and crop
#     trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
#     trans.RandSpatialCropd(keys=["image", "label"], roi_size=size, random_center=True, random_size=False),
#     trans.EnsureTyped(keys=["image", "label"], data_type='tensor', dtype=torch.float16),
    
#     ])

# transform_train_resample_nolabel = trans.Compose([
#     trans.CropForegroundd(keys=["image"], source_key="image", margin=3, return_coords=False),
#     RandResampled(keys=["image"], prob=prob, mode=('area')),
#     # spatial transforms
#     trans.RandZoomd(keys=["image"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('trilinear')),
#     trans.RandRotated( keys=["image"], range_x=0.2, range_y=0.2, range_z=0.2, prob=prob, padding_mode='zeros', mode=("bilinear")),
#     trans.RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
#     trans.RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
#     trans.RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
#     trans.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
#     trans.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
#     trans.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
#     # Intensity transforms
#     trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=prob),
#     trans.RandHistogramShiftd(keys=["image"], num_control_points=(3, 10), prob=prob),
#     trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#     trans.RandScaleIntensityd(keys=["image"], factors=0.2, prob=prob),
#     # pad if needed and crop
#     trans.SpatialPadd(keys=["image"], spatial_size=size, mode='constant'),
#     trans.RandSpatialCropd(keys=["image"], roi_size=size, random_center=True, random_size=False),
#     trans.EnsureTyped(keys=["image"], data_type='tensor', dtype=torch.float16),
    
#     ])

# transform_train_nolabel = trans.Compose([
#     # spatial transforms
#     trans.RandZoomd(keys=["image"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('trilinear')),
#     trans.RandRotated( keys=['image'], range_x=0.2, range_y=0.2, range_z=0.2, prob=prob, padding_mode='zeros', mode=("bilinear")),
#     trans.RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
#     trans.RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
#     trans.RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
#     trans.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
#     trans.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
#     trans.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
#     # Intensity transforms
#     trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=prob),
#     trans.RandHistogramShiftd(keys=["image"], num_control_points=(3, 10), prob=prob),
#     trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#     trans.RandScaleIntensityd(keys=["image"], factors=0.2, prob=prob),
#     # pad if needed and crop
#     trans.SpatialPadd(keys=["image"], spatial_size=size, mode='constant'),
#     trans.RandSpatialCropd(keys=["image"], roi_size=size, random_center=True, random_size=False),
#     trans.EnsureTyped(keys=["image"], data_type='tensor', dtype=torch.float16),
    
#     ])
# transform_val = trans.Compose([
#     ConvertBrainpowerToMultiChannel(keys="label"),
#     trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),   
#     trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
#     trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
#     trans.RandSpatialCropd(keys=["image", "label"], roi_size=size, random_center=True, random_size=False),
#     trans.EnsureTyped(keys=["image", "label"], data_type='tensor', dtype=torch.float16),
#     ])

# transform_infer = trans.Compose([
#     ConvertBrainpowerToMultiChannel(keys="label"),
#     #trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),   
#     trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
#     #trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
#     trans.EnsureTyped(keys=["image"], data_type='tensor', dtype=torch.float16),
#     ])