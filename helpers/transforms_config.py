import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import monai.transforms as trans
from monai.transforms.utils import allow_missing_keys_mode
from .transforms import ConvertToMultiChannel


def get_transforms(label=True):
    """ Get the transform for the dataset and split.
    dataset: str, name of the dataset: 'brainpower', 'brats', 'sailor' or 'embrace'
    split: str, name of the split: 'train', 'val' or 'infer'
    label: bool, whether to transform the label
    resample: bool, whether to resample the image
    """

    # label vs no label 
    if label == True:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    transform = [
        trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
        trans.EnsureTyped(keys=["image"], data_type='tensor', dtype=torch.float16),
        ]

    if label == True:
        transform.insert(0, ConvertToMultiChannel(keys="label"))
    
    # compose transforms
    transform = trans.Compose(transform)
    
    return transform





