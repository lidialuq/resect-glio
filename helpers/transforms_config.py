import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import monai.transforms as trans
from monai.transforms.utils import allow_missing_keys_mode
from .transforms import ConvertToMultiChannel

"""
Put together al transforms 
"""

def get_transforms(label=True):
    """ Choose transforms according to label availability
    label: bool, whether to transform the label
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





