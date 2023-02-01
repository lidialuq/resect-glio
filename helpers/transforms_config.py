import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import monai.transforms as trans
rom monai.transforms.utils import allow_missing_keys_mode
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
        trans.CropForegroundd(keys=keys, source_key="image", margin=3, return_coords=False),   
        trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
        trans.SpatialPadd(keys=keys, spatial_size=(128,128,128), mode='constant'),
        trans.EnsureTyped(keys=["image"], data_type='tensor', dtype=torch.float16),
        ]

    if label == True:
        transform.insert(0, ConvertToMultiChannel(keys="label"))
    
    # compose transforms
    transform = trans.Compose(transform)
    
    return transform



def invtrans_prediction(prediction, data):
    transform = [
        trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),   
        trans.SpatialPadd(keys=["image", "label"], spatial_size=(128,128,128), mode='constant'),
        ]

    transformed_data = transform(data)
    prediction.applied_operations = transformed_data["label"].applied_operations
    seg_dict = {"label": prediction}
    with allow_missing_keys_mode(transform):
        inverted_pred = transform.inverse(seg_dict)

    return inverted_pred["label"]


