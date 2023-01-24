#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:38:56 2022

@author: lidia
"""

import os
import numpy as np
import pickle
from os.path import join
from tqdm import tqdm
import nibabel as nib
import typing
import glob

import torch
from torch.nn import Softmax
from torch.utils.data import DataLoader

from monai.networks.nets import DynUNet
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai import metrics

from helpers.transforms_config import get_transforms
from helpers.dataloader import BrainpowerDatasetTSD
from helpers.network import Network
from preprocessing import move_T1_bet, move_to_others, apply_mask

'''
Predict on Brainpower using an ensamble of 5 supervised or semi-supervised models.
Make sure that the output_classses are the same as the ones used for training, for example:
output_classes = ['edema', 'enhancing'] means edema=1 and enhancing=2 in the prediction. Background is 0.
'''


def load_models(model_list: list, config: dict) -> list:
    """Load models from model_list
    Args:
        model_list (list): list of paths to models
        config (dict): config dictionary
    Returns:
        models (list): list of models
    """
    models = []
    for k in range(len(model_list)):
        if config['semisup']:
            model = Network(len(config['sequences']), config['out_channels'], (128,128,128))
            semisup_model = model_list[k] 
            model.load_state_dict(torch.load(semisup_model, map_location=torch.device('cpu'))['state_dict'])
            model.to(config['device'])
            models.append(model)
        else:
            model = DynUNet(spatial_dims=3,
                            in_channels=len(config['sequences']),
                            out_channels=config['out_channels'],
                            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                            filters=(64, 96, 128, 192, 256, 384, 512),
                            dropout=0,
                            norm_name='INSTANCE',
                            act_name='leakyrelu',
                            deep_supervision=True,
                            deep_supr_num=2,
                            res_block=False,
                            trans_bias=False)
            sup_model = model_list[k]
            model.load_state_dict(torch.load(sup_model, map_location=torch.device('cpu'))['state_dict'])
            model.to(config['device'])
            models.append(model)
            
    return models
            

def infer_one_with_ensable(models: list, data: dict, config: dict) -> list:
    """Infer one image with an ensamble of models
    Args:
        models (list): list of loaded models from load_models()
        data (dict): dict (one batch from dataloader)
        config (dict): config dictionary
    Returns:
        prediction (list(np.array)): list of prediction, ensabled_probabilities. 
            shape is (x,y,z) for both
    """
    # usefull definitions
    inferer = SlidingWindowInferer((128,128,128), sw_batch_size=4, overlap=0.25, mode='gaussian')
    softmax = Softmax(dim=1)
    # do inference for all models, then ensable
    input_volume = data["image"].to(config['device'])
    ensamble = []
    for model in models:
        model.eval()
        # Inference with one model
        with torch.no_grad():
            if config['semisup']:
                output1 = inferer(input_volume.float(), model.branch1)
                output2 = inferer(input_volume.float(), model.branch2)
                output = (softmax(output1) + softmax(output2))/2
            else:
                output = inferer(input_volume.float(), model)
            ensamble.append(output)
    # ensamble
    ensambled_output = torch.mean(torch.stack(ensamble), dim=0)
    output = torch.argmax(ensambled_output, dim=1)
    #output_onehot = one_hot(output.long(), num_classes=config['out_channels']).permute(0, 4, 1, 2, 3).type(torch.float32).cpu()
    prediction = output.squeeze().detach().cpu().numpy().astype('float32') 
    ensambled_output = ensambled_output.squeeze()[1].detach().cpu().numpy().astype('float32') #[1] to exclude background

    return (prediction, ensambled_output) # shapes are (x,y,z)



def save_prediction(prediction, data, config, save_file_name='prediction.nii.gz'):
    """Save prediction to nifti file, save also original images so that they have 
    the same shape as the prediction
    Args:
        prediction (numpy array): prediction from infer_one_with_ensable()
        data (dict): dict (one batch from dataloader)
        config (dict): config dictionary
        save_file_name (str): name to save the prediction, include .nii.gz
    """
    # open header from original image
    original_nii_path = join(data['path'][0], f"T1_brain.nii.gz")
    original_nii = nib.load(original_nii_path)
    # make folder in predictions_folder with subject name
    subject_folder = join(config['save_folder'], data['subject'][0])
    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)
    # save original images (must be saved again because of removing 0s)
    orig_data = data['image'].cpu().detach().numpy().astype('float32')
    print(orig_data.shape)
    for i, seq in enumerate(config['sequences']): 
        img = orig_data[0, i, :, :, :]
        img = nib.Nifti1Image(img, original_nii.affine)
        nib.save(img, join(subject_folder, f"{seq}.nii.gz"))
    # save prediction
    prediction_nii = nib.Nifti1Image(prediction, original_nii.affine, original_nii.header)
    nib.save(prediction_nii, join(subject_folder, save_file_name))


def get_volumes(prediction: np.array, data: dict, metrics_dic:dict) -> dict:
    """Calculate volumes for each class. Output as an update to metrics_dic.
    Args:
        prediction (numpy array): prediction after argmax
        metrics_dic (dict): dict with metrics to update
    Returns:
        metrics_dic (dict): updated dict with metrics to update
    """
    # calculate volumes
    volumes = []
    for i, _ in enumerate(['edema', 'enhancing']):
        volumes.append(np.count_nonzero(prediction == i+1))
    # update metrics_dic
    metrics_dic['edema_volume'].append(volumes[0])
    metrics_dic['enhancing_volume'].append(volumes[1])
    metrics_dic['subjects'].append(data['subject'][0])

    return metrics_dic

def make_save_folder(config):
    """Make save folder for predictions and metrics
    Args:
        config (dict): config dictionary
    """
    # make save folder
    if config['semisup']:
        save_folder = join(config['predictions_folder'], f'semisup')
    else:
        save_folder = join(config['predictions_folder'], f'sup')
    if not os.path.exists(save_folder): os.mkdir(save_folder) 
    return save_folder



# Path to models and config
model_path_enhancing = ['/mnt/CRAI-NAS/all/lidfer/Segmentering/BrainpowerSemisup/saved_models/semisup_97_kX/semisup_97_k0/2022-11-29/epoch_1000/checkpoint-epoch1000.pth',
                '/mnt/CRAI-NAS/all/lidfer/Segmentering/BrainpowerSemisup/saved_models/semisup_97_kX/semisup_97_k1/2022-12-01/epoch_1000/checkpoint-epoch1000.pth',
                '/mnt/CRAI-NAS/all/lidfer/Segmentering/BrainpowerSemisup/saved_models/semisup_97_kX/semisup_97_k2/2022-12-02/epoch_1000/checkpoint-epoch1000.pth',
                '/mnt/CRAI-NAS/all/lidfer/Segmentering/BrainpowerSemisup/saved_models/semisup_97_kX/semisup_97_k3/2022-12-03/epoch_1000/checkpoint-epoch1000.pth',
                '/mnt/CRAI-NAS/all/lidfer/Segmentering/BrainpowerSemisup/saved_models/semisup_97_kX/semisup_97_k4/2022-12-05/epoch_1000/checkpoint-epoch1000.pth']

config = {'device': torch.device('cuda:1'),
          'semisup': True, 
          'sequences': ['T1', 'T1c', 'Flair', 'T2'],
          'predictions_folder': '/mnt/CRAI-NAS/all/lidfer/Segmentering/BrainpowerSemisup/predictions/eduardo_second',
          'data_folder': '/mnt/CRAI-NAS/all/lidfer/Datasets/BrainPower',
          'out_channels': 2,
         }    
config['save_folder'] = make_save_folder(config)

metrics_dic = {'edema_volume': list(),
            'enhancing_volume': list(),
            'subjects': list()}

# Load dataset
transform = get_transforms('brainpower', split='infer', label=False, resample=False)
test_ds = BrainpowerDatasetTSD(config['data_folder'], data_type='all_48h_preop',
                            transform=transform, split='infer',
                            kfold=None, input=['T1', 'T1c', 'Flair', 'T2'])
val_loader = DataLoader(test_ds, batch_size=1, num_workers=4)    


# Do inference
for data in tqdm(val_loader):
    models_edema = load_models(model_path_edema, config)
    models_enhancing = load_models(model_path_enhancing, config)
    prediction_edema = infer_one_with_ensable(models_edema, data, config)
    prediction_enhancing = infer_one_with_ensable(models_enhancing, data, config)
    prediction = combine_predictions(prediction_edema, prediction_enhancing)
    metrics_dic = get_volumes(prediction, data, metrics_dic)
    save_prediction(prediction, data, config, save_file_name='prediction.nii.gz')
    

# Save metrics
with open(join(config['save_folder'], 'test_metrics.pth'), 'wb') as f:
    pickle.dump(metrics_dic, f)
print(metrics_dic)

        
