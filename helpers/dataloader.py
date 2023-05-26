#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:59:53 2022

@author: lidia
"""
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class GbmDataset(Dataset):
    """Make dataset for BraTS dataset 2021"""

    def __init__(self, root_dir, label='False', transform=None, 
                input=['t1', 't1ce', 'flair', 't2']):
        """
        Args:
            root_dir (string): Directory containing all subjects
            label (bool): Whether to ground truth label is available
            transform (callable, optional): Optional transform to be applied
                on a sample.
            input (list): Which sequences to use as input, must be a list of strings
            from ['flair', 't1', 't1ce', 't2']. 
        Returns: 
            dictionary with keys "path", "subject", "image" (C,H,W,D) tensor and "label", 
            a (H,W,D) tensor (if mode='val') with:
                0 = background, 1 = enhancing
        """
        self.root = root_dir
        self.subjects = [i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, i))]     
        self.subjects.sort()   
        self.transform = transform
        self.input = input
        self.label = label
    
    def __len__(self):
        return len(self.subjects)
    
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        # load sequences into one tensor of shape (C,H,W,D)
        image = []        
        for seq in self.input: 
            nii = nib.load(os.path.join(self.root, self.subjects[idx], 'preprocessed', f'{seq}.nii.gz'))
            image.append(nii.get_fdata())
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # make dictionary and apply transforms
        if self.label:
            # load label into one tensor of shape (H,W,D)
            label = nii = nib.load(os.path.join(self.root, self.subjects[idx], 'preprocessed', f'seg.nii.gz'))  
            label = torch.from_numpy(nii.get_fdata())
            sample = {'image': image, 'label': label, 'subject': self.subjects[idx], 
                    'path': os.path.join(self.root, self.subjects[idx])}
        else:
            sample = {'image': image, 'subject': self.subjects[idx],
                    'path': os.path.join(self.root, self.subjects[idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample

