#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:59:53 2022

@author: lidia
"""
from typing import Optional
import os
import nibabel as nib
import numpy as np
import random
import torch
from torch.utils.data import random_split

from torch.utils.data import Dataset
import time
from glob import glob

def kfold_split(dataset, k=5, use_fold = 0, seed=42):
    """Split subjects into k folds, return list of training and validations 
    subjects with type list(str) using np.array_split.
    k_folds can be used to specify which fold to use for validation, 
    if k_folds=5, then use_fold can be 0,1,2,3,4.
    """
    # make list with k integers as equal as possible that sum to len(dataset)
    splits = np.array_split(np.arange(len(dataset)), k)
    length_splits = [len(split) for split in splits]
    splits = random_split(dataset, length_splits, generator=torch.Generator().manual_seed(seed))

    # concatename all splits except the one we want to use for validation using torch.utils.data.ConcatDataset
    train_ds = []  
    train_indices = []
    for i in range(k):
        if i != use_fold:
            train_ds.append(splits[i])
            train_indices.extend(splits[i].indices)

    train_ds = torch.utils.data.ConcatDataset(train_ds)
    val_ds = splits[use_fold]

    return train_ds, val_ds, train_indices, val_ds.indices


    # assert use_fold <= k, "use_fold should be less than k"
    # subjects.sort()
    # random.seed(seed)
    # random.shuffle(subjects)
    # folds = np.array_split(subjects, k)
    # folds = [list(fold) for fold in folds]
    # training = []
    # validation = []
    # for i in range(k):
    #     validation.append(folds[i])
    #     training.append([item for fold in folds if fold is not folds[i] for item in fold])
    # training = [str(sub) for sub in training[use_fold-1]]
    # validation = [str(sub) for sub in validation[use_fold-1]]
    # return training, validation


class BratsDataset(Dataset):
    """Make dataset for BraTS dataset 2021"""

    def __init__(self, root_dir, transform=None, 
                input=['t1', 't1ce', 'flair', 't2'], substraction=False):
        """
        Args:
            root (string): Directory containing all subjects
            transform (callable, optional): Optional transform to be applied
                on a sample.
            kfold (int): Number of folds to split the dataset into
            use_fold (int): Which fold to use for validation
            input (list): Which sequences to use as input, must be a list of strings
            from ['flair', 't1', 't1ce', 't2']. OBS: if substraction is True, then
            input must contain 't1' and 't1ce' as first and second element.
            substraction: use t1ce - t1 as input
        Returns: 
            dictionary with keys "path", "image", a (C,H,W,D) tensor and "label", 
            a (H,W,D) tensor with:
                0 = background, 1 = necrosis, 2 = edema, 4 = enhancing
        """
        self.root = root_dir
        self.subjects = [i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, i))]     
        self.subjects.sort()   
        self.transform = transform
        self.input = input
        if substraction:
            assert input[0] == 't1' and input[1] == 't1ce', "if substraction is True, then input must contain 't1' and 't1ce' as first and second element."
        self.substraction = substraction
    
    def __len__(self):
        return len(self.subjects)
    
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        t0 = time.time()
        # load sequences into one tensor of shape (C,H,W,D)
        image = []        
        for seq in self.input: 
            nii = nib.load(os.path.join(self.root, self.subjects[idx], self.subjects[idx]+f'_{seq}.nii.gz'))
            image.append(nii.get_fdata())
        if self.substraction:
            t1substraction = image[1] - image[0]
            image.append(t1substraction)
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # load label into one tensor of shape (H,W,D)
        label = nii = nib.load(os.path.join(self.root, self.subjects[idx], self.subjects[idx]+'_seg.nii.gz'))        
        label = torch.from_numpy(nii.get_fdata())
        
        # make dictionary and transform
        sample = {'image': image, 'label': label, 'path': os.path.join(self.root, self.subjects[idx])}
        if self.transform:
            sample = self.transform(sample)
        return sample

class BrainpowerDatasetTSD(Dataset):
    """Brainpower structure as it is in TSD after exporting (either one or
    three dates per patient allowed, if 3 will only use the middle one"""

    def __init__(self, root_dir, data_type, split, transform=None, kfold=None, use_fold=1, input=['T1', 'T1c', 'Flair', 'T2']):
        """
        Args:
            root_dir (string): Directory containing all subjects
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split: 'train', 'val', 'semisup' or 'infer'
            data: 
        Returns: 
            dictionary with keys "image", a (C,H,W,D) tensor and "label", a
            (H,W,D) tensor with:
                0 = background, 1 = edema, 2 = enhancing
        """
        if data_type == 'seg':
            self.root_dir = os.path.join(root_dir, '2016_2018_3D_87seg')
        elif data_type == 'noseg_3D':
            self.root_dir = os.path.join(root_dir, 'not2016_3D_48h')
        elif data_type == 'noseg_2D':
            self.root_dir = os.path.join(root_dir, 'not2016_not3D_48h')
        elif data_type == '52test':
            self.root_dir = os.path.join(root_dir, '2016_2018_3D_52test')
        elif data_type == '35train':
            self.root_dir = os.path.join(root_dir, '2016_2018_3D_35train')
        elif data_type == 'test':
            self.root_dir = os.path.join(root_dir, 'all_test')
        elif data_type == 'all_48h_preop':
            self.root_dir = os.path.join(root_dir, 'all_48h_preop')
        if not data_type:
            self.root_dir = root_dir

        self.subjects = [i for i in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, i))] 
        self.subjects.sort()
        self.input = input

        # sort subjects for reproducibility, then choose split
        if kfold:
            training, val = kfold_split(self.subjects, k=kfold, use_fold = use_fold, seed=42)
            if split == 'train': 
                self.subjects = training
            elif split == 'val':
                self.subjects = val
            elif split == 'semisup' or split == 'infer' or split == 'test':
                self.subjects = self.subjects
        else:
            self.subjects = self.subjects
             
        self.transform = transform
        self.split = split
    
    def __len__(self):
        return len(self.subjects)
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        
        # load sequences into one tensor of shape (C,H,W,D)
        #sequences = ['T1', 'T1c', 'Flair', 'T2']
        image = []
        date = os.listdir(os.path.join(self.root_dir, self.subjects[idx]))
        
        # Pick either only date or middle date if 3 dates or last date id 2 dates
        if len(date) == 1:
            date = date[0]
        elif len(date) == 2:    #OBS PREOP NOW!!!!!!!!!!
            date.sort()
            date = date[0]
        elif len(date) == 3:
            date.sort()
            date = date[1]
            
        for seq in self.input:
            nii = nib.load(os.path.join(self.root_dir, self.subjects[idx], date, seq+'_brain.nii.gz'))
            image.append(nii.get_fdata())

        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # load label into one tensor of shape (H,W,D)
        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            nii_final = os.path.join(self.root_dir, self.subjects[idx], date, 'Final.nii.gz')
            nii = nib.load(nii_final)
            label = torch.from_numpy(nii.get_fdata())
            sample = {'image': image, 'label': label, 'subject': self.subjects[idx], 'date': date,
                      'path': os.path.join(self.root_dir, self.subjects[idx], date), 'root': self.root_dir}
        if self.split == 'semisup' or self.split == 'infer':
            sample = {'image': image, 'subject': self.subjects[idx], 'date': date,
                    'path': os.path.join(self.root_dir, self.subjects[idx], date), 'root': self.root_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ImpressDataset(Dataset):
    """Impress structure as received from Siri"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing all subjects
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Returns: 
            dictionary with keys "image", a (C,H,W,D) tensor and "label", a
            (H,W,D) tensor with:
                0 = background, 1 = edema, 2 = enhancing
        """
        self.root_dir = root_dir

        # get all visit folders from root directory
        self.subjects = [i for i in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, i))] 
        self.subjects.sort()
        self.visits = {'subject':[], 'visit':[], 'visit_path':[]}
        # list all folders in each subjects folder
        for sub in self.subjects:
            visits = [visit for visit in os.listdir(os.path.join(self.root_dir, sub)) if os.path.isdir(os.path.join(self.root_dir, sub, visit))]
            print(visits)
            visits.sort()
            self.visits['subject'].extend([sub]*len(visits))
            self.visits['visit'].extend(visits)
            self.visits['visit_path'].extend([os.path.join(self.root_dir,sub,visit, f'2020_V{visit[-1]}_results', 'mni') for visit in visits])

        self.transform = transform

    def __len__(self):
        return len(self.visits['visit'])
    def __repr__(self):
        return self.visits

    def __getitem__(self, idx):
        
        # load sequences into one tensor of shape (C,H,W,D)
        sequences = ['Flair', 'T1', 'T1c', 'T2']
        image = []
        for seq in sequences: 
            nii = nib.load(os.path.join(self.visits['visit_path'][idx], seq+'.nii.gz'))
            image.append(nii.get_fdata())
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # load label into one tensor of shape (H,W,D)
        sample = {'image': image, 'patient': self.visits['subject'][idx], 'visit': self.visits['visit'][idx],
                'path': self.visits['visit_path'][idx], 'root': self.root_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample

class SailorEmbraceDataset(Dataset):
    """Sailor/Impress structure as received from Jonas (and after preprocessing)"""

    def __init__(self, root_dir, transform=None, dataset='sailor', input=['t1', 't1ce', 'flair', 't2']):
        """
        Args:
            root_dir (string): Directory containing all subjects
            transform (callable, optional): Optional transform to be applied
                on a sample.
            dataset (string): 'sailor' or 'embrace'
        Returns: 
            dictionary with keys "image", a (C,H,W,D) tensor and "label", a
            (H,W,D) tensor with:
                0 = background, 1 = edema, 2 = enhancing
        """
        assert(dataset in ['sailor', 'embrace'])
        self.root_dir = os.path.join(root_dir, dataset)

        # get all visit folders from root directory
        self.subjects = [i for i in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, i))] 
        self.subjects.sort()
        self.transform = transform
        input = [x.lower() for x in input]
        # if input contains 't1c' change to 't1ce'            
        self.sequences = [x if x != 't1c' else 't1ce' for x in input]

    def __len__(self):
        return len(self.subjects)
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        
        # load sequences into one tensor of shape (C,H,W,D)
        image = []
        for seq in self.sequences: 
            nii = nib.load(os.path.join(self.root_dir, self.subjects[idx], seq+'_brain.nii.gz'))
            image.append(nii.get_fdata())
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        label_path = os.path.join(self.root_dir, self.subjects[idx], 'seg.nii.gz')
        nii = nib.load(label_path)
        label = torch.from_numpy(nii.get_fdata())
        
        # load label into one tensor of shape (H,W,D)
        sample = {'image': image, 'label': label, 'subject': self.subjects[idx], 'date': 'unknown',
                'path': os.path.join(self.root_dir, self.subjects[idx]), 'root': self.root_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample


class WMHDataset(Dataset):
    """WMH structure as received from Martin (and after preprocessing)"""

    def __init__(self, root_dir, transform=None, input=['t1', 'flair']):
        """
        Args:
            root_dir (string): Directory containing all subjects
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Returns: 
            dictionary with keys "image", no label
        """

        # get all visit folders from root directory
        self.subjects = [i for i in os.listdir(os.path.join(self.root_dir, 'flair'))] 
        self.subjects.sort()
        self.transform = transform
        self.sequences = input

    def __len__(self):
        return len(self.subjects)
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        
        # load sequences into one tensor of shape (C,H,W,D)
        image = []
        for seq in self.sequences: 
            nii = nib.load(os.path.join(self.root_dir, seq, self.subjects[idx]))
            image.append(nii.get_fdata())
        # make a dummy label of zeros with same shape as image[0]
        label = torch.from_numpy(np.zeros(image[0].shape))
        # stack images
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # load label into one tensor of shape (H,W,D)
        sample = {'image': image, 'label': label, 'subject': self.subjects[idx], 'date': 'unknown',
                'path': os.path.join(self.root_dir, 'flair', self.subjects[idx]), 'root': self.root_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    root_dir = "/scratch/users/lidialu/brainpower"
    train_ds = BrainpowerDatasetTSD(root_dir, split='train', 
                                transform=None, data_type='seg')
    semi3D_ds = BrainpowerDatasetTSD(root_dir, split='semisup', 
                                transform=None, data_type='noseg_3D')
    print(f"Train dataset: {len(train_ds)}, Semi3D dataset: {len(semi3D_ds)}")
