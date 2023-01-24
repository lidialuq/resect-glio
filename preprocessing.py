
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:01:49 2022

@author: lidia
"""

import glob
import os
import shutil
import ants
from fsl.wrappers import fslreorient2std
import subprocess
from datetime import datetime
from operator import itemgetter
import sys

""" This script contains all functions needed to preprocess brain MRI by:
    -Reorienting to STD
    -Doing bias correction
    -Resampling to 1,1,1mm voksels
    -Corregistering (affine)
    -Skull-stripping
    
Usage: 
    Before using this script, use copy_seq() in copy_files.py to copy anatomic
    files to new folder for preprocessing.
    Follow Step 1 and Step 2 at the bottom of the script. See move_T1_bet for 
    more info
    
"""

root = '/mnt'
scans = ['t1ce', 't1', 'flair', 't2']
scans_seg = ['t1ce', 't1', 'flair', 't2', 'seg']

# call fslreorient2std from fsl docker
def fsl_reorient(study_folder):
    for seq in scans_seg: 
        nii_path = os.path.join(study_folder, seq+'.nii.gz')
        fslreorient2std(nii_path, nii_path)
            
# Bias correction, resample and co-register
def resample_coregister(study_folder):
    niis = []
    for seq in scans: 
        nii = os.path.join(study_folder, seq+'.nii.gz')
        nii = ants.image_read(nii)
        nii = ants.n3_bias_field_correction(nii)
        niis.append( ants.resample_image(nii,(1,1,1),False,0))
        
    cnt = 1
    for nii in niis[1:]:
        ouput_path = os.path.join(study_folder, f'{scans[cnt]}.nii.gz')
        final = ants.registration(fixed=niis[0], moving=nii, type_of_transform ='Affine')
        ants.image_write(final['warpedmovout'], ouput_path)
        cnt += 1
    ouput_path = os.path.join(study_folder, f'{scans[0]}.nii.gz')
    ants.image_write(niis[0], ouput_path)

    # resample segmentation without registration
    nii = os.path.join(study_folder, 'seg.nii.gz')
    nii = ants.image_read(nii)
    nii = ants.resample_image(nii,(1,1,1),False,0)
    ouput_path = os.path.join(study_folder, f'seg.nii.gz')
    ants.image_write(nii, ouput_path)
    
    
def move_T1_bet(mode, root):
    """"Use to skullstrip using hd-bet. Mode must be either move or unmove. 
    On move mode, moves all T1 files for all patients and in all study folders 
    in root to a tmp folder in root called tmp_bet/i. In unmove mode, moves 
    files back, and also moves files created by hd-bet to a folder called 
    others in the study folder. Usage:
        -From this script, run move_T1_bet('move', root) (root should be in NAS)
        -ssh to miniserver2-0.local
        -conda activate pytorch
        -hd-bet -i [root]/tmp_bet/i -o [root]/tmp_bet/o -device [int]
        -From this script, run move_T1_bet('unmove', root) 
    """    
    study_folders = glob.glob(os.path.join(root, '*'))
    tmp_folder_i = os.path.join(root, 'tmp_bet', 'i')
    tmp_folder_o = os.path.join(root, 'tmp_bet', 'o')

    if mode == 'move':
        os.makedirs(tmp_folder_i)
        os.makedirs(tmp_folder_o)

        for study_folder in study_folders:
            code = os.path.split(study_folder)[1]
            src = os.path.join(study_folder, "t1.nii.gz")
            dst = os.path.join(tmp_folder_i, f't1-{code}.nii.gz' )
            os.rename(src, dst)
    elif mode == 'unmove':
        # Move mask and already brain extracted T1 image to root (in others folder)
        files = glob.glob(os.path.join(tmp_folder_o, '*'))
        for file in files: 
            filename = os.path.basename(file)
            code = filename.split('-')[1].split('.')[0]
            os.makedirs(os.path.join(root, code, 'others'), exist_ok=True)
            if file.endswith('mask.nii.gz'):
                dst = os.path.join(root, code, 'others', 't1_mask.nii.gz')
            else: 
                dst = os.path.join(root, code, 'others', 't1_brain.nii.gz')
            os.rename(file, dst)
        # Move original T1 files back to root
        files = glob.glob(os.path.join(tmp_folder_i, '*'))
        for file in files: 
            filename = os.path.basename(file)
            code = filename.split('-')[1].split('.')[0]
            dst = os.path.join(root, code, 't1.nii.gz')
            os.rename(file, dst)
        
        os.rmdir(tmp_folder_i)
        os.rmdir(tmp_folder_o)
        os.rmdir(os.path.join(root, 'tmp_bet'))
        
def call_hdbet(folder):
    cmd = f'hd-bet -i {folder}/tmp_bet/i -o {folder}/tmp_bet/o -device 0'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if output or error:
        print(output, error)

def move_to_others(study_folder):
    """Move non skull-stripped files to others folder to make place for skull
    stripped"""
    for seq in scans_seg: 
        src = os.path.join(study_folder, seq+'.nii.gz')
        dst = os.path.join(study_folder, 'others', seq+'_head.nii.gz')
        os.rename(src, dst)
    src =  os.path.join(study_folder, 'others', 't1_brain.nii.gz')
    dst = os.path.join(study_folder, 't1_brain.nii.gz')
    os.rename(src, dst)               
    
def apply_mask(study_folder):
    """ Skull-strip by appling mask created by hd-bet"""
    mask = os.path.join(study_folder, 'others', 't1_mask.nii.gz')
    mask = ants.image_read(mask)
    for seq in scans_seg: 
        output_path = os.path.join(study_folder, seq+'_brain.nii.gz')
        nii = os.path.join(study_folder, 'others', seq+'_head.nii.gz')
        nii = ants.image_read(nii)
        brain = ants.mask_image(nii, mask)
        brain_no_background = ants.crop_image(brain, label_image=mask, label=1)
        ants.image_write(brain_no_background, output_path)

############################################################################################

study_folders = glob.glob(os.path.join(root, '*'))

# Do all preprocessing but skull-stripping
# This takes ca 5.5 min per study_folder if 3D, about a minute otherwise. 

print('\n\n' + '*'*120)
print('Starting preprocessing. This will take between 1-6 min per patient depending on the original resolution of the data.')
print('*'*120 + '\n')

for idx, study_folder in enumerate(study_folders):
    print(idx)
    print(study_folder)
    #fsl_reorient(study_folder)
    resample_coregister(study_folder)

# Skull-stripping with hd-bet, including moving images to file format
# that hd-bet can read (and back)

print('\n\n' + '*'*120)
print('Starting skull-stripping. This will take 20-40 sec per patient.')
print('*'*100 + '\n')

move_T1_bet('move', root)
call_hdbet(root)
move_T1_bet('unmove', root)

# Apply brain mask from hd-bet to other sequences

print('\n\n' + '*'*120)
print(' Apply brain mask from hd-bet to other sequences. This will take 5-7 sec per patient.')
print('*'*120 + '\n')

for study_folder in study_folders:
    if not os.path.exists(os.path.join(study_folder, 't1_brain.nii.gz')):
        print(study_folder)
        move_to_others(study_folder)
        apply_mask(study_folder)

