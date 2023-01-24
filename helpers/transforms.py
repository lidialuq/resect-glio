#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:24:48 2022

@author: lidia
"""

import torch
import numpy as np
from monai.transforms import MapTransform, Transform
import monai.transforms as trans
import random

"""
Transforms, who would have guessed.
"""

classes = ['enhancing'] #['edema', "enhancing"], ["edema"]

class ConvertBratsToMultiChannel(MapTransform):
    """
    Convert label from (H,W,D) to (C,H,W,D) where the channels correspond to
    (background, edema, enhancing, necrosis). Converts label to float16.
    The original brats from 2021 data is
    0 = background, 1 = necrosis, 2 = edema, 4 = enhancing
    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            # add background inc necrosis
            if classes == ["enhancing"]:
                result.append(np.logical_or(np.logical_or(d[key] == 0, d[key] == 1), d[key] == 2))
                result.append(d[key] == 4)  # add enhancing
            elif classes == ["edema"]:
                result.append(np.logical_or(np.logical_or(d[key] == 0, d[key] == 1), d[key] == 4))
                result.append(d[key] == 2)  # add edema
            elif classes == ['edema', 'enhancing']:
                result.append(np.logical_or(d[key] == 0, d[key] == 1))
                result.append(d[key] == 2)  # add edema
                result.append(d[key] == 4)  # add enhancing
            elif classes == ['edema', 'enhancing', 'necrosis']:
                result.append(np.logical_or(d[key] == 0))
                result.append(d[key] == 2)  # add edema
                result.append(d[key] == 4)  # add enhancing
                result.append(d[key] == 1)  # add necrosis
            d[key] = torch.stack(result, axis=0).type(torch.HalfTensor)
        return d


class ConvertBrainpowerToMultiChannel(MapTransform):
    '''Brainpower data is: 0 = background, ink necrosis, 1 = edema, 2 = ehancing'''

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            if classes == ['edema', 'enhancing']:
                result.append(d[key] == 0)  # background inc necrosis
                result.append(d[key] == 1)  # edema
                result.append(d[key] == 2)  # enhancing
            elif classes == ['enhancing']:
                result.append(np.logical_or(d[key] == 0, d[key] == 1))  # background inc necrosis and edema
                result.append(d[key] == 2)  # enhancing
            elif classes == ['edema']:
                result.append(np.logical_or(d[key] == 0, d[key] == 2))  # background inc necrosis and enhancing
                result.append(d[key] == 1)  # edema

            d[key] = torch.stack(result, axis=0).type(torch.HalfTensor)

        return d

class ConvertSailorToMultiChannel(MapTransform):
    '''Sailor/embrace data is: 0 = background, 1 = blood, 2 = kontrastladende vev som ikke er tumor, 3 = resttumor
    Her slår vi sammen 2 og 3 til en klasse (contrast enhancing), og 1 og 0 til en klasse (background)'''

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            result.append(np.logical_or(d[key] == 0, d[key] == 1)) #background is background + blood
            result.append(np.logical_or(d[key] == 2, d[key] == 3)) #enhancing is contrast enhancing though to not be tumor + resttumor
            d[key] = torch.stack(result, axis=0).type(torch.HalfTensor)

        return d

class RandResampled(MapTransform):
    """
    Resample image and label to simulate 2D MRI 
    """

    def __init__(self, keys, prob=0.5, mode=('area', 'nearest')):
        super().__init__(keys)
        self.prob = prob
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        original_shape = d[self.keys[0]].shape
        original_shape = original_shape[-3:]
        if random.random() < self.prob:
            resample_axis = random.randint(0, 2)
            if resample_axis == 0:
                spatial_size = (random.randint(20, 60), -1, -1)
            elif resample_axis == 1:
                spatial_size = (-1, random.randint(20, 60), -1)
            elif resample_axis == 2:
                spatial_size = (-1, -1, random.randint(20, 60))

            resample = trans.Compose([
                trans.Resized(keys=self.keys, spatial_size=spatial_size,
                                mode=self.mode, size_mode='all'),
                trans.Resized(keys=self.keys, spatial_size=original_shape,
                                mode=self.mode, size_mode='all')
            ])

            d = resample(d)

        return d

