#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:24:48 2022

@author: lidia
"""

import torch
from monai.transforms import MapTransform

"""
Transforms, who would have guessed.
"""

classes = ['enhancing'] #['edema', "enhancing"], ["edema"]

class ConvertToMultiChannel(MapTransform):
    """
    Convert label from (H,W,D) to (C,H,W,D) where the channels correspond to
    (background, enhancing). Converts label to float16.
    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            # add background inc necrosis
            if classes == ["enhancing"]:
                result.append(d[key] == 0)  # background
                result.append(d[key] == 1)  # add enhancing
            d[key] = torch.stack(result, axis=0).type(torch.HalfTensor)
        return d

