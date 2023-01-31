# encoding: utf-8
from typing import Callable, Dict, Union, Tuple, List, Optional

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

class Network(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 ):
        super(Network, self).__init__()

        self.branch1 = DynUNet(spatial_dims=3,
                        in_channels=in_channels,
                        out_channels=out_channels,
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

        self.branch2 = DynUNet(spatial_dims=3,
                        in_channels=in_channels,
                        out_channels=out_channels,
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


    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            pred2 = self.branch2(data)
            return (pred1, pred2)

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
        


if __name__ == '__main__':
    # test that everything runs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Network(4, 4, (128,128,128)).to(device)
    left = torch.randn(2, 4, 128, 128, 128).to(device)
    right = torch.randn(2, 4, 128, 128, 128).to(device)
    out = model(left)
    print(f'Output shape: {out.shape}')