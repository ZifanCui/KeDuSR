
import torch
import torchvision
from distutils.version import LooseVersion
from torch import nn as nn
from torch.nn import init as init
from .dcn import ModulatedDeformConvPack, modulated_deform_conv


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 250:
            # logger = get_root_logger()
            # logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
            print(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)



class FlowGuidedDCN(ModulatedDeformConvPack):
    '''Use other features to generate offsets and masks'''


    def forward(self, x, feat, flows):
        '''input: input features for deformable conv: N, C, H, W.
           fea: other features used for generating offsets and mask: N, C, H, W.
           flows: N, 2, H, W.
        '''
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)

        offset = torch.tanh(torch.cat((o1, o2), dim=1)) * 15 # max_residue_magnitude
        offset = offset + flows.flip(1).repeat(1, offset.size(1)//2, 1, 1)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 250:
            print('FlowGuidedDCN: Offset mean is {}, larger than 100.'.format(offset_mean))
            # offset = offset.clamp(-50, 50)
            # return None

        
        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)