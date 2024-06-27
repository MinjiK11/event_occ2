import logging

import torch.nn as nn
import torch
from os.path import join
import numpy as np

from .submodel import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer
from .unet import UNet, UNetRecurrent
from mmdet.models.builder import BACKBONES


#5 (bins)
# conf = {'num_bins': 5,
#  'skip_type': 'sum',
#  'recurrent_block_type': 'convlstm',
#  'num_encoders': 3,
#  'base_num_channels': 32,
#  'num_residual_blocks': 2,
#  'use_upsample_conv': False,
#  'norm': 'BN'}
# num_encoders = 3..?

# @BACKBONES.register_module()
# class BaseModel(nn.Module):
#     """
#     Base class for all models
#     """
#     def __init__(self):
#         super(BaseModel, self).__init__()
#         #self.config = config
#         self.logger = logging.getLogger(self.__class__.__name__)

#     def forward(self, *input):
#         """
#         Forward pass logic

#         :return: Model output
#         """
#         raise NotImplementedError

#     def summary(self):
#         """
#         Model summary
#         """
#         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         self.logger.info('Trainable parameters: {}'.format(params))
#         self.logger.info(self)
        
class BaseE2VID(nn.Module):
    def __init__(
        self,
        num_bins,
        skip_type,
        num_encoders,
        base_num_channels,
        num_residual_blocks,
        use_upsample_conv,
        norm,
        recurrent_block_type
    ):
        super().__init__()
        self.num_bins = num_bins  # number of bins in the voxel grid event tensor
        self.skip_type=skip_type
        self.num_encoders=num_encoders
        self.base_num_channels=base_num_channels
        self.num_residual_blocks=num_residual_blocks
        self.use_upsample_conv=use_upsample_conv
        self.norm=norm
        self.recurrent_block_type=recurrent_block_type

@BACKBONES.register_module()
class E2VID(BaseE2VID):
    def __init__(self,
                num_bins,
                skip_type,
                num_encoders,
                base_num_channels,
                num_residual_blocks,
                use_upsample_conv,
                norm,
                recurrent_block_type):
        super(E2VID, self).__init__(
                num_bins,
                skip_type,
                num_encoders,
                base_num_channels,
                num_residual_blocks,
                use_upsample_conv,
                norm,recurrent_block_type)

        self.unet = UNet(num_input_channels=self.num_bins,
                         num_output_channels=128,
                         skip_type=self.skip_type,
                         activation='sigmoid',
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor), None

@BACKBONES.register_module()
class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self,
                 num_bins,
                skip_type,
                num_encoders,
                base_num_channels,
                num_residual_blocks,
                use_upsample_conv,
                norm,
                recurrent_block_type):
        super(E2VIDRecurrent, self).__init__(num_bins,
        skip_type,
        num_encoders,
        base_num_channels,
        num_residual_blocks,
        use_upsample_conv,
        norm,
        recurrent_block_type)

        # try:
        #     self.recurrent_block_type = str(config['recurrent_block_type'])
        # except KeyError:
        #     self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=128,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states
