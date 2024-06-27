import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
import os
import cv2
import numpy

from .submodel import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU

savepath='/share/E2VID_share/MVSEC_share/outdoor_day2/layer_output'
'''
Each residual 이 더하는 지 혹은 합쳐지는 지.
'''
def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def skip_sum(x1, x2):
    return x1 + x2

def save_output(output,savepath,i):
    if type(output)==tuple:
        output=output[0]
        filename='encoder'+str(i)+'.png'
    else:
        filename='decoder'+str(i)+'.png'

    tmp=output.cpu().detach().numpy()
    tmp=numpy.transpose(tmp,(0,2,3,1))
    tmp=numpy.squeeze(tmp,0)

    max=numpy.max(tmp)
    min=numpy.min(tmp)
    tmp=(tmp-min)/(max-min)
    tmp=tmp*255

    cv2.imwrite(os.path.join(savepath,filename),tmp[:,:,0])
    

class BaseUNet(nn.Module):
    def __init__(self, 
                 num_input_channels, 
                 num_output_channels=1, 
                 skip_type='sum', 
                 activation='sigmoid',
                 num_encoders=4, 
                 base_num_channels=32, 
                 num_residual_blocks=2, 
                 norm=None, 
                 use_upsample_conv=True):
        
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm
        
        # upsampling method
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders) # encoder layer를 지날 때마다 chennel의 개수는 2배가 됨.

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        # 각 encoder layer에서의 input channel 개수 저장
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        # 각 encoder layer에서의 output channel 개수 저장
        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        # activation function
        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList() # nn.Module을 저장
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm)) # channel 수 변화 X, encoder를 다 거치고 난 뒤에 residual block 

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)])) # 각 encoder layer에서의 input channel 개수 저장 (각 encoder layer에서의 output channel 개수와 대응) 

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            # UpsampleConvLayer (in_channels, out_channels, kernel_size, stried, padding, activation, norm)
            # TransposedConvLayer (in_channels, out_channels, kernel_size, stried, padding, activation, norm)
            # channel의 개수를 반으로 줄임.
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)



class UNet(BaseUNet):
    def __init__(self, 
                 num_input_channels, 
                 num_output_channels=1, 
                 skip_type='sum', 
                 activation='sigmoid',
                 num_encoders=4, 
                 base_num_channels=32, 
                 num_residual_blocks=2, 
                 norm=None, 
                 use_upsample_conv=True):
        
        super(UNet, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                   num_encoders, base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        # stem
        # 2D conv layer with 5x5 kernel size, 32 output channel (does not reduce height/width)
        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            # 2D conv layer with 5x5 kernel size, reduce height, width by half
            self.encoders.append(ConvLayer(input_size, output_size, kernel_size=5,
                                           stride=2, padding=2, norm=self.norm))

        self.build_resblocks()
        self.tail=ConvLayer(self.encoder_output_sizes[-1],num_output_channels,kernel_size=3,stride=1,padding=2)
        # self.build_decoders()
        # self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # # decoder
        # for i, decoder in enumerate(self.decoders):
        #     x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        x=self.tail(x)
        return x


    
class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, 
                 num_input_channels, 
                 num_output_channels=1, 
                 skip_type='sum',
                 recurrent_block_type='convlstm', 
                 activation='sigmoid', 
                 num_encoders=4, 
                 base_num_channels=32,
                 num_residual_blocks=2, 
                 norm=None, 
                 use_upsample_conv=True):
        
        super(UNetRecurrent, self).__init__(num_input_channels, 
                                            num_output_channels, 
                                            skip_type, 
                                            activation,
                                            num_encoders, 
                                            base_num_channels, 
                                            num_residual_blocks, 
                                            norm,
                                            use_upsample_conv)

        
        self.head = ConvLayer(self.num_input_channels, 
                              self.base_num_channels,
                              kernel_size=5, 
                              stride=1, 
                              padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(input_size, 
                                                    output_size,
                                                    kernel_size=5, 
                                                    stride=2, 
                                                    padding=2,
                                                    recurrent_block_type = recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.tail=ConvLayer(self.encoder_output_sizes[-1],num_output_channels,kernel_size=3,stride=1,padding=2)
        # self.build_decoders()
        # self.build_prediction_layer()

    def forward(self, x, prev_states): ## x : [1,5,256,256] ## x : [1,5,260,346]
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # Head with Conv
        x = self.head(x)
        head = x

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # Encoder
        blocks = [ ]
        states = [ ]
        
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i]) # hidden state, tuple (hidden state, cell state)
            blocks.append(x) # hidden state
            states.append(state)
            #save_output(state,savepath,i)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)
        
        x=self.tail(x)
        return x, states