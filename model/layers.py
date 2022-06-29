from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor, nn
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork,LastLevelP6P7

class BackboneWithFPN(torch.nn.Module):
    '''
    Wrapper for backbone and FPN from torchvision
    '''
    def __init__(self, fpn_channels: int):
        '''
        Args:
        fpn channels: number of output channels of all FPN layers
        '''
        super(BackboneWithFPN, self).__init__()
        # Get a backbone
        _cnn = models.regnet_x_400mf(pretrained=True)
        #_cnn = models.regnet_x_3_2gf(pretrained=True)
        # Extract  layers 
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "p3",
                "trunk_output.block3": "p4",
                "trunk_output.block4": "p5",
            },
        )
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.fpn_channels = fpn_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=fpn_channels,
            #extra_blocks=LastLevelP6P7(in_channels=fpn_channels, out_channels=fpn_channels)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x

class FPN(nn.Module):
    '''
    Feature Pyramid Network FPN that takes feature maps the backbone as 
    (lateral) inputs. All input feature maps are transformed to have equal 
    channels using 1x1 convolutions. Except for the first (top) layer all lower 
    layers with higher resolution are summed with an upsampled version of the 
    layer above. Each Layer is run through another 3x3 Convolution to yield the 
    final pyramid layer. The present version uses three input input feature maps 
    (C3-C5; bottom to top) and three pyramid layers (P5-P3; top to bottom). 
    P5 is obtained directly from C5 run through a 1x1 conv layer. P4 and P3 are 
    additionally summed with upsampled versions of P5 and P4 respectively.
    Note that the original FCOS implementation uses two more layers at the top 
    (P6, P7). These are omitted here.
    '''
    def __init__(self,fpn_channels: int):
        '''
        Args:
        fpn channels: number of output channels of all FPN layers
        '''
        super(FPN,self).__init__()
        # 1x1 convs to equalize chennels across feature maps
        self.conv_C3 = nn.Conv2d(
            64, fpn_channels, kernel_size=1, padding = 'same'
        )
        self.conv_C4 = nn.Conv2d(
            160, fpn_channels, kernel_size=1, padding = 'same'
        )
        self.conv_C5 = nn.Conv2d(
            400, fpn_channels, kernel_size=1, padding = 'same'
        )
        # 3x3 convs producing the final pyramid layers
        self.conv_P3 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding = 'same'
        )
        self.conv_P4 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding = 'same'
        )
        self.conv_P5 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding = 'same'
        )
        #self.relu =  nn.ReLU()

    def forward(self,C3: Tensor, C4: Tensor, C5: Tensor) -> Dict[str,Tensor]:
        '''
        Args:
        C3,C4,C5: feature maps extracted from backbone

        Returns: 
        features: dictionary containing the feature maps of the FPN
                  {'p3': P3, 'p4': P4, 'p5': P5}
        '''
        # equalize channels of backbone inputs
        P5 = self.conv_C5(C5)
        P4 = self.conv_C4(C4)
        P3 = self.conv_C3(C3)
        # add upsampled upper layers to lower layers
        P4 = P4 + torch.nn.functional.interpolate(P5, scale_factor=2)
        P3 = P3 + torch.nn.functional.interpolate(P4, scale_factor=2)
        # produce final pyramid layers
        P3 = self.conv_P3(P3)
        P4 = self.conv_P4(P4)
        P5 = self.conv_P5(P5)
        # gather pyramid layers in dict and return
        features = {'p3': P3, 'p4': P4, 'p5': P5}
        return features