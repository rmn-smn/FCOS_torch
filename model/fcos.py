from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import math
from torch import Tensor, nn
import torchvision.ops
from model.layers import BackboneWithFPN
from model.loss import FCOSLoss

class Head(nn.Module):
    '''
    Prediction Head of the FCOS detector, consisting of two stems, 
    for classification and shared bounding box and centerness regression.
    Each stem consists of a specified number of 3x3 convolutions with 
    specified outpout channels and a final 3x3 convolution with either
    4 output channels (bounding box regression)
    1 output channel (centerness regression)
    num_classes channels (classification)
    Each feature map is passed through the stems to yield the final predictions.
    All feature maps share the same weights. 
    '''
    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        '''    
        Args:
        num_classes:   number of classes output by the classification head
        fpn_channels:  output channels of the FPN network
        stem_channels: list of ints defining the numbers of conv layers and 
                       their output channels for the stems      
        '''
        super().__init__()

        # classification and boxreg/centerness stem layer list
        stem_cls = []
        stem_box = []

        # fill stem layer list. 
        # select depth and channels accoring to stem_channels.
        # Pattern: i x (3x3 Conv -> ReLu)
        self.num_classes = num_classes
        for i in range(len(stem_channels)):
            if i == 0:
                input = fpn_channels
            else:
                input = stem_channels[i-1]

            stem_cls.append(
                nn.Conv2d(
                    in_channels = input, 
                    out_channels = stem_channels[i], 
                    kernel_size=3, 
                    stride = 1, 
                    padding=1,
                    bias = True
                )
            )
            stem_cls.append(nn.ReLU())

            stem_box.append(
                nn.Conv2d(
                    in_channels = input, 
                    out_channels = stem_channels[i], 
                    kernel_size=3, 
                    stride = 1, 
                    padding=1,
                    bias = True
                )
            )
            stem_box.append(nn.ReLU())

        # add layers list as modules
        self.add_module('stem_cls', nn.Sequential(*stem_cls))
        self.add_module('stem_box', nn.Sequential(*stem_box))

        # final prediction layers for classificaion, box and centerness
        # regression
        self.pred_cls =  nn.Conv2d(
                in_channels = fpn_channels, 
                out_channels = num_classes, 
                kernel_size=3, 
                stride = 1, 
                padding=1,
            )  # Class prediction conv
        self.pred_box = nn.Conv2d(
                in_channels = fpn_channels, 
                out_channels = 4, 
                kernel_size=3, 
                stride = 1, 
                padding=1,
            )  # Box regression conv
        self.pred_ctr = nn.Conv2d(
                in_channels = fpn_channels, 
                out_channels = 1, 
                kernel_size=3, 
                stride = 1, 
                padding=1,
            )  # Centerness conv

        # initialise modules with bias = 0 and weights with std = 0.01
        # initialise negative classification bias for focal loss
        # ref: https://arxiv.org/abs/1708.02002
        for modules in [self.stem_cls, self.stem_box,
                        self.pred_cls, self.pred_box,
                        self.pred_ctr]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        # at the beginning of training all class predictions are labelled 
        # foreground with a confidence of approx. pi. This prevents a large 
        # number of backgound predictions pollute the loss at the start of 
        # training
        pi = 0.01
        torch.nn.init.constant_(self.pred_cls.bias, -math.log((1-pi)/pi))
        
    def forward(
        self,features_all_levels: Dict[str, Tensor]
    ) -> Tuple[List[Tensor]]:
        '''    
        Args:
        features_all_levels: dictionary containg the feature maps of the FPN

        Returns:
        Tuple of Dicts contaning the predicted tensors of the Head
        Dict style: {'p3': Tensor, 'p4': Tensor, 'p5': Tensor}
        boxreg_deltas -> Tensor dims: (Batch, Channels, Height, Width, 4)
        centerness_logits -> Tensor dims: (B, C, H, W, )
        class_logits -> Tensor dims: (B, C, H, W, num_classes)
        '''
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        # iterate over feature levels
        for i,(level,feature) in enumerate(features_all_levels.items()):
            # pass feature map through stems
            class_logits_one_level = self.stem_cls(feature)
            boxreg_deltas_one_level = self.stem_box(feature)
            centerness_logits_one_level = self.stem_box(feature)

            # pass stem outputs thorugh final prediction layers
            class_logits_one_level = self.pred_cls(class_logits_one_level)
            boxreg_deltas_one_level = self.pred_box(boxreg_deltas_one_level)
            centerness_logits_one_level = self.pred_ctr(centerness_logits_one_level)

            h,w = feature.shape[2:]
            # permute channels to dim=-1
            # reshape bboxes, classes, center-ness to collapse h,w
            centerness_logits_one_level = torch.reshape(
                centerness_logits_one_level.permute([0,2,3,1]), [-1, h * w, 1]
            )
            boxreg_deltas_one_level = torch.reshape(
                boxreg_deltas_one_level.permute([0,2,3,1]), [-1, h * w, 4]
            )
            class_logits_one_level = torch.reshape(
                class_logits_one_level.permute([0,2,3,1]), 
                [-1, h * w, self.num_classes]
            ) 

            # gather predictions of each level in dict
            boxreg_deltas[level] = boxreg_deltas_one_level
            class_logits[level] = class_logits_one_level
            centerness_logits[level] = centerness_logits_one_level   

        return boxreg_deltas,centerness_logits,class_logits

class FCOS(nn.Module):
    '''
    FCOS main network to perform anchorless object detection on input images. 
    This module sets up the backbone feature extractor, feature pyramid 
    network (FPN) and loss object. Currently a RegNetX-400MF from torchvision 
    is used but any Resnet-like feature extractor should be usable as well, 
    as long as intermediate feature maps at different levels can be extracted 
    to fed them to the FPN
    '''
    def __init__(
      self,
      num_classes: int,
      fpn_strides: Dict[str, int],
      fpn_channels: int = 256, 
      stem_channels: list[int] = [64, 64],
      device: str = 'cpu'   
    ):
        '''
        Args:
        num_classes:   number of object classes for classification
        fpn_channels:  number of output channels of the FPN layers
        stem_channels: list of numbers defining the output channels of each 
                       stem layer (and implicitly the number of layers)
        device:        device to use for computations ('cpu' or 'cuda')
        '''
        super(FCOS,self).__init__()
        
        self.device = device

        # define backbone network and extract feature maps
        # self.backbone = Backbone(num_classes,classification=False)
        # _cnn = models.regnet_x_400mf(pretrained=True)
        # self.backbone = feature_extraction.create_feature_extractor(
        #     _cnn,
        #     return_nodes={
        #         "trunk_output.block2": "c3",
        #         "trunk_output.block3": "c4",
        #         "trunk_output.block4": "c5",
        #     },
        # )
        # # define FPN, Head, and Loss
        # self.fpn = FPN(fpn_channels)

        self.backbone_fpn = BackboneWithFPN(fpn_channels)
        self.head = Head(num_classes,fpn_channels,stem_channels)
        self.fcos_loss = FCOSLoss(fpn_strides,num_classes,device)

        self.num_classes = num_classes
        self.h = None
        self.w = None
        self.fpn_strides = fpn_strides

        self.fpn_locations = {
            key: None for key in self.fpn_strides.keys()
        }
      
    def forward(
        self,
        images: Tensor,
        targets: Tensor = Optional[None],
        test_score_thresh: Optional[float] = 0.3, 
        test_nms_thresh: Optional[float] = 0.5,
      ):
        """
        Args:
        images:  Batch of images as tensors of dim: (B, C, H, W).
        targets: Batch of object targets for training as tensors of 
                 dim: (B, N, 5).
                 Each target N_i has format (x0, y0, x0, y0, C). (x0,y0),(x1,y1) 
                 are lower and upper coordinates of object bounding box in 
                 pixels. C is an integer representing the object class.
                 Only provided during Training.
        test_score_thresh: Remove predictions with scores lower than this value.
                           Only neccessary during inference
        test_nms_thresh:   IoU threshold for nms during inference. Only 
                           neccessary during inference

        Returns:
        Losses during training and predictions during inference.
        """       

        # get input height and width
        self.h,self.w = images.shape[1:3]
     
        # run backbone and extract feature maps
        # backbone_feats = self.backbone(images)
        # C5 = backbone_feats['c5']
        # C4 = backbone_feats['c4']
        # C3 = backbone_feats['c3']

        # # run feature pyramid network
        # self.fpn_features = self.fpn(C3, C4, C5) 

        self.fpn_features = self.backbone_fpn(images)
        # generate feature map locations (coordinates of the maps on the image)
        for level in self.fpn_features.keys():
          location = self.generate_locations(
              self.fpn_features[level],
              self.fpn_strides[level]
          )
          self.fpn_locations[level] = location

        # run feature maps through head to make preditions
        pred_bbox_dict,pred_cent_dict,pred_cls_dict = self.head(
            self.fpn_features
        )
        # during training compute losses, during inference post process 
        # predictions for output
        if self.training:
            return self.fcos_loss(
                pred_bbox_dict,pred_cent_dict,pred_cls_dict,
                targets,self.fpn_locations
            )
        else:
            return self.forward_infer(
                images,
                pred_cls_dict, pred_bbox_dict, pred_cent_dict,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
        
    def forward_infer(
        self,
        image: Tensor,
        pred_cls_logits: Dict[str,Tensor],
        pred_boxreg_deltas: Dict[str,Tensor],
        pred_ctr_logits: Dict[str,Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (Batch = 1). 

        Args:
        image: input image

        Returns:
        pred_boxes:   Dict of Tensors of dim: (N, 4) with (x0,y0,x1,y1) pixel 
                      coordinates of predicted boxes.      
        pred_classes: Dict of Tensors of dim: (N, ) with predicted class labels for 
                      these boxes
        pred_scores:  Dict of Tensors of dim: (N, ) with predicted scores for 
                      these boxes
                    All dict styles: {'p3': Tensor, 'p4': Tensor, 'p5': Tensor}
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in self.fpn_locations.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid() * level_ctr_logits.sigmoid()
            )

            #get the maximum score and highest probable class per predictions
            top_scores, top_scores_indices = level_pred_scores.max(dim=1)
            top_classes = top_scores_indices.to(torch.int32)

            # convert deltas to image coordinates
            level_boxes = self.delta2coords(level_name,level_deltas)

            # retain predictions with score above threshold
            level_pred_classes = top_classes[top_scores >= test_score_thresh]
            level_pred_boxes = level_boxes[top_scores >= test_score_thresh]   
            level_pred_scores = top_scores[top_scores >= test_score_thresh]  

            # box coordinates should reamin inside image. clip boxes
            h,w = image.shape[2:]
            level_pred_boxes[:, 0] = torch.clamp(level_pred_boxes[:, 0] , 0, h)
            level_pred_boxes[:, 1] = torch.clamp(level_pred_boxes[:, 1] , 0, w)
            level_pred_boxes[:, 2] = torch.clamp(level_pred_boxes[:, 2] , 0, h)
            level_pred_boxes[:, 3] = torch.clamp(level_pred_boxes[:, 3] , 0, w)

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = self.class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]

        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )

    def generate_locations(self,feature,stride):
        h,w = feature.shape[2:]
        x = torch.arange(0,w*stride,stride) + stride//2
        y = torch.arange(0,h*stride,stride) + stride//2
        X,Y = torch.meshgrid(x,y)
        return torch.reshape(torch.stack([X,Y],dim = -1),[-1,2]).to(torch.float32)

    def delta2coords(self,level,boxes):
        # convert normalized box deltas to image coordinates
        x,y = self.fpn_locations[level][:,0],self.fpn_locations[level][:,1]
        stride = self.fpn_strides[level]
        # we need to set backgorund boxes (-1) to zero so that the correct
        # correct location is retrieved (exactly (x,y))
        mask = torch.where(boxes[:,:4]<0,1,0)
        boxes[mask==1]=0
        boxes[:,0] = x - boxes[:,0]*stride #x_0
        boxes[:,1] = y - boxes[:,1]*stride #y_0
        boxes[:,2] = boxes[:,2]*stride + x #x_1
        boxes[:,3] = boxes[:,3]*stride + y #y_1     

        # clip negative values to zero   
        return torch.clip(boxes,0,None)

    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
        """
        Non-maximum suppression removes overlapping bounding boxes.

        Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

        Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
        """

        if (not boxes.numel()) or (not scores.numel()):
            return torch.zeros(0, dtype=torch.long)

        keep = None
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)

        return keep


    def class_spec_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float = 0.5,
    ):
        """
        Wrap `nms` to make it class-specific. 

        Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        max_coordinate = boxes.max()
        offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.nms(boxes_for_nms, scores, iou_threshold)
        return keep

if __name__ == '__main__':
    image = torch.rand(2, 3, 224, 224)
    target = torch.tensor(
        [[[0,0,1,1,2,0.5],[0,0,0.5,0.5,5,0.5]],
        [[0,0,1,1,2,0.5],[0,0,0.5,0.5,5,0.5]]]
    )
    nclass = 10
    fpn_channels = 128
    stem_channels = [128,128]
    fpn_strides = {
        'p3': 8,
        'p4': 16,
        'p5': 32,
    }

    fcos = FCOS(nclass,fpn_strides,fpn_channels,stem_channels)
    loss = fcos(image,target)

