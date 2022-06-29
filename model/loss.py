from typing import Dict
import torch
from torch import Tensor, nn
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

class FCOSLoss():
    '''
    Loss Object for FCOS detector. Encodes training targets on the feature map
    and computes losses for classification, bounding box and centerness 
    regression.
    '''
    def __init__(
        self,
        fpn_strides: Dict[str, int],
        num_classes:  int,
        device: str = 'cpu'
    ):
        '''
        Args:
        fpn_strides: Dictionary containing the stride of each fpn layer
                     dict style: {'p3': int, 'p4': int, 'p5': int}
        num_classes: number of classes used for classification
        device:      device to use for computations ('cpu' or 'cuda')
        '''
        super().__init__()
        self.device = device

        self.fpn_strides = fpn_strides
        self.fpn_locations = {}
        # fpn distances give thresholds for assigning labels on the feature 
        # maps in the loss object
        self.fpn_distances = \
            {'m'+key[1:]: value*8 for (key, value) in fpn_strides.items()}
        self.fpn_distances['m2']=0 
        self.num_classes = num_classes       
        # loss normalizer
        self._normalizer = 150


    def __call__(
        self,
        pred_reg_dict: Dict[str, Tensor],
        pred_ctr_dict: Dict[str, Tensor],
        pred_cls_dict: Dict[str, Tensor],
        targets,
        locations
    ):
        '''
        Encode training targets on the feature map and computes losses for 
        classification, bounding box and centerness regression.

        Args:
        pred_reg_dict: Dict of Tensors of dim: (N, 4) with (x0,y0,x1,y1) pixel 
                       coordinates of predicted boxes.      
        pred_ctr_dict: Dict of Tensors of dim: (N, ) with predicted class labels 
                       for these boxes
        pred_cls_dict: Dict of Tensors of dim: (N, ) with predicted scores for 
                       these boxes
                    All dict styles: {'p3': Tensor, 'p4': Tensor, 'p5': Tensor} 
        targets:        Batch of object targets for training as tensors of 
                        dim: (B, N, 5).
                        Each target N_i has format (x0, y0, x0, y0, C). 
                        (x0,y0),(x1,y1) are lower and upper coordinates of 
                        object bounding box in pixels. C is an integer 
                        representing the object class.
        locations:      dict containing feature map locations (coordinates of 
                        the maps on the image)
                        dict style: {'p3': Tensor, 'p4': Tensor, 'p5': Tensor}
        '''

        
        batch_size = targets.shape[0]
        encoded_targets_list = []

        # encode targets seperately for every image in batch
        for i in range(batch_size):
            encoded_targets_all_levels = self.encode_targets(
                targets[i,...],locations
            )
            encoded_targets_list.append(encoded_targets_all_levels) 
        
        # convert list of box dictionaries to dictionary of batched boxes
        encoded_targets = default_collate(encoded_targets_list)
        # concatenate all targets levels into one tensor
        encoded_targets = torch.cat(
            list(encoded_targets.values()), dim=1
        )
        pred_cls = torch.cat(
            list(pred_cls_dict.values()), dim=1
        )
        pred_reg = torch.cat(
            list(pred_reg_dict.values()), dim=1
        )
        pred_ctr = torch.cat(
            list(pred_ctr_dict.values()), dim=1
        )  

        ## debug:    
        # print(
        #     "max_cls_logit: {}; max_cls_prob: {}".format(
        #         pred_cls[0,...].max(),pred_cls[0,...].sigmoid().max()))

        #compute losses
        #convert class target to one-hot ecoding
        cls_targets = (encoded_targets[:,:,4]).to(torch.int64)
        # take into account that -1 label corresponds to zero-vector
        cls_targets_one_hot = torch.eye(
            self.num_classes+1)[cls_targets,:][...,:-1]
        # compute focal loss for classes
        cls_loss = sigmoid_focal_loss(
            pred_cls,cls_targets_one_hot.to(
                dtype = torch.float32, device= self.device
            )
        )

        # compute l1 loss for bounding boxes 
        reg_loss = 0.25*F.l1_loss(
            pred_reg,encoded_targets[:,:,:4],reduction='none'
        )
        reg_loss[encoded_targets[:,:,:4]<0] *= 0

        # compute BCE loss for centerness (which is between [0,1], hence the BCE)
        ctr_loss = F.binary_cross_entropy_with_logits(
            pred_ctr.squeeze(),encoded_targets[:,:,5],reduction='none'
        )
        ctr_loss[encoded_targets[:,:,5]<0] *= 0

        # compute number of feature map locations associated with a target
        num_valid_locations = (
            (encoded_targets[:,:,4] != -1).sum()/batch_size
        ).item()

        # normalize losses
        self._normalizer = 0.9 * self._normalizer + 0.1 * num_valid_locations

        return {
            "cls_loss": cls_loss.sum()/(self._normalizer * batch_size),
            "reg_loss": reg_loss.sum()/(self._normalizer * batch_size),
            "ctr_loss": ctr_loss.sum()/(self._normalizer * batch_size),
        }      

    def generate_locations(self,feature,stride):
        ''' 
        Compute image pixel coordinates of feature map:
        x_im = x_feat*stride + stride/2 
        y_im = y_feat*stride + stride/2 

        Args:
        feature: a feature map Tensor from the FPN
        stride:  stride of the feature map wrt. to the input image

        Returns:
        locations Tensor with pixel coordinates of the image
        '''
        h,w = feature.shape[2:]
        x = torch.arange(0,w*stride,stride) + stride//2
        y = torch.arange(0,h*stride,stride) + stride//2
        X,Y = torch.meshgrid(x,y)
        return torch.reshape(torch.stack([X,Y],dim = -1),[-1,2]).to(torch.float32)

    def encode_targets(
        self,
        targets: Tensor,
        locations: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        '''
        Encode the training targets on the feature map. 
        FCOS is an anchorless detector. Therefore, every location on the feature
        map(s) represents a possible anchor for detection. For encoding the 
        targets, the folowing procedure is applied on every feature map level i:
        1.  Compute distance (delta) map of every target location (x0,y0,x1,y1) 
            on the feature map (x,y): 
            t = (x - x0); l = y - y0; b = x - x1; r = y - y1
        2.  compute a maximum-distance-per-location-map for every target: 
            ltrb = max(l,t,r,b)
        3.  create mask = (ltrb < m_i) & (ltrb > m_{i-1}) for every target
        4.  compute index target map by condensing all masks to a single mask retaining
            the target with the smalles area (x1 - x0) * (y1 - y0) 
        5.  convert target coordinates (x0,y0,x1,y1) to deltas (l,t,r,b)/s 
            (s = feature level stride)
        6.  assign target properties (l,t,r,b,c) to index target map,
            set all unassigned locations to -1 (background)
        7.  compute and add centerness for each encoded target location
        Note:   This method and all methods called therein are NOT batched and 
                have to be called for individual samples!
        Args:
        targets: Tensor containing training targets, dim: (N,5)
        locations:  dict containing feature map locations (coordinates of 
                    the maps on the image), dim: (H*W,)
                    dict style: {'p3': Tensor, 'p4': Tensor, 'p5': Tensor}
        Returns: 
        encoded_targets_all_levels: dict containing encoded targets for all
                                    feature map level, dim: (H*W,6). dict style: 
                                    {'p3': Tensor, 'p4': Tensor, 'p5': Tensor}
        '''
        encoded_targets_all_levels = {
            key: None for key in locations.keys()
        }
        # debug: 
        # import matplotlib.pyplot as plt
        # plt.pcolor(x.reshape([28,28]),y.reshape([28,28]),match_matrix[:,0].reshape([28,28]))
        # iterate over feature map levels

        for i,level in enumerate(self.fpn_strides.keys()):
            x = locations[level][:,0].to(self.device)
            y = locations[level][:,1].to(self.device)
            x0 = targets[:,0]
            y0 = targets[:,1]
            x1 = targets[:,2]
            y1 = targets[:,3]
            # compute distances (delta)
            t = x[:,None] - x0[None,:]
            l = y[:,None] - y0[None,:]
            b = x1[None,:] - x[:,None]
            r = y1[None,:] - y[:,None]

            ltrb = torch.stack([l, t, r, b], axis = 2)

            # compute max(ltrb)
            match_matrix = ltrb.min(dim = 2).values > 0
            ltrb = ltrb.max(dim = 2).values

            # compute mask
            match_matrix &= (
                (ltrb < self.fpn_distances['m'+level[1:]])
                & (ltrb > self.fpn_distances['m'+str(int(level[1:])-1)])
            )
            match_matrix = match_matrix.to(torch.float32)

            # compute target areas
            target_areas = (x1 - x0) * (y1 - y0) 
            # scale mask by area
            match_matrix *=1e8 - target_areas[None,:]

            # condense mask along target dimension, retaining 
            # targets with smallest area
            match_quality,match_index = match_matrix.max(dim=1)
            # set backgound locations to -1
            match_index[match_quality < 1e5] = -1
            # assign target properties
            encoded_boxes = targets[match_index.clip(min=0)]
            # convert target coordinates to delta           
            encoded_boxes = self.coords2delta(encoded_boxes,locations[level],self.fpn_strides[level])
            encoded_boxes[match_index < 0, :] = -1
            # compute target centerness
            encoded_centerness = self.centerness_targets(encoded_boxes)
            # concatenate boxes (classes) and centerness
            encoded_target = torch.cat([encoded_boxes,encoded_centerness[...,None]],dim=1)
            # gather encoded targets of different levels
            encoded_targets_all_levels[level] = encoded_target

        return encoded_targets_all_levels

    def coords2delta(
        self,
        targets: Tensor,
        location: Tensor,
        stride: int
    ) -> Tensor:
        '''
        Convert pixel coordinates to deltas: (x0,y0,x1,y1) -> (t,l,b,r)

        Args: 
        targets: targets in pixel coordinates, dim: (H*W,4+)
        location: feature map of some level, dim: (H*W,)
        stride: stride of the provided feature map

        Returns:
        targets: targets with deltas, dim: (H*W,4+)
        '''
        x,y = location[:,0].to(self.device),location[:,1].to(self.device)
        targets[:,0] = (x - targets[:,0])/stride #t
        targets[:,1] = (y - targets[:,1])/stride #l
        targets[:,2] = (targets[:,2] - x)/stride #b
        targets[:,3] = (targets[:,3] - y)/stride #r
        return targets

    def centerness_targets(self,targets: Tensor) -> Tensor:   
        '''
        Compute centerness of targets

        Args:
        targets: targets with deltas (t,l,b,r), dim: (H*W,4+)
        Returns: 
        centerness_target: centerness, dim: (H*W,)
        ''' 
        t = targets[:,0]
        l = targets[:,1]
        b = targets[:,2]
        r = targets[:,3]
        lr = torch.stack([l, r], axis = 1)
        tb = torch.stack([t, b], axis = 1)
        centerness_target = torch.sqrt(
            (lr.min(dim = 1).values/(lr.max(dim = 1).values+1e-10))
            *(tb.min(dim = 1).values/(tb.max(dim = 1).values+1e-10))
        )
        centerness_target[targets[:,0] < 0] = -1

        return centerness_target
        

if __name__ == '__main__':

    NUM_CLASSES = 20
    BATCH_SIZE = 2
    IMAGE_SHAPE = (224, 224)

    # create some meaningless dummy data
    images = torch.rand(BATCH_SIZE, 3, 224, 224)
    target = torch.tensor(
        [[[0,0,1,1,2,0.5],[0,0,0.5,0.5,5,0.5]],
        [[0,0,1,1,2,0.5],[0,0,0.5,0.5,5,0.5]]]
    )
    fpn_strides = {
        'p3': 8,
        'p4': 16,
        'p5': 32,
    }
    fpn_features = {
        'p3': images[:,:,::8,::8],
        'p4': images[:,:,::16,::16],
        'p5': images[:,:,::32,::32],
    }
    fpn_locations = {
        'p3': images[0,:2,::8,::8].reshape(-1,2),
        'p4': images[0,:2,::16,::16].reshape(-1,2),
        'p5': images[0,:2,::32,::32].reshape(-1,2),
    } 
    dummy_bboxes = torch.rand([BATCH_SIZE,IMAGE_SHAPE[0]**2,4])
    dummy_bboxes_dict = {
        'p3': dummy_bboxes[:,::8**2,:],
        'p4': dummy_bboxes[:,::16**2,:],
        'p5': dummy_bboxes[:,::32**2,:]
    }
    dummy_centerness = torch.rand([BATCH_SIZE,IMAGE_SHAPE[0]**2])
    dummy_centerness_dict = {
        'p3': dummy_centerness[:,::8**2],
        'p4': dummy_centerness[:,::16**2],
        'p5': dummy_centerness[:,::32**2]
    }
    dummy_classes = torch.rand([BATCH_SIZE,IMAGE_SHAPE[0]**2,NUM_CLASSES])
    dummy_classes_dict = {
        'p3': dummy_classes[:,::8**2,:],
        'p4': dummy_classes[:,::16**2,:],
        'p5': dummy_classes[:,::32**2,:]
    }

    loss = FCOSLoss(fpn_strides,NUM_CLASSES)(
        dummy_bboxes_dict,dummy_centerness_dict,
        dummy_classes_dict,target,fpn_locations)
    print(loss)

    
    
