from __future__ import annotations
from typing import Tuple,Any,Optional,Callable
import os
import torch
from  torchvision.datasets import VOCDetection
from xml.etree.ElementTree import parse as ET_parse
from PIL import Image
from model.inference import inverse_image_transform, show_image
from datasets.transforms import *

class MyVOCDetection(VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    SUBCLASSED to perform fixed transformations and return image path
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        image_size (int): transformed image size
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(
        self, 
        root: str, 
        year: str = "2012", 
        image_set: str = "train", 
        download: bool = False, 
        image_size: int = 224,
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
        transforms: Optional[Callable] = None):
        super().__init__(
            root, year, image_set, download, 
            transform, target_transform, transforms
        )

        self.image_size = image_size
        VOC_CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(VOC_CLASSES)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(VOC_CLASSES)
        }
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # call transformation method to also return image path    
        path, img, target = self.transformations(img, target)

        return path, img, target
    
    def transformations(self,image,target):

        _transforms = [
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                to_bgr255=False
            ),
            PadToSquare(0),
            Flip(0.5,'horizontal'),
            Resize(224)
        ]
        image_transform = Compose(_transforms)

        image_path = os.path.join(
            target['annotation']['folder'],target['annotation']['filename']
        )   
        targ_class = torch.tensor(
            [int(self.class_to_idx[targ['name']]) 
            for targ in target['annotation']['object']]
        )
        target_box = torch.tensor(
            [[float(coord) for coord in box['bndbox'].values()]
            for box in target['annotation']['object']]
        )

        # Transform input image to CHW tensor and normalize.
        image,target_box = image_transform(image,target_box)
    
        # Concatenate GT classes with GT boxes; shape: (N, 5)
        target = torch.cat([target_box, targ_class[...,None]], dim=1)

        # Pad to 40 boxes
        target = torch.cat(
            [target, torch.zeros(40 - len(target), 5).fill_(-1.0)]
        )
        return image_path,image, target

if __name__ == '__main__':

    batch_size = 64
    num_classes = 20
    file_path = os.path.join(os.sep,'Volumes','Storage','Datasets','voc')
    train_ds = MyVOCDetection(
        file_path,image_set = 'train', download = False,
    )

    for i in range(4):
        path,image,targets = train_ds[i]

        show_image(
            inverse_image_transform(image),
            targets = targets,
            idx_to_class = train_ds.idx_to_class
        )

