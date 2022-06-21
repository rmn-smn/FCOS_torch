import torch
import numpy as np
from torchvision.transforms import functional as F_tv
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class ToTensor():
    def __call__(self, image, target):
        return F_tv.to_tensor(image), target


class Normalize():
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F_tv.normalize(image, mean=self.mean, std=self.std)
        return image, target

class PadToSquare():
    def __init__(self,pad_value):
        self.pad_value = pad_value
    def __call__(self, img, target):
        # from: https://github.com/eriklindernoren/PyTorch-YOLOv3
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=self.pad_value)
        
        target[:, 0] += pad[0]
        target[:, 1] += pad[2]
        target[:, 2] += pad[1]
        target[:, 3] += pad[3]

        return img, target

class Resize():
    def __init__(self,size):
        self.size = size
    def __call__(self, img, target):
        c, h_old, w_old = img.shape
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        c, h_new, w_new = img.shape
        target[:,[0,2]] /= (w_old/w_new)
        target[:,[1,3]] /= (h_old/h_new)
        return img,target

class Flip():
    def __init__(self,p,direction='horizontal'):
        self.p = p  
        self.direction = direction  
    def __call__(self,img, target):
        if np.random.random() < self.p:
            c, h, w = img.shape
            if self.direction == 'horizontal':
                img = torch.flip(img, [-1])
                y_max = h - target[:, 0]
                y_min = h - target[:, 2]
                target[:, 0] = y_min
                target[:, 2] = y_max
            if self.direction == 'vertical':
                img = torch.flip(img, [-2])
                x_max = w - target[:, 1]
                x_min = w - target[:, 3]
                target[:, 1] = x_min
                target[:, 3] = x_max        
        return img, target