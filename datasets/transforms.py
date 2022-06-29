import torch
import numpy as np
from torchvision.transforms import functional as F_tv
import torch.nn.functional as F
import random
import cv2

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
        self.size=size
    def __call__(self, img, target):
        c, h_old, w_old = img.shape
        img = F_tv.resize(img,size=self.size)
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
                y_max = w - target[:, 0]
                y_min = w - target[:, 2]
                target[:, 0] = y_min
                target[:, 2] = y_max
            if self.direction == 'vertical':
                img = torch.flip(img, [-2])
                x_max = h - target[:, 1]
                x_min = h - target[:, 3]
                target[:, 1] = x_min
                target[:, 3] = x_max        
        return img, target

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None):
        if random.randint(0,2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None):
        if random.randint(0,2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None):
        if random.randint(0,2):
            swap = self.perms[random.randint(0,len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None):
        if random.randint(0,2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes


class RandomBrightness(object):
    def __init__(self, delta=32/255):
        assert delta >= 0.0
        assert delta <= 1#255.0
        self.delta = delta

    def __call__(self, image, boxes=None):
        if random.randint(0,2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class ToCV2Image(object):
    def __call__(self, tensor, boxes=None):
        return tensor.numpy().astype(np.float32).transpose((1, 2, 0)), boxes
        
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes):
        im = image #.copy()
        im, boxes = self.rand_brightness(im, boxes)
        if random.randint(0,2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes = distort(im, boxes)
        return self.rand_light_noise(im, boxes)