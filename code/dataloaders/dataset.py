import os
import cv2
import torch
import random
import math
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
import gridmask
import pdb
import torch.nn as nn
from functools import reduce 
# from augmentations.ctaugment import test_np, test_np_2

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        # ops_strong_cutout=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        # self.ops_strong_cutout =  ops_strong_cutout

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong) # , self.ops_strong_cutout
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
    
class BaseDataSets_glas(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        # ops_strong_cutout=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.to_tensor = transforms.ToTensor()
        # self.ops_strong_cutout =  ops_strong_cutout

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image = Image.open(self._base_dir + "/train/images/" + "{}.bmp".format(case))
            label = Image.open(self._base_dir + "/train/masks/" + "{}.bmp".format(case))
        else:
            image = Image.open(self._base_dir + "/test/images/" + "{}.bmp".format(case))
            label = Image.open(self._base_dir + "/test/masks/" + "{}.bmp".format(case))
        image = self.to_tensor(image)
        label = self.to_tensor(label)
        # print(torch.max(label))
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong) # , self.ops_strong_cutout
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class BaseDataSets_isic2016(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        # ops_strong_cutout=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.to_tensor = transforms.ToTensor()
        # self.ops_strong_cutout =  ops_strong_cutout

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image = Image.open(self._base_dir + "/train/images/" + "{}.jpg".format(case))
            label = Image.open(self._base_dir + "/train/masks/" + "{}_Segmentation.png".format(case))
        else:
            image = Image.open(self._base_dir + "/test/images/" + "{}.jpg".format(case))
            label = Image.open(self._base_dir + "/test/masks/" + "{}_Segmentation.png".format(case))
        image = self.to_tensor(image)
        label = self.to_tensor(label)
        # print(torch.max(label))
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong) # , self.ops_strong_cutout
            else:
                sample = self.transform(sample)
                
        sample["idx"] = idx
        return sample   
        # else:

        #     image, label = image.cpu().detach().numpy(), label.squeeze(1).cpu().detach().numpy()

        #     x, y = image.shape[2], image.shape[3]
        #     image = zoom(image, (1, 1, 224 / x, 224 / y), order=0)
        #     image = torch.from_numpy(image).float().cuda()
        #     sample = {"image": image, "label": label}
        #     sample["idx"] = idx
        #     return sample
    
# def random_rot_flip(image, label=None):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     if label is not None:
#         label = np.rot90(label, k)
#         label = np.flip(label, axis=axis).copy()
#         return image, label
#     else:
#         return image
def random_rot_flip(image, label=None):  
    k = np.random.randint(0, 4)
    # image = np.rot90(image, k)
    image = np.rot90(image, k, axes=(1, 2))
    # axis = np.random.randint(0, 2)
    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k, axes=(1, 2))
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def rand_affine(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    affine = transforms.RandomAffine(degrees = 90,translate=(0.5,0.5),shear=30)
    return affine(image)

def gaussian_blur(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    blur = transforms.GaussianBlur(3)
    return blur(image)

# def rand_gray(image):
#     if not torch.is_tensor(image):
#         np_to_tensor = transforms.ToTensor()
#         image = np_to_tensor(image)

#     gray = transforms.RandomGrayscale(p=0.2)
#     return gray(image)

def rand_gray(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    if image.size(0) == 1: # Check if the image has only 1 channel
        image = torch.cat((image, image, image), 0) # Convert to 3-channel image by duplicating the channel

    gray = transforms.RandomGrayscale(p=0.2)
    return gray(image)


def grid_mask(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    mask = gridmask.Gridmask()
    return mask(image)

# def test_np(image_weak,ops_strong,strong_index,img_height,img_width):
#     image_strong = Image.new(image_weak.mode, image_weak.size)  
#     image_strong.paste(image_weak) 
#     # image_strong_no_cutout = Image.new(image_weak.mode, image_weak.size)  
#     # image_strong_no_cutout.paste(image_weak) 
#     # image_strong = augmentations.cta_apply(image_weak, ops_strong)

#     strong_index = [True if "cutout"==item[0] else False for item in ops_strong]
#     # height_loc_all = []
#     # width_loc_all = []
#     for (op, args), index in zip(ops_strong, strong_index):
#         if index == True:
#             height_loc = np.random.randint(low=img_height // 2, high=img_height)
#             width_loc = np.random.randint(low=img_height // 2, high=img_width)  
#     #         # height_loc = 0
#     #         # width_loc = 0 
#     #         # height_loc_all.append(height_loc)
#     #         # width_loc_all.append(width_loc)
#             image_strong = OPS[op].f(image_strong, height_loc, width_loc, *args) # height_loc, width_loc, 
#         else:
#             image_strong = OPS[op].f(image_strong, *args)
#     return image_strong

# def test_np_2(label_aug,ops_strong,strong_index,img_height,img_width):
#     label_aug_strong = Image.new(label_aug.mode, label_aug.size)  
#     label_aug_strong.paste(label_aug) 
#     for (op, args), index in zip(ops_strong, strong_index):
#         if index == True:
#             height_loc = np.random.randint(low=img_height // 2, high=img_height)
#             width_loc = np.random.randint(low=img_height // 2, high=img_width)
#             label_aug_strong = OPS[op].f(label_aug_strong, height_loc, width_loc, *args) # height_loc, width_loc, 
#         else:
#             label_aug_strong = OPS[op].f(label_aug_strong, *args)
#     return label_aug_strong

class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong): # , ops_strong_cutout
        image, label = sample["image"], sample["label"]
        # image = image.permute(1, 2, 0).contiguous()
        # image = self.resize_glas_image(image)  
        # label = self.resize_glas_image(label)
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # # fix dimensions
        # image = torch.from_numpy(image.astype(np.float32))
        # label = torch.from_numpy(label.astype(np.uint8)).squeeze(0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)

        # img_height, img_width = image_weak.size

        # image_strong = Image.new(image_weak.mode, image_weak.size)  
        # image_strong.paste(image_weak) 
        # # # image_strong_no_cutout = Image.new(image_weak.mode, image_weak.size)  
        # # # image_strong_no_cutout.paste(image_weak) 

        # strong_index = [True if "cutout"==item[0] else False for item in ops_strong]
        # if True not in strong_index:
        #     image_strong = augmentations.cta_apply(image_weak, ops_strong)
        # else:
        #     image_strong,_,_ = augmentations.cta_apply(image_weak, ops_strong)

        
        # # image_strong = test_np(image_weak,ops_strong,strong_index,img_height,img_width)

        # # height_loc_all = []
        # # width_loc_all = []
        # for (op, args), index in zip(ops_strong, strong_index):
        #     if index == True:
        #         # height_loc = np.random.randint(low=img_height // 2, high=img_height)
        #         # width_loc = np.random.randint(low=img_height // 2, high=img_width) 
        # #         # height_loc = 0
        # #         # width_loc = 0 
        # #         # height_loc_all.append(height_loc)
        # #         # width_loc_all.append(width_loc)
        #         image_strong = OPS[op].f(image_strong, *args) # height_loc, width_loc, 
        #     else:
        #         image_strong = OPS[op].f(image_strong, *args)
                # image_strong_no_cutout = OPS[op].f(image_strong_no_cutout, *args)
                # try:
                #     image_strong = OPS[op].f(image_strong, *args)
                # except:
                #     print("error!!!!")          
                #       

        # image_strong_no_cutout = augmentations.cta_apply(image_weak, ops_strong_cutout)
        # image_strong_cutout = Image.new(image_weak.mode, image_weak.size)  
        # image_strong_cutout.paste(image_weak) 
        # strong_index = [True if "cutout"==item[0] else False for item in ops_strong_cutout]
        # for (op, args), index in zip(ops_strong_cutout, strong_index):
        #     if index == True:
        #         height_loc = np.random.randint(low=img_height // 2, high=img_height)
        #         width_loc = np.random.randint(low=img_height // 2, high=img_width)  
        #         image_strong_cutout, _, _ = OPS[op].f(image_strong_cutout, height_loc, width_loc, *args)
        #     else:
        #         image_strong_cutout = OPS[op].f(image_strong_cutout, *args)

        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        # label_aug_strong = test_np_2(label_aug,ops_strong,strong_index,img_height,img_width)
        # label_aug_strong = augmentations.cta_apply(transforms.ToPILImage()(label), ops_strong)  
        # height_loc = np.random.randint(low=img_height // 2, high=img_height)
        # width_loc = np.random.randint(low=img_height // 2, high=img_width)
        # print(height_loc)
        # print(width_loc)
        
        # label_aug_strong = Image.new(label_aug.mode, label_aug.size)  
        # label_aug_strong.paste(label_aug) 
        # # label_aug_strong = transforms.ToPILImage()(label)
        # strong_cutout = [item for item in ops_strong if "cutout"==item[0]]
        # if len(strong_cutout) != 0:
        #     upper_coord_all = []
        #     lower_coord_all = []
        #     label_aug_strong = Image.new(label_aug.mode, label_aug.size)  
        #     label_aug_strong.paste(label_aug) 
        # for (op, args), height_loc, width_loc in zip(strong_cutout, height_loc_all, width_loc_all):
        # for (op, args), index in zip(ops_strong, strong_index):
        #     if index == True:
        #         height_loc = np.random.randint(low=img_height // 2, high=img_height)
        #         width_loc = np.random.randint(low=img_height // 2, high=img_width)
                # label_aug_strong,_,_ = OPS[op].f(label_aug_strong, *args) # height_loc, width_loc, 
            # else:
            #     label_aug_strong = OPS[op].f(label_aug_strong, *args)
            # upper_coord_all.append(upper_coord)
            # lower_coord_all.append(lower_coord)

        # height_loc = np.random.randint(low=img_height // 2, high=img_height)
        # width_loc = np.random.randint(low=img_height // 2, high=img_width)

        #     masks=[]
        #     for upper_coord, lower_coord in zip(upper_coord_all, lower_coord_all):
        #         mask = torch.ones(size=[img_height, img_width])
        #         for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        #             for j in range(upper_coord[1], lower_coord[1]):  # For every row
        #                 mask[i,j] = 0
        #         masks.append(mask)
        #     mask_final = reduce(lambda x, y: x * y, masks) 
        #     # label_aug_strong = augmentations.cta_apply(label_aug, strong_cutout)
        # else:
        #     mask_final = torch.ones(size=[img_height, img_width])
        #     label_aug_strong = label_aug

        label_aug = to_tensor(label_aug).squeeze(0)
        # label_aug_strong = to_tensor(label_aug_strong).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()
        # label_aug_strong = torch.round(255 * label_aug_strong).int()

        try:
            image_strong = to_tensor(image_strong)
        except:
            print("error")

        sample = {
            "image": image,
            "label": label,
            "image_weak": to_tensor(image_weak),
            
            "image_strong": image_strong,

            # "image_strong_no_cutout": to_tensor(image_strong_no_cutout),
            # "image_strong_cutout": to_tensor(image_strong_cutout),
            "label_aug": label_aug,
            # "label_aug_strong": label_aug_strong,
            # "mask_final": mask_final,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    
    def resize_glas_image(self, image):
        _, x, y = image.shape
        return zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)



    
class RandomGenerator_w(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob
 
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)
 
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
 
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))
 
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
 
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)
 
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
 
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
 
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
 
        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask
 
        mask = mask.expand_as(img)
        img = img * mask
 
        return img
 
class Grid_Mask(nn.Module):
    def __init__(self, d1=20, d2=80, rotate=90, ratio=0.4, mode=1, prob=0.8):
        super(Grid_Mask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)
 
    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)
 
    
    def forward(self, x):
        if not self.training:
            return x
        
        return self.grid(x)

class RandomGenerator_s(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = color_jitter(image).type("torch.FloatTensor")
        image = rand_affine(image).type("torch.FloatTensor")
        image = gaussian_blur(image).type("torch.FloatTensor")
        image = rand_gray(image).type("torch.FloatTensor")
#         grid_mask = Grid_Mask()
#         image = grid_mask(image).type("torch.FloatTensor")
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {"image": image, "label": label}
        return sample
    
    
# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample["image"], sample["label"]
#         # ind = random.randrange(0, img.shape[0])
#         # image = img[ind, ...]
#         # label = lab[ind, ...]
#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         x, y = image.shape
#         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.uint8))
#         sample = {"image": image, "label": label}
#         return sample

class RandomGenerator(object):  
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        _, x, y = image.shape
        image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).squeeze(0)
        sample = {"image": image, "label": label}
        return sample
