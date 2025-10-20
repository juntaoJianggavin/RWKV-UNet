import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import transform

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-30, 30)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_brightness_contrast(image, label):
    brightness_factor = np.random.uniform(-0.2, 0.2)
    image = image * (1 + brightness_factor)
    contrast_factor = np.random.uniform(0.8, 1.2)
    image = (image - image.mean()) * contrast_factor + image.mean()
    image = np.clip(image, 0, 1) if image.max() <= 1 else np.clip(image, 0, 255)
    return image, label


def random_crop(image, label):
    x, y = image.shape
    crop_ratio = np.random.uniform(0.8, 1.2)
    crop_size = (int(x * crop_ratio), int(y * crop_ratio))
    if crop_size[0] >= x:
        crop_size = (x, crop_size[1])
    if crop_size[1] >= y:
        crop_size = (crop_size[0], y)
    x_start = np.random.randint(0, x - crop_size[0] + 1)
    y_start = np.random.randint(0, y - crop_size[1] + 1)
    image_crop = image[x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
    label_crop = label[x_start:x_start+crop_size[0], y_start:y_start+crop_size[1]]
    image = transform.resize(image_crop, (x, y), order=0, preserve_range=True)
    label = transform.resize(label_crop, (x, y), order=0, preserve_range=True)
    
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Utilize it if the pretrained weights are not used
       # if random.random() > 0.5:
            #image, label = random_rot_flip(image, label)
        #if random.random() > 0.8:
            #image, label = random_rotate(image, label)

        if random.random() > 0.9:
            image, label = random_brightness_contrast(image, label)
        
        if random.random() > 0.9:
            image, label = random_crop(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        assert (image.shape[0] == self.output_size[0]) and (image.shape[1] == self.output_size[1])
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
