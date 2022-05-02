"""
CREATE DATASETS
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

import os
import cv2
import torch
import random
import imageio
import os.path

import numpy as np
import torch.utils.data as data

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from imutils import paths
from random import shuffle

from pycocotools.coco import COCO
from torchvision.datasets import DatasetFolder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision.datasets.coco import CocoDetection

# pylint: disable=E1101

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# custom dataset class
class CustomDataset(data.Dataset):
    def __init__(self, images, labels= None, transforms = None):
        self.labels = labels
        self.images = images
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = self.images[index][:]
        
        if self.transforms:
            data = self.transforms(data)
        
        if self.labels is not None:
            return (data, self.labels[index])
        else:
            return data

def load_Caltech101(root,train_transform,val_transform):
    image_paths = list(paths.list_images(root))

    data = []
    labels = []
    for img_path in tqdm(image_paths):
        label = img_path.split(os.path.sep)[-2]
        if label == "BACKGROUND_Google":
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        data.append(img)
        labels.append(label)
        
    data = np.array(data)
    labels = np.array(labels)

    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    print(f"Total Number of Classes: {len(lb.classes_)}")

    # divide the data into train, validation, and test set
    (X, x_val , Y, y_val) = train_test_split(data, labels, test_size=0.2,  stratify=labels,random_state=42)
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)
    print(f"x_train examples: {x_train.shape}\nx_test examples: {x_test.shape}\nx_val examples: {x_val.shape}")
    
    train_data = CustomDataset(x_train, y_train, train_transform)
    val_data = CustomDataset(x_val, y_val, val_transform)
    test_data = CustomDataset(x_test, y_test, val_transform)


    return train_data, val_data, test_data

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nz=100, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': target}

    def __setitem__(self, index, value):
        self.noise[index] = value

    def __len__(self):
        return len(self.imgs)


class CocoClsDataset(data.Dataset):
    def __init__(self, root_dir, ann_file, img_dir, phase, transform, less_sample=False):
        self.ann_file = os.path.join(root_dir, ann_file)
        self.img_dir = os.path.join(root_dir, img_dir)
        self.transforms = transform
        self.coco = COCO(self.ann_file)


        cat_ids = self.coco.getCatIds()
        categories = self.coco.dataset['categories']
        self.id2cat = dict()
        for category in categories:
            self.id2cat[category['id']] = category['name']
        self.id2label = {category['id'] : label for label, category in enumerate(categories)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.labels = []
        self.new_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            if len(target) > 0:
                x, y, w, h = target[0]['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                if target[0]['area'] <= 0 or w < 1 or h < 1 or target[0]['iscrowd']:
                    continue
                self.new_ids.append(img_id)
                cat_id = target[0]['category_id']
                self.labels.append(self.id2label[cat_id])

        print('total_length of dataset:', len(self))


    def __len__(self):
        return len(self.new_ids)


    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.new_ids[idx]
        label = self.labels[idx]
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.img_dir,path)).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
  

        return img, label

