from pathlib import Path
from PIL import Image
import cv2
import dataset.transforms as T

import torch
import torch.utils.data
import torchvision

class CASIA_Dataset(object):
    def __init__(self, img_folder, transforms):
        self._transforms = transforms
        with open(img_folder,'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pathes = self.data[idx]
        img_path, mask_path, edge_path, label = pathes.split(' ')
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        edge = cv2.imread(edge_path,0)

        target = {'mask': mask, 'edge': edge}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img,target):
        img = img.resize((self.size,self.size), self.interpolation)
        target['mask'] = target['mask'].resize((self.size,self.size), self.interpolation)
        target['edge'] = target['edge'].resize((self.size/4,self.size/4), self.interpolation)
        img = self.toTensor(img)
        target['mask'] = self.toTensor(target['mask'])
        target['edge'] = self.toTensor(target['edge'])
        img.sub_(0.5).div_(0.5)
        return img,target


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided CASIA path {root} does not exist'
    # modify to the real path
    PATHS = {
        "train": root / "...",
        "val": root / "...",
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CASIA_Dataset(img_folder, transforms=resizeNormalize(size=512))

    return dataset