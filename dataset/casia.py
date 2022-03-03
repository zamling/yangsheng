from pathlib import Path
from PIL import Image
import cv2

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
        label = int(label)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        edge = cv2.imread(edge_path,0)

        if self._transforms is not None:
            img, mask, edge = self._transforms(img, mask, edge)

        return img, mask, edge, label



class resizeNormalize(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img,mask,edge):
        img = cv2.resize(img,(self.size,self.size),self.interpolation)
        mask = cv2.resize(mask,(self.size,self.size),self.interpolation)
        edge = cv2.resize(edge,(int(self.size/4),int(self.size/4)),self.interpolation)
        img = self.toTensor(img)
        mask = torch.as_tensor(mask).unsqueeze(0)
        edge = torch.as_tensor(edge).unsqueeze(0)

        img.sub_(0.5).div_(0.5)
        return img,mask,edge


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided CASIA path {root} does not exist'
    # modify to the real path
    PATHS = {
        "train": root / "mydata.txt",
        "val": root / "mydata.txt",
    }

    img_folder = PATHS[image_set]
    dataset = CASIA_Dataset(img_folder, transforms=resizeNormalize(size=512))

    return dataset