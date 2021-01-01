# -*- coding: utf-8 -*-
import os
import glob
import random
import torch.utils.data as data
import torchvision.transforms as trans
import torchvision.transforms.functional as TF
from PIL import Image

k = 5

class MyRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def input_transform(hr_size):
    return trans.Compose(
        [
            trans.RandomCrop(hr_size),
            MyRotationTransform([0,90]),
            trans.RandomHorizontalFlip(p=0.5),
            trans.RandomVerticalFlip(p=0.5)
        ]
    )

class SRTrainDataset(data.Dataset):
    def __init__(self, img_dir, hr_size=(78,78), scale=3, input_transform=input_transform, show=False):
        super(SRTrainDataset, self).__init__()
        path_pattern = os.path.join(img_dir, "*.*")
        self.show = show
        self.filename_list = glob.glob(path_pattern, recursive=True)
        self.input_transform = input_transform
        self.hr_size = hr_size
        self.lr_size = (hr_size[0]//scale, hr_size[1]//scale)
        self.norm_trans = trans.ToTensor()

    def __getitem__(self, index):
        input_file = self.filename_list[index]
        img = Image.open(input_file)
        patch_hr = self.input_transform(self.hr_size)(img)
        patch_lr = trans.Resize(self.lr_size)(patch_hr)
        if self.show:
            return patch_lr, patch_hr
        return self.norm_trans(patch_lr), self.norm_trans(patch_hr)
    
    def __len__(self):
        return len(self.filename_list)


def test():
    trainset = SRTrainDataset('training_hr_images',(200,200))
    trainset.show = True
    x = trainset[k]
    x[0].show()
    x[1].show()

if __name__ == "__main__":
    test()