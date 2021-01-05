# -*- coding: utf-8 -*-
import os
import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms as trans
import torchvision.transforms.functional as TF
from PIL import Image

k = 5

class MyRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return TF.rotate(x, self.angles)

    
class MyHSVTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        trans_list = [TF.adjust_brightness, TF.adjust_contrast, TF.adjust_gamma, TF.adjust_saturation]
        random.shuffle(trans_list)
        for my_trans in trans_list:
            x = my_trans(x, random.uniform(0.5,1.5))
        return x

def input_transform(hr_size, pad, angles, h_p, v_p):
    return trans.Compose(
        [
            MyHSVTransform(),
            trans.Pad(pad),
            trans.RandomCrop(hr_size),
            MyRotationTransform(angles),
            trans.RandomHorizontalFlip(p=h_p),
            trans.RandomVerticalFlip(p=v_p)
        ]
    )

def mask_transform(angles, h_p, v_p):
    return trans.Compose(
        [
            trans.ToPILImage(),
            MyRotationTransform(angles),
            trans.RandomHorizontalFlip(p=h_p),
            trans.RandomVerticalFlip(p=v_p),
            trans.ToTensor()
        ]
    )    


class SRDataset(data.Dataset):
    def __init__(self, img_dir, lr_size=(64,64), scale=3, angles=0, mode='train', show=False):
        super(SRDataset, self).__init__()
        path_pattern = os.path.join(img_dir, "*.*")
        self.mode = mode
        self.show = show
        self.filename_list = glob.glob(path_pattern, recursive=True)
        self.input_transform = trans.RandomCrop
        if mode=='train':
            self.input_transform = input_transform
        self.lr_size = lr_size
        self.norm_trans = trans.ToTensor()
        self.scale = scale
        self.angles = angles

    def __getitem__(self, index):
        input_file = self.filename_list[index]
        img = Image.open(input_file)
        width, height = img.size
        self.hr_size = (self.lr_size[0]*self.scale, self.lr_size[1]*self.scale)
        if self.mode=='inference':
            patch_lr = img
            patch_hr = 0
            mask = torch.ones((3,height*3,width*3))
        elif self.mode=='val':
            patch_hr = self.input_transform((height//3*3,width//3*3))(img)
            patch_lr = trans.Resize((height//3,width//3), interpolation=Image.NEAREST)(patch_hr)
            patch_hr = self.norm_trans(patch_hr)
            mask = torch.ones(patch_hr.shape)
        elif self.mode=='train':
            pad = [0, 0, 0, 0]
            if self.hr_size[0] > height:
                pad[3] = self.hr_size[0]-height
            if self.hr_size[1] > width:
                pad[2] = self.hr_size[1]-width
            pad = tuple(pad)
            h_p, v_p = random.randint(0,1), random.randint(0,1)
            patch_hr = self.input_transform(self.hr_size, pad, self.angles, h_p, v_p)(img)
            patch_lr = trans.Resize(self.lr_size, interpolation=Image.NEAREST)(patch_hr)
            patch_hr = self.norm_trans(patch_hr)
            mask = torch.ones(patch_hr.shape)
            if pad[3] > 0:
                mask[:,height:,:] = torch.zeros((3, patch_hr.shape[1]-height, patch_hr.shape[2]))
            if pad[2] > 0:
                mask[:,:,width:] = torch.zeros((3, patch_hr.shape[1], patch_hr.shape[2]-width))
            mask = mask_transform(self.angles, h_p, v_p)(mask)
        if self.show:
            return patch_lr, patch_hr
        return self.norm_trans(patch_lr), patch_hr, mask
    
    def __len__(self):
        return len(self.filename_list)


def test():
    trainset = SRDataset('training_hr_images',(200,200))
    trainset.show = True
    x = trainset[k]
    x[0].show()
    trans.ToPILImage()(x[1]).show()

if __name__ == "__main__":
    test()