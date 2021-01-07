# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:26:08 2020

@author: Lin
"""

import os
import torch
import torchvision.transforms as trans
from PIL import Image
from model.drln import make_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOAD_MODEL = os.path.join("result","best.pth")
input_path = "testing_lr_images"
save_path = 'submit'
if not os.path.isdir(save_path):
  os.mkdir(save_path)
  
drln  = make_model(None).to(device)
drln.stop_block = 7
drln.eval()
if os.path.isfile(LOAD_MODEL):
    print("load model...",end="")
    drln.load_state_dict(torch.load(LOAD_MODEL))
    print("done")

with torch.no_grad():
    for dirPath, dirNames, fileNames in os.walk(input_path):
        for f in fileNames:
            img_file = os.path.join(dirPath, f)
            img = Image.open(img_file)
            width, height = img.size
            img = trans.ToTensor()(img).to(device).unsqueeze(0)
            mask = torch.ones((3,height*3,width*3)).to(device).unsqueeze(0)
            output = drln(img, mask)
            im = trans.ToPILImage()(output[0].cpu())
            im.save(os.path.join(save_path,f))
