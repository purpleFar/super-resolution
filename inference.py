# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:35:06 2021

@author: Lin
"""

# -*- coding: utf-8 -*-
import os
import torch
import torchvision.transforms as trans
from PIL import Image
from model.drln import make_model
import matplotlib.pyplot as plt

LOAD_MODEL = "DRLN_BDX3.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

drln  = make_model(None).to(device)
if os.path.isfile(LOAD_MODEL):
    print("load model...",end="")
    drln.load_state_dict(torch.load(LOAD_MODEL))
    print("done")


img = Image.open('09.bmp')#.crop((0,0,48,48))
img.save("09_lr.png")
drln.eval()
input_ = trans.ToTensor()(img).to(device).view(1,3,26,26)
out = drln(input_).squeeze(0).cpu()
plt.imshow(trans.ToPILImage()(out))
pil = trans.ToPILImage()(out)
pil.convert("RGB").save("09_hr.png")
    