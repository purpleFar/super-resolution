# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from data import SRTrainDataset
from model.drln import make_model

LOAD_MODEL = os.path.join("result","load.pth")

epochs = 1000000

img_dir = 'training_hr_images'
#img_dir = 'test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = SRTrainDataset(img_dir)

train_loader = DataLoader(trainset, num_workers=16, batch_size=16, shuffle=True)

drln  = make_model(None).to(device)
if os.path.isfile(LOAD_MODEL):
    print("load model...",end="")
    drln.load_state_dict(torch.load(LOAD_MODEL))
    print("done")

criterion  = nn.L1Loss()

# optimizer = optim.SGD(drln.parameters(), lr=0.00001, momentum=0.9)
optimizer = optim.Adam(drln.parameters(), lr=0.00008, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
stepLR = optim.lr_scheduler.StepLR(optimizer, 2e5, gamma=0.5)


img = Image.open('09.bmp') # .crop((11,82,37,108))

iter_num = 0

for epoch in range(epochs):
    counter, running_loss = 0, 0
    drln.train()
    for input_, hr_img in train_loader:
        input_ = input_.to(device)
        hr_img = hr_img.to(device)
        optimizer.zero_grad()
        output = drln(input_)
        loss = criterion(output, hr_img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        counter += 1
        stepLR.step()
    if epoch%20==0:
        print('{} epoch:'.format(epoch+1), running_loss/counter)
        drln.eval()
        input_ = trans.ToTensor()(img).to(device).view(1,3,26,26)
        out = drln(input_).view(3,78,78).cpu()
        pil = trans.ToPILImage()(out)
        pil.convert("RGB").save("09_hr.bmp")
        torch.save(drln.state_dict(), os.path.join("result","latest.pth"))
    
    
    


