# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from PIL import Image
from data import SRDataset
from model.drln import make_model
import matplotlib.pyplot as plt

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

LOAD_MODEL = os.path.join("result","best.pth")
#LOAD_MODEL = "DRLN_BDX3.pt"

epochs = 1000000

train_img_dir = 'mytrainset'
train_img_dir = 'training_hr_images'
val_img_dir = 'testing_lr_images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_augmentation(dataset):
    r1 = random.randint(96,128)
    r2 = random.randint(96,128)
    if random.randint(0,9)>8:
        dataset.angles = random.randint(0,1)*90
        dataset.lr_size = (r1,r1)
    else:
        dataset.angles = 0
        dataset.lr_size = (r1,r2)

trainset = SRDataset(train_img_dir)
valset = SRDataset(val_img_dir, trainset=False)

train_loader = DataLoader(trainset, num_workers=0, batch_size=1, shuffle=True)
val_loader = DataLoader(valset, num_workers=0, batch_size=1, shuffle=False)

drln  = make_model(None).to(device)
if os.path.isfile(LOAD_MODEL):
    print("load model...",end="")
    drln.load_state_dict(torch.load(LOAD_MODEL))
    print("done")

criterion = nn.L1Loss()
# optimizer = optim.SGD(drln.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(drln.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
stepLR = optim.lr_scheduler.StepLR(optimizer, 3e02, gamma=0.5)


iter_num = 0
best_loss = 0.05782179960182735


for epoch in range(epochs):
    counter, running_loss = 0, 0
    drln.train()
    for input_, hr_img, mask in train_loader:
        input_ = input_.to(device)
        hr_img = hr_img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = drln(input_, mask)
        loss = criterion(output, hr_img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        counter += 1
        stepLR.step()
        data_augmentation(trainset)
        # out = trans.ToPILImage()(output[0].cpu())
        # plt.imshow(out)
        # plt.show()
        # out = trans.ToPILImage()(hr_img[0].cpu())
        # plt.imshow(out)
        # plt.show()
        # out = trans.ToPILImage()((mask[0]*255).cpu())
        # plt.imshow(out)
        # plt.show()
    print('train {} epoch:'.format(epoch+1), running_loss/counter)
    if epoch%1==0:
        drln.eval()
        counter, running_loss = 0, 0
        for input_, hr_img, mask in val_loader:
            with torch.no_grad():
                input_ = input_.to(device)
                hr_img = hr_img.to(device)
                mask = mask.to(device)
                output = drln(input_, mask)                
                loss = criterion(output, hr_img)
                running_loss += loss.item()
                counter += 1
        if best_loss > running_loss/counter:
            best_loss = running_loss/counter
            torch.save(drln.state_dict(), os.path.join("result","best.pth"))
            # inp = trans.ToPILImage()(input_[0].cpu())
            out = trans.ToPILImage()(output[0].cpu())
            # plt.imshow(inp)
            # plt.show()
            plt.imshow(out)
            plt.show()
        print('val {} epoch: {}, best: {}'.format(epoch+1, running_loss/counter, best_loss))
        '''
        inp = trans.ToPILImage()(input_[0].cpu())
        out = trans.ToPILImage()(output[0].cpu())
        out2 = trans.ToPILImage()(hr_img[0].cpu())
        plt.imshow(inp)
        plt.show()
        plt.imshow(out)
        plt.show()
        plt.imshow(out2)
        plt.show()
        
        print('{} epoch:'.format(epoch+1), running_loss/counter)
        input_ = trans.ToTensor()(img).to(device).unsqueeze(0)
        out = drln(input_).squeeze(0).cpu()
        pil = trans.ToPILImage()(out)
        plt.imshow(img)
        plt.show()
        plt.imshow(pil)
        plt.show()
        # pil.convert("RGB").save("09_hr.bmp")
        torch.save(drln.state_dict(), os.path.join("result","latest.pth"))
        # input()
        '''
    
    
    


