# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from data import SRDataset
from model.drln import make_model
import matplotlib.pyplot as plt

seed = 50

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

LOAD_MODEL = os.path.join("result","best.pth")

epochs = 500

save_path = 'submit'
train_img_dir = 'mytrainset'
train_img_dir = 'training_hr_images'
val_img_dir = 'testing_lr_images'
test_img_dir = 'testing_lr_images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_augmentation(dataset,min_=48,max_=48):
    r1 = random.randint(min_,max_)
    r2 = random.randint(48,48)
    if random.randint(0,9)>-1:
        dataset.angles = random.randint(0,1)*90
        dataset.lr_size = (r1,r1)
    else:
        dataset.angles = 0
        dataset.lr_size = (r1,r2)

trainset = SRDataset(train_img_dir, mode='train')
valset = SRDataset(val_img_dir, mode='val')
testset = SRDataset(test_img_dir, mode='inference')

train_loader = DataLoader(trainset, num_workers=3, batch_size=16, shuffle=True)
val_loader = DataLoader(valset, num_workers=0, batch_size=1, shuffle=False)
test_loader = DataLoader(testset, num_workers=0, batch_size=1, shuffle=False)

drln  = make_model(None).to(device)
if os.path.isfile(LOAD_MODEL):
    print("load model...",end="")
    drln.load_state_dict(torch.load(LOAD_MODEL))
    print("done")

criterion = nn.L1Loss()
# optimizer = optim.SGD(drln.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(drln.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
stepLR = optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.5)

p = 0
iter_num = 0
best_loss = 100
best_val_loss = 100
min_=4
max_=6

for epoch in range(epochs):
    counter, running_loss = 0, 0
    drln.train()
    data_augmentation(trainset)
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
        data_augmentation(trainset, min_, max_)
    stepLR.step()
    loss = running_loss/counter
    print('train, block:{}, epoch:{}, loss:{:.6}'.format(drln.stop_block, epoch+1, loss))
    p += 1
    if best_loss>=loss:
        best_loss = loss
        p = 0
    if p>12:
        if drln.stop_block<7:
            drln.stop_block += 1
            min_ += 7
            max_ += 7
        else:
            min_ = max_ = 48
        p = 0

    if drln.stop_block>6:
        drln.eval()
        counter, running_loss = 0, 0
        with torch.no_grad():
            for input_, hr_img, mask in val_loader:
                input_ = input_.to(device)
                hr_img = hr_img.to(device)
                mask = mask.to(device)
                output = drln(input_, mask)                
                loss = criterion(output, hr_img)
                running_loss += loss.item()
                counter += 1
            if best_val_loss > running_loss/counter:
                best_val_loss = running_loss/counter
                torch.save(drln.state_dict(), os.path.join("result","best.pth"))
                for input_, _, mask in test_loader:
                    input_ = input_.to(device)
                    mask = mask.to(device)
                    output = drln(input_, mask) 
                    output = trans.ToPILImage()(output[0].cpu())
                    output.save(os.path.join(save_path,_[0].split('/')[-1]))
                    # plt.imshow(output)
                    # plt.show()
        print('val, epoch:{}, loss:{:.6}, best_loss:{:.6}'.format(epoch+1, running_loss/counter, best_val_loss))
    torch.save(drln.state_dict(), os.path.join("result","block{}_latest.pth".format(drln.stop_block)))
