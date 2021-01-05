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

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

LOAD_MODEL = os.path.join("result","best.pth")

epochs = 1000000

train_img_dir = 'mytrainset'
train_img_dir = 'training_hr_images'
val_img_dir = 'testing_lr_images'
test_img_dir = 'testing_lr_images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_augmentation(dataset):
    r1 = random.randint(64,96)
    r2 = random.randint(64,96)
    if random.randint(0,9)>5:
        dataset.angles = random.randint(0,1)*90
        dataset.lr_size = (r1,r1)
    else:
        dataset.angles = 0
        dataset.lr_size = (r1,r2)

trainset = SRDataset(train_img_dir, mode='train')
valset = SRDataset(val_img_dir, mode='val')
testset = SRDataset(test_img_dir, mode='inference')

train_loader = DataLoader(trainset, num_workers=0, batch_size=2, shuffle=True)
val_loader = DataLoader(valset, num_workers=0, batch_size=1, shuffle=False)
test_loader = DataLoader(testset, num_workers=0, batch_size=1, shuffle=False)

drln  = make_model(None).to(device)
if os.path.isfile(LOAD_MODEL):
    print("load model...",end="")
    drln.load_state_dict(torch.load(LOAD_MODEL))
    print("done")

criterion = nn.L1Loss()
# optimizer = optim.SGD(drln.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(drln.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
stepLR = optim.lr_scheduler.StepLR(optimizer, 1e01, gamma=0.5)


iter_num = 0
best_loss = 0.6

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
        stepLR.step()
        data_augmentation(trainset)
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
            drln.eval()
            for input_, _, mask in test_loader:
                with torch.no_grad():
                    input_ = input_.to(device)
                    mask = mask.to(device)
                    output = drln(input_, mask) 
                    output = trans.ToPILImage()(output[0].cpu())
                    plt.imshow(output)
                    plt.show()
        print('val {} epoch: {}, best: {}'.format(epoch+1, running_loss/counter, best_loss))
    
    
    


