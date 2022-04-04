from network import *
import torch
import torch.nn as nn
from collections import OrderedDict
from torch import optim
import os
import torchvision
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from network import *
from attention import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self,hp):
        super(Model, self).__init__()
        self.hp = hp
        self.net = network()

        self.net.apply(weights_init_normal).to(device)

        self.sample_train_params = self.parameters()
        self.loss = nn.L1Loss()
        self.optimizer = optim.Adam(self.sample_train_params, hp.learning_rate)
        self.hp = hp
        self.step = 0

    def forward(self, img, img_gt):
        self.optimizer.zero_grad()

        output = self.net(img)

        loss = self.loss(output, img_gt)

        loss.backward()
        self.optimizer.step()
        
        return loss

    def append_dir(self, name):
        if not os.path.exists(self.hp.saved_models):
            os.mkdir(self.hp.saved_models)
        pathappend = self.hp.saved_models + "/" + name
        if not os.path.exists(pathappend):
            os.makedirs(pathappend)
        return pathappend

    def test(self, dataloader_Test, epoch):
        pathvrst = self.append_dir("val_output")
        self.optimizer.zero_grad()
        self.eval()
        img_stack = []
        for i, data in enumerate(dataloader_Test):
            img, img_gt= data
            img = img.to(device)
            with torch.no_grad():
                msk_big = self.net(img)
            img_stack.append(img[0])
            img_stack.append(img_gt[0])
            img_stack.append(msk_big[0])

            img_stack = torch.stack(img_stack, dim=0)
            save_image(img_stack, f"{pathvrst}/{i}+{str(epoch)+'epochs'}.png")

            img_stack = []

            # save_image(img_gt, f"{pathvrst}/{str(i) + 'gt'}+{str(epoch)+'epochs'}.png")
            # save_image(img, f"{pathvrst}/{str(i) + 'org'}+{str(epoch)+'epochs'}.png")
            
        
        
        
            
