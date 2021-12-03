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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Saliency_Model(nn.Module):
    def __init__(self,hp):
        super(Saliency_Model, self).__init__()
        self.hp = hp
        self.net = Saliceny_network()

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
        img = []
        for i, data in enumerate(dataloader_Test):
            img, img_gt, WWs, HHs, names = data
            img = img.to(device)
            with torch.no_grad():
                msk_big = self.net(img)
            save_image(msk_big, f"{pathvrst}/{i}+{str(epoch)+'epochs'}.png")
            save_image(img_gt, f"{pathvrst}/{str(i) + 'gt'}+{str(epoch)+'epochs'}.png")
            save_image(img, f"{pathvrst}/{str(i) + 'org'}+{str(epoch)+'epochs'}.png")
            
            # msk_big = msk_big.squeeze(1)
            # msk_big = msk_big.cpu().numpy()
            # for b, _msk in enumerate(msk_big):
            #     name = names[b]
            #     WW = WWs[b]
            #     HH = HHs[b]
            #     _msk = Image.fromarray((_msk*255).astype(np.uint8))
            #     _msk = _msk.resize((WW, HH))
            #     _msk.save(f"{pathvrst}/{name}.png")

        
        
        
            
