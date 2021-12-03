import os
from posixpath import split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.utils import save_image

class saliency_dataset(Dataset):
    def __init__(self, img_dir, gt_dir, img_format='png', gt_format='png'):
        names2 = []
        names1 = []

        names1 = ['.'.join(name.split('.')[:-1]) for root, directories, filenames in os.walk(img_dir) for name in filenames if name.endswith(img_format) and '_mask' not in name ]
        names2 = ['.'.join(name.split('.')[:-1]) for root, directories, filenames in os.walk(img_dir) for name in filenames if name.endswith(img_format) and '_mask' in name and 'mask_' not in  name]
        names1, names2 = sorted(names1), sorted(names2)
        

        self.img_filenames = [os.path.join(img_dir, name.split(' ')[0] + '/' + name + '.' + img_format) for name in names1]
        self.gt_filenames = [os.path.join(gt_dir, name.split(' ')[0] + '/' + name + '.' + gt_format) for name in names2]
        transform_img = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        transform_gt = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        self.transform_img = transforms.Compose(transform_img)
        self.transform_gt = transforms.Compose(transform_gt)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        img_gt = Image.open(self.gt_filenames[index]).convert('RGB')
        img_gt = transforms.Resize((256, 256))(img_gt)
        img_gt = self.transform_gt(img_gt)

        img = Image.open(self.img_filenames[index]).convert('RGB')
        img = transforms.Resize((256, 256 ))(img)
        img = self.transform_img(img)

        save_image(img_gt, 'img_gt.jpg')
        save_image(img, 'img.jpg')

        return img, img_gt
        



def get_dataloader_(opt):
    images_path = os.path.join(os.path.join(opt.base_dir, opt.dataset_name))
    s_dataset = saliency_dataset(images_path, images_path)
    train_size = int(0.8 * len(s_dataset))
    test_size = len(s_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(s_dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=2, drop_last=False)
    dataloader_test = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=2, drop_last=False)
    return dataloader_train, dataloader_test

