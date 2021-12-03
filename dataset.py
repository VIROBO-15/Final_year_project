import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.utils import save_image

class saliency_dataset(Dataset):
    def __init__(self, img_dir, gt_dir, img_format='jpg', gt_format='png'):
        names1 = ['.'.join(name.split('.')[:-1]) for name in os.listdir(gt_dir)]
        names2 = ['.'.join(name.split('.')[:-1]) for name in os.listdir(img_dir)]
        names = list(set(names1) & set(names2))
        self.img_filenames = [os.path.join(img_dir, name + '.' + img_format) for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name + '.' + gt_format) for name in names]
        transform_list =[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.onehot = np.zeros(201)
        self.names = names

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        name = self.names[index]
        img_gt = Image.open(self.gt_filenames[index]).convert('RGB')
        img_gt = transforms.Resize((256, 256))(img_gt)
        img_gt = self.transform(img_gt)
        WW, HH = img_gt.size(1), img_gt.size(2)

        img = Image.open(self.img_filenames[index]).convert('RGB')
        img = transforms.Resize((WW, HH ))(img)
        img = self.transform(img)

        # save_image(img_gt, 'img_gt.jpg')
        # save_image(img, 'img.jpg')

        return img, img_gt, WW, HH, name



def get_dataloader_(opt):
    images_path = os.path.join(os.path.join(opt.base_dir, 'Dataset/ECSSD'), 'images')
    gt_path = os.path.join(os.path.join(opt.base_dir, 'Dataset/ECSSD'), 'ground_truth_mask')
    s_dataset = saliency_dataset(images_path, gt_path)
    train_size = int(0.98 * len(s_dataset))
    test_size = len(s_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(s_dataset, [train_size, test_size])
    #print("Helllooooooooooooooo")

    dataloader_train = DataLoader(s_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=8, drop_last=False)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    return dataloader_train, dataloader_test

