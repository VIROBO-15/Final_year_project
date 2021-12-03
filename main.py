import sys
import os
sys.path.append(os.getcwd())

import argparse
from model import *
#from Saliency_dataset import get_dataloader_
import time
import torch
from dataset import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#from tensorplot import Visualizer
# 1 e-1, 1e-2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Saliency')

    parser.add_argument('--dataset_name', type=str, default='sketchy')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--backbone_name', type=str,
                        default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    #parser.add_argument('--root_dir', type=str, default=os.getcwd())
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=70)
    parser.add_argument('--eval_freq_iter', type=int, default=20)
    parser.add_argument('--print_freq_iter', type=int, default=20)
    parser.add_argument('--aux_lambda', type=float, default=1.)
    parser.add_argument('--draw_frequency', type=int, default=10000)
    parser.add_argument('--saved_models', type=str,
                        default=os.path.join(os.getcwd(), 'models'))
    parser.add_argument('--wr', default=5e-3, type=float)
    


    hp = parser.parse_args()
    dataloader_Train_sal, dataloader_Test_sal = get_dataloader_(hp)
    print(hp)
    print(device)
    # if hp.debug == False and not torch.cuda.is_available():
    #     sys.exit("GPU NOT DETECTED, RE RUN THIS CODE")

    os.makedirs(hp.saved_models, exist_ok=True)

    # model.load_state_dict(torch.load('VGG_ShoeV2_model_best.pth', map_location=device))
    model = Saliency_Model(hp)
    model.to(device)
    step_count, best_accuracy, maxfm, mae = -1, 0, 0, 0

    for i_epoch in range(hp.max_epoch):
        for i_batch, batch_data in enumerate(dataloader_Train_sal):
            img, img_gt, WWs, HHs, names = batch_data
            img, img_gt = img.to(device), img_gt.to(device)
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model(img, img_gt)

            if (step_count + 0) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, Loss:{}'.format(i_epoch, i_batch, step_count, loss))
            if (step_count + 1) % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    model.test(dataloader_Test_sal, step_count)
                    torch.save(model.state_dict(), os.path.join(hp.saved_models, '{}_model_{}_iter.pth'.format(hp.dataset_name, step_count)))
    #save the model
    
