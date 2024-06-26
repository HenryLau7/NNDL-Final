import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100



def get_dataloaders(root_dir, batch_size=32, cutout=True):
    
    # if model == 'deit':
    if cutout:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Resize((96, 96)),
            # transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
            transforms.RandomRotation(15),            # 随机旋转±20度
            transforms.ToTensor(),                  # 转换为张量
            Cutout(n_holes=5, length=32),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 归一化
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),                  # 转换为张量
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 归一化
        ])

        
    else:
        # train_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),  # 随机裁剪
        #     transforms.RandomHorizontalFlip(),      # 随机水平翻转
        #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
        #     transforms.RandomRotation(20),            # 随机旋转±20度
        #     transforms.ToTensor(),                  # 转换为张量
        #     Cutout(n_holes=1, length=16),
        #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 归一化
        # ])
        
        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),                  # 转换为张量
        #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 归一化
        # ])
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
            transforms.RandomRotation(15),            # 随机旋转±20度
            transforms.ToTensor(),                  # 转换为张量
            # Cutout(n_holes=5, length=32),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 归一化
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),                  # 转换为张量
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 归一化
        ])


    train_dataset = CIFAR100(root_dir, train=True, transform=train_transform)
    test_dataset = CIFAR100(root_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


# dataset = CUB200Dataset(root_dir='./CUB_200_2011/',train=True,transform=None)
# print(dataset.__len__)


class Cutout(object):
    """ 随机遮挡图像中的一部分 """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 需要被处理的图像。
        Returns:
            PIL Image: 被遮挡后的图像。
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = np.clip(y - self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            y2 = np.clip(y + self.length // 2, 0, h)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
