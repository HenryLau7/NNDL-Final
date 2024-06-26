import numpy as np
from utils import *
import matplotlib.pyplot as plt
import torch
from dataset import get_dataloaders
from model import my_deit,create_resnet18,create_resnet18_SimCLR,create_resnet34
from optimizer import optimizer
from train import trainer,test
import pytz
import os
import datetime
import itertools
from dataset import get_dataloaders
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import  CIFAR100
import argparse



def main(args):
    # print(args)
    # print(f"Pretrained={args.pretrained}, Training with lr={args.learning_rate}, decay={args.weight_decay}")
    
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')    
    
    # time
    china_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.datetime.now(china_tz)
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
        
    log_dir = args.logpath + '{}_{}_{}_lr_{}_decay_{}_pretrained_{}_dropout_{}_cutout_{}_cutmix_{}_{}/'.format(args.date, args.model, args.opt ,args.learning_rate, args.weight_decay, args.pretrained, args.dropout,args.cutout, args.cutmix, args.comment)
    writer = SummaryWriter(log_dir + "log")    

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    
    train_loader, test_loader = get_dataloaders(root_dir=args.datapath,batch_size=args.batch_size, cutout=args.cutout)
    
    # model = create_resnet34(num_classes=args.num_classes,pretrained=args.pretrained,dropout=args.dropout)
    # model.cuda()
    
    # Resnet18
    # 
    # model = create_pit(num_classes=args.num_classes, pretrained= True)
    # model = PiTTiny32(num_classes=100)
    if args.model == 'ResNet18':
        model = create_resnet18(num_classes=args.num_classes,pretrained=args.pretrained,dropout=args.dropout)
    elif args.model == 'ResNet18+SimCLR':
        simclr = torch.load('./logs/checkpoint_0500.pth.tar')['state_dict']
        simclr_new = {}
        for k, v in simclr.items():
            if k.startswith("backbone.") and not k.startswith('backbone.fc'):
                new_key = k[len("backbone."):]
                simclr_new[new_key] = v
            else:
                simclr_new[k] = v
        model = create_resnet18_SimCLR(simclr_new,num_classes=args.num_classes,dropout= args.dropout)
        # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert len(parameters) == 2  # fc.weight, fc.bias
        
    elif args.model == 'ResNet34':
        model = create_resnet34(num_classes=args.num_classes,pretrained=args.pretrained,dropout=args.dropout)
    elif args.model == 'deit':
        model = my_deit(num_classes=args.num_classes, pretrained=args.pretrained)
    
    model.to(device)

    
    # resnet18+simclr
    # simclr = torch.load('../checkpoint_0100.pth.tar')['state_dict']
    # simclr_new = {}
    # for k, v in simclr.items():
    #     if k.startswith("backbone."):
    #         new_key = k[len("backbone."):]
    #         simclr_new[new_key] = v
    #     else:
    #         simclr_new[k] = v

    # model = create_resnet18_SimCLR(simclr_new,num_classes=args.num_classes,dropout= args.dropout)
    
    
    opt = optimizer(model,opt= args.opt, lr=args.learning_rate, weight_decay=args.weight_decay)   
    # schedular = CosineAnnealingLR(opt,T_max=100,eta_min=0)
    schedular = StepLR(opt,step_size=30,gamma=0.2)
    # schedular = MultiStepLR(opt, milestones=[20, 40, 60, 80], gamma=0.2)
    # schedular = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)

    
    Trainer = trainer(model, opt, schedular, args.epoch, train_loader, test_loader, args.cutmix, log_dir, writer, device)
    Trainer.train()
    
    if args.plot:
        plot_metrics(Trainer, log_dir)
    
    checkpoint = torch.load(log_dir + 'best_model.pth')
    model.load_state_dict(checkpoint['parameters'])
    test(model,test_loader,device)
    
    return 0


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datapath', default= '../data/')
    # parser.add_argument('--logpath', default='./logs/')
    # parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--epoch', type=int, default=1000)
    # parser.add_argument('--learning_rate', type=float, default=3e-4)
    # parser.add_argument('--seed', type=int, default=43)
    # parser.add_argument('--date',default='0618')
    # parser.add_argument('--plot',default=True)
    # parser.add_argument('--reg',type=float, default=3e-4)
    # parser.add_argument('--decay',type=float, default=0.9)
    # parser.add_argument('--pretrained',type=float, default=0.9)

    
    # learning_rate = [3e-3,3e-4]
    # decay= [3e-3,3e-4]

    # pretrain = [True, False]
    # for pretrained, lr, wd in itertools.product(pretrain,learning_rate, decay):
        
    # pretrained = True
    # lr = 0.0003
    # wd = 0.0003

    class args:
        datapath = '../data/'
        logpath = './logs/'
        # batch_size = 512
        batch_size = 512
        epoch = 200
        seed = 43
        date = '0626'
        plot = True
        num_classes = 100
        learning_rate = 3e-3
        # opt = 'SGD'
        # opt = 'SGD'
        opt = 'AdamW'
        # learning_rate = 0.1
        # learning_rate = 3e-3
        # learning_rate = 0.1
        # weight_decay = 0.05
        weight_decay = 1e-4
        pretrained = True
        device_id = 0
        dropout = True
        cutout = True
        cutmix = False
        model = 'ResNet18+SimCLR'
        # model = 'ResNet18'
        # model = 'ResNet34'
        # model = 'deit'
        # setting = 'simclr'
        comment = 'StepLR_30_0.2'
        # comment = 'CosLR'
        # comment = 'MulLR'
        
        @classmethod
        def to_dict(cls):
            return {attr: getattr(cls, attr) for attr in cls.__dict__ if not callable(getattr(cls, attr)) and not attr.startswith("__")}
        
        @classmethod
        def print_dict(cls):
            import json
            print(json.dumps(cls.to_dict(), indent=4))

    # 打印类属性作为字典
    args.print_dict()
    main(args)

