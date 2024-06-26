import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np


def relu(x):
    return np.maximum(x,0)

def softmax(x):    
    exp_x = np.exp(x - np.max(x,axis=1,keepdims=True))  # 从x中减去最大值以提高数值稳定性
    return exp_x / np.sum(exp_x, axis=1,keepdims=True)  # 对列求和，适用于二维数组的情况    
    
def cross_entropy_loss(y_true, y_pred):
    # 避免对数0
    return -np.sum(y_true * np.log(y_pred + 1e-12))

def get_one_hot_label(y, classes):
    one_hot = np.zeros((y.size, classes))
    one_hot[np.arange(y.size), y] = 1

    return one_hot    

def plot_metrics(trainer, log_dir):
    train_epochs = range(1, len(trainer.train_losses) + 1)
    val_epochs = range(trainer.epochs_per_vali, trainer.epochs_per_vali * len(trainer.val_losses) + 1, trainer.epochs_per_vali)
    
    plt.figure(figsize=(10, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, trainer.train_losses, label='Training Loss',lw=2.0)
    plt.plot(val_epochs, trainer.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=18)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, trainer.train_accuracies, label='Training Accuracy')
    plt.plot(val_epochs, trainer.val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(log_dir + 'plot.png',dpi=300)
    
    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    
    targets = (targets, shuffled_targets, lam)
    
    return data, targets

# def cutmix_criterion(preds, targets):
#     targets1, targets2, lam = targets
#     return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)
