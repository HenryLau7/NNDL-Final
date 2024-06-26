import torch.optim as optim


def optimizer(model, opt, lr, weight_decay):
    # pretrained_params = [param for name, param in model.named_parameters() if 'fc' not in name]
    # new_params = model.fc.parameters()

    # optimizer = optim.Adam([
    #     {'params': pretrained_params, 'lr': lr*0.1,  'weight_decay': weight_decay*0.1 },  # lr and weight_decay are both 0.1 times
    #     {'params': new_params, 'lr': lr, 'weight_decay': weight_decay}          
    # ])
    if opt == 'Adam':
        optimizer = optim.Adam(params=model.parameters(),lr=lr,weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(params=model.parameters(),lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif opt == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(),lr=lr, eps=1e-8, weight_decay=0.05)
    
    return optimizer