import torch
import torch.nn as nn
from torchvision import models
import timm
from safetensors.torch import load_file
from pit import pit_ti
from deit import deit_small_patch16_224

class PiTTiny32(nn.Module):
    def __init__(self, num_classes=100):
        super(PiTTiny32, self).__init__()
        # self.model = timm.create_model('pit_ti_224', pretrained=False,checkpoint_path='./model.safetensors')
        self.model = pit_ti(num_classes=100,pretrained=False)
        keys_to_ignore = ['pos_embed', 'patch_embed.conv.weight', 'head.weight','head.bias']
        state_dict = load_file('./model.safetensors')
        
        new_state_dict = {}
        for key, value in state_dict.items():
            # Map the checkpoint keys to the model keys
            if "transformers.1.pool" in key:
                new_key = key.replace("transformers.1.pool", "pools.0")
            elif "transformers.2.pool" in key:
                new_key = key.replace("transformers.2.pool", "pools.1")
            else:
                new_key = key
            new_state_dict[new_key] = value
        # import pdb;pdb.set_trace()
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k not in keys_to_ignore}
        
        self.model.load_state_dict(filtered_state_dict, strict = False)        

        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
    def forward(self, x):
        # import pdb;pdb.set_trace()
        return self.model(x)
    

    
class my_deit(nn.Module):
    def __init__(self, num_classes=100, pretrained = True):
        super(my_deit, self).__init__()
        self.model = deit_small_patch16_224(pretrained=False)
        if pretrained:
            checkpoint = torch.load('./logs/deit_small_patch16_224-cd65a155.pth')
            self.model.load_state_dict(checkpoint['model'])
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        

# class resnet18_simclr(nn.Module):
#     def __init__(self, num_classes=100, simclr_model=None, dropout=True):
#         super(resnet18_simclr, self).__init__()
#         self.backbone = models.resnet18(weights = False, num_classes=128)
#         self.dim_mlp = self.backbone.fc.in_features
#         self.backbone.fc = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp), nn.ReLU(), self.backbone.fc)
#     # import pdb;pdb.set_trace()

#         self.backbone.load_state_dict(simclr_model)
#         if dropout:
#             self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5), 
#             nn.Linear(128, num_classes)
#         )
#         else:
#             self.classifier = nn.Linear(128, num_classes)  # 新的线性层
        
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
        
#         return x
    
def create_resnet18_SimCLR(simclr_model, num_classes, dropout):
    model = models.resnet18(weights = None, num_classes=num_classes)
    # dim_mlp = backbone.fc.in_features
    # backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), backbone.fc)
    # import pdb;pdb.set_trace()

    
    log =  model.load_state_dict(simclr_model,strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    num_features = model.fc.in_features  
    if dropout:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 添加 Dropout
            nn.Linear(num_features, num_classes)
        )
    else:
        model.fc = nn.Linear(num_features, num_classes) 
        
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False
    
    return model
    


def create_resnet34(num_classes=200,pretrained=True,dropout=True):
    model = models.resnet18(weights = pretrained, num_classes=num_classes)

    num_features = model.fc.in_features  
    
    if dropout:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 添加 Dropout
            nn.Linear(num_features, num_classes)
        )
    else:
        model.fc = nn.Linear(num_features, num_classes) 

    return model


def create_resnet18(num_classes=200, pretrained=True, dropout=True):
    model = models.resnet18(weights = pretrained)

    num_features = model.fc.in_features  
    
    if dropout:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 添加 Dropout
            nn.Linear(num_features, num_classes)
        )
    else:
        model.fc = nn.Linear(num_features, num_classes) 

    return model



# vit = PiTTiny32(num_classes=100)
# resnet18 = create_resnet18(num_classes=100)
# resnet34 = create_resnet34(num_classes=100,pretrained=True)
# print(f'PiT-Tiny的参数量: {sum(p.numel() for p in vit.parameters()) / 1e6:.2f}M')
# print(f'resnet18的参数量: {sum(p.numel() for p in resnet18.parameters()) / 1e6:.2f}M')
# print(f'resnet34的参数量: {sum(p.numel() for p in resnet34.parameters()) / 1e6:.2f}M')
# print(f'deit_small的参数量: {sum(p.numel() for p in deit_small.parameters()) / 1e6:.2f}M')

simclr = torch.load('./logs/checkpoint_0500.pth.tar')['state_dict']
simclr_new = {}
for k, v in simclr.items():
    if k.startswith("backbone."):
        new_key = k[len("backbone."):]
        simclr_new[new_key] = v
    else:
        simclr_new[k] = v
            
            
# model = create_resnet18_SimCLR(simclr_new,num_classes=100,dropout= 100)
# import pdb;pdb.set_trace()