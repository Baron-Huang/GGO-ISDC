import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)
# 获取项目根目录（假设你的脚本在项目根目录的子文件夹中）
# 例如，如果脚本在 project/src/utils/ 下，那么项目根目录是 project/
project_root = os.path.dirname(current_dir)  # 如果脚本在 src/ 下
# project_root = os.path.dirname(os.path.dirname(current_dir))  # 如果脚本在 src/utils/ 下

# 将项目根目录插入到 sys.path 的最前面，优先级最高
sys.path.insert(0, project_root)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Main_func_SOTAs.APIs.nystrom_attention import NystromAttention
from Read_Feats_Datasets import Read_Feats_Datasets
from Training_Testing_for_SOTA.training_testing_for_transmil import training_for_transmil, testing_for_transmil
from torch.utils.data import DataLoader
from Utils.Setup_Seed import setup_seed

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.4
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_size, n_classes, mDim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=mDim)
        self._fc1 = nn.Sequential(nn.Linear(input_size, mDim), nn.ReLU(), nn.Dropout(0.25))
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=mDim)
        self.layer2 = TransLayer(dim=mDim)
        self.norm = nn.LayerNorm(mDim)
        self._fc2 = nn.Linear(mDim, self.n_classes)


    def forward(self, x):
        h = x #[B, n, 1024]

        #h = self._fc1(h) #[B, n, 512]
        h = self._fc1(h.squeeze(0)).unsqueeze(0)
        #h = self.sc_layer(h.squeeze(0)).unsqueeze(0)

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]


        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]

        return logits, h

if __name__ == "__main__":
    random_seed = 1
    batch_size = 1
    num_classes = 2   # CAMELYON16: 2, CAMELYON16: 3, CAMELYON16: 3, CAMELYON16: 2
    epoch = 100
    gpu_device = 1
    mode_stats = 'train'  # train or test
    weight_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/TransMIL_CAMELYON16.pth'
    testing_weights_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/TransMIL_CAMELYON16.pth'
    data_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/CAMELYON16/CAMELYON_pretrained_without_PE_public/'
    roc_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Predictions/CAMELYON16/TransMIL'
    resu_text_path =\
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/TransMIL_train_log.txt'

    layers_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/CAMELYON16/TransMIL'
    feats_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/CAMELYON16/TransMIL'


    setup_seed(random_seed)
    train_dataset = Read_Feats_Datasets(
        data_path=data_read_path + r'/Train/Feats',
        label_path=data_read_path + r'/Train/Labels')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    test_dataset = Read_Feats_Datasets(
        data_path=data_read_path + r'/Test/Feats',
        label_path=data_read_path + r'/Test/Labels')
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    val_dataset = Read_Feats_Datasets(
        data_path=data_read_path + r'/Test/Feats',
        label_path=data_read_path + r'/Test/Labels')
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


    transmil_net = TransMIL(768, n_classes=num_classes)
    #test_x = torch.randn((85, 768))
    #pre_y, _, _ = abmil_net(test_x)

    #print(abmil_net)

    #print(pre_y.shape)

    transmil_net = transmil_net.cuda(gpu_device)

    #### lr_fn: vit_amu or vit_amu
    if mode_stats == 'train':
        training_for_transmil(mil_net=transmil_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                           proba_mode=False, lr_fn='vit_amu', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2, current_lr=None,
                           data_parallel=False, weight_path=weight_path, proba_value=0.85, class_num = num_classes,
                           bags_stat = True, resu_text_path = resu_text_path)

        transmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        transmil_net.load_state_dict(transmil_weight, strict=True)
        testing_for_transmil(test_model=transmil_net, train_loader=train_loader, val_loader=val_loader,
                             proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                             out_mode=None, proba_mode=False, class_num=num_classes, roc_save_path=roc_save_path,
                             bags_stat=True, bag_relations_path=None, resu_text_path=resu_text_path)

    elif mode_stats == 'test':
        transmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        transmil_net.load_state_dict(transmil_weight, strict=True)

        ### get layers vlaues
        from Results_codes.get_layers import get_layers

        get_layers(layer=transmil_net._fc2, save_path=layers_save_path)

        ### get features
        from Results_codes.get_features import Get_Features

        get_feats = Get_Features(layer=transmil_net._fc1, end_no=-5, save_path=feats_save_path,
                                 out_or_in='in', with_pe = False)
        get_feats.regis_layer()

        new_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        testing_for_transmil(test_model = transmil_net, train_loader=new_train_loader, val_loader=val_loader,
                           proba_value = None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode = None, proba_mode=False, class_num=num_classes, roc_save_path = roc_save_path,
                           bags_stat=True, bag_relations_path = None, resu_text_path = resu_text_path)
        ### get features
        get_feats.get_feats_grads()

    else:
        assert print('mode state error!!!')
