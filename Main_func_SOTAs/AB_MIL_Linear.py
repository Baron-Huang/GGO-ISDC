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
from Main_func_SOTAs.APIs.nystrom_attention import NystromAttention
from Utils.Setup_Seed import setup_seed
from Read_Feats_Datasets import Read_Feats_Datasets
from Training_Testing_for_SOTA.training_testing_for_abmil import training_for_abmil, testing_for_abmil
from torch.utils.data import DataLoader


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



class AttentionMIL(nn.Module):
    def __init__(self, in_features, num_classes=2, L=768, D=192, n_leision = 5, attn_mode="gated", dropout_node=0.0):
        super().__init__()
        self.L = L
        self.D = D
        self.K = 1

        self.attn_mode = attn_mode


        self.MLP = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.ReLU(),
        )

        if attn_mode == 'gated':
            self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Sigmoid()
            )

            self.attention_weights = nn.Linear(self.D, self.K)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_node) if dropout_node>0.0 else nn.Identity(),
            nn.Linear(self.L*self.K, num_classes)
        )

    def forward(self, x):
        H = self.MLP(x)  # NxL

        if self.attn_mode == 'gated':
            A_V = self.attention_V(H)  # NxD
            A_U = self.attention_U(H)  # NxD
            A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
        else:
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        logits = self.classifier(M)

        return logits, A ,H


if __name__ == "__main__":
    random_seed = 1
    batch_size = 1
    num_classes = 2  # CAMELYON16: 2, CAMELYON16: 3
    epoch = 100
    gpu_device = 0
    mode_stats = 'train'  # train or test
    weight_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/AB_MIL_L_CAMELYON16.pth'
    testing_weights_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/AB_MIL_L_CAMELYON16.pth'
    roc_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Predictions/CAMELYON16/AB_MIL_L'
    resu_text_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/AB_MIL_Linear_train_log.txt'
    layers_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/CAMELYON16/AB_MIL_L'
    feats_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/CAMELYON16/AB_MIL_L'

    data_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/CAMELYON16/CAMELYON_pretrained_without_PE_public/'


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

    abmill_net = AttentionMIL(768, attn_mode='linear', dropout_node=0.1, num_classes=num_classes).cuda(gpu_device)
    #test_x = torch.randn((85, 768))

    if mode_stats == 'train':
        training_for_abmil(mil_net=abmill_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                           proba_mode=False, lr_fn='vit_amu', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2,
                           current_lr=None, data_parallel=False, weight_path=weight_path, proba_value=0.85,
                           class_num = num_classes, bags_stat = True, resu_text_path = resu_text_path)

        abmill_weight = torch.load(testing_weights_path, map_location='cuda:0')
        abmill_net.load_state_dict(abmill_weight, strict=True)
        testing_for_abmil(test_model=abmill_net, train_loader=train_loader, val_loader=val_loader,
                          proba_value=None, test_loader=test_loader, gpu_device=gpu_device, out_mode=None,
                          proba_mode=False, class_num=num_classes, roc_save_path=roc_save_path,
                          bags_stat=True, bag_relations_path=None, resu_text_path = resu_text_path)

    elif mode_stats == 'test':
        abmill_weight = torch.load(testing_weights_path, map_location='cuda:0')
        abmill_net.load_state_dict(abmill_weight, strict=True)

        ### get layers vlaues
        from Results_codes.get_layers import get_layers

        get_layers(layer=abmill_net.classifier[1], save_path=layers_save_path)

        ### get features
        from Results_codes.get_features import Get_Features

        get_feats = Get_Features(layer=abmill_net.MLP[1], end_no=961, save_path=feats_save_path, with_pe=False)
        get_feats.regis_layer()

        new_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        testing_for_abmil(test_model=abmill_net, train_loader=new_train_loader, val_loader=val_loader,
                          proba_value=None, test_loader=test_loader, gpu_device=gpu_device, out_mode=None,
                          proba_mode=False, class_num=num_classes, roc_save_path=roc_save_path,
                          bags_stat=True, bag_relations_path=None, resu_text_path=resu_text_path)

        ### get features
        get_feats.get_feats_grads()

    else:
        assert print('mode state error!!!')







































