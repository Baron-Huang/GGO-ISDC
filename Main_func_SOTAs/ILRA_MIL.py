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
import math
from Utils.Setup_Seed import setup_seed
from torch.utils.data import DataLoader
from Read_Feats_Datasets import Read_Feats_Datasets
from Training_Testing_for_SOTA.training_testing_for_ilramil import training_for_ilramil, testing_for_ilramil

"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, _ = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat

import torch.nn as nn
import torch.distributed as dist


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret = self.mha(global_embedding, X)
        return ret


class ILRA(nn.Module):
    def __init__(self, num_layers=3, feat_dim=384, n_classes=2, hidden_feat=128, num_heads=4, topk=1, ln=False):
        super().__init__()
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(num_layers):
            block = GAB(dim_in=feat_dim if idx == 0 else hidden_feat,
                        dim_out=hidden_feat,
                        num_heads=num_heads,
                        num_inds=topk,
                        ln=ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=hidden_feat, num_heads=num_heads, num_seeds=topk, ln=ln)

        # classifier
        self.classifier = nn.Linear(in_features=hidden_feat, out_features=n_classes)

        initialize_weights(self)
        print(f"ilra2~")

    def forward(self, x):
        for block in self.gab_blocks:
            x = block(x)

        feat = self.pooling(x)
        logits = self.classifier(feat)

        logits = logits.squeeze(1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat


if __name__ == "__main__":
    random_seed = 1
    batch_size = 1
    num_classes = 2 # CAMELYON16: 2, CAMELYON16: 3, CAMELYON16: 3, CAMELYON16: 3
    epoch = 100
    gpu_device = 1
    mode_stats = 'train'  # train or test
    weight_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/ILRA_MIL_CAMELYON16.pth'
    testing_weights_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/ILRA_MIL_CAMELYON16.pth'
    data_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/CAMELYON16/CAMELYON_pretrained_without_PE_public/'
    roc_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Predictions/CAMELYON16/ILRA_MIL'
    resu_text_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/ILRA_MIL_train_log.txt'

    layers_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/CAMELYON16/ILRA_MIL'
    feats_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/CAMELYON16/ILRA_MIL'

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



    irramil_net = ILRA(feat_dim=768, n_classes=num_classes)
    #test_x = torch.randn((85, 768))
    #pre_y, _, _ = abmil_net(test_x)

    #print(abmil_net)

    #print(pre_y.shape)

    irramil_net = irramil_net.cuda(gpu_device)

    if mode_stats == 'train':
        training_for_ilramil(mil_net=irramil_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                           proba_mode=False, lr_fn='vit_amu', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2, current_lr=None,
                           data_parallel=False, weight_path=weight_path, proba_value=0.85, class_num = num_classes,
                           bags_stat = True, resu_text_path = resu_text_path)

        irramil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        irramil_net.load_state_dict(irramil_weight, strict=True)
        testing_for_ilramil(test_model=irramil_net, train_loader=train_loader, val_loader=val_loader,
                            proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                            out_mode=None, proba_mode=False, class_num=num_classes,
                            roc_save_path=roc_save_path, bags_stat=True, bag_relations_path=None,
                            resu_text_path=resu_text_path)

    elif mode_stats == 'test':
        irramil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        irramil_net.load_state_dict(irramil_weight, strict=True)

        ### get layers vlaues
        from Results_codes.get_layers import get_layers

        get_layers(layer=irramil_net.classifier, save_path=layers_save_path)

        ### get features
        from Results_codes.get_features import Get_Features

        get_feats = Get_Features(layer=irramil_net.pooling, end_no=-5, save_path=feats_save_path,
                                 out_or_in='in', with_pe=False)
        get_feats.regis_layer()

        new_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


        testing_for_ilramil(test_model = irramil_net, train_loader=new_train_loader, val_loader=val_loader,
                           proba_value = None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode = None, proba_mode=False, class_num=num_classes,
                           roc_save_path = roc_save_path, bags_stat=True, bag_relations_path = None,
                           resu_text_path = resu_text_path)

        ### get features
        get_feats.get_feats_grads()

    else:
        assert print('Error!!!')
