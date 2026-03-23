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
import numpy as np
from Main_func_SOTAs.APIs.multiheadatt import MultiheadLinearAttention
from Main_func_SOTAs.APIs.linearatt import MultiheadLinearAttention
from Read_Feats_Datasets import Read_Feats_Datasets
from Utils.Setup_Seed import setup_seed
from torch.utils.data import DataLoader
from Training_Testing_for_SOTA.training_testing_for_dgrmil import training_for_dgrmil, testing_for_dgrmil

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout= d
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class CrossLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.attn = MultiheadLinearAttention(embed_dim=dim,num_heads=8,dropout=d)

    def forward(self,q,k,v):

        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        x,attention= self.attn(q,k,v)

        return x.permute(1,0,2), attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout1d(drop)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.act2(x)
        return x

class optimizer_triple(nn.Module):
    def __init__(self, in_feature,out_feature,drop=0.):
        super().__init__()
        self.infeature = in_feature
        self.outfeature = out_feature
        self.drop_rate = drop
        #print(self.infeature)
        self.fc1 = nn.Linear(self.infeature,self.outfeature)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(self.drop_rate)

        self.fc2 = nn.Linear(self.outfeature,self.outfeature)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(self.drop_rate)

    def forward(self, x, mode):
        if mode == 'global':
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = self.act2(x)

        else:
            x = self.fc1(x)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.drop2(x)

        return x

class DGRMIL(nn.Module):
    def __init__(self, in_features, num_classes=2, L=512, D=128, n_lesion = 11,
                 attn_mode="gated", dropout_node=0.0,dropout_patch=0.0,initialize=False):
        super().__init__()
        self.L = L
        self.D = D
        self.n_lesion = n_lesion
        self.attn_mode = attn_mode
        self.initialize = initialize
        # global lesion representation learning
        self.m = 0.4

        self.lesionRrepresentation = nn.Parameter(torch.randn(1,self.n_lesion, in_features))
        self.normalcenter = nn.Parameter(torch.randn(1, self.L),requires_grad=False)
        self.postivecenter = nn.Parameter(torch.randn(1, self.L),requires_grad=False)
        # encoder instances ->

        self.token = nn.Parameter(torch.randn(1, 1, L))


        self.triple_optimizer = optimizer_triple(in_feature=in_features,out_feature=self.L,drop=dropout_patch)

        self.encoder_instances = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            nn.LayerNorm(self.L),
        )

        # encoder global lesion representation ->
        self.encoder_globalLesion = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            nn.LayerNorm(self.L),
        )


        self.crossffn = nn.Sequential(
            nn.Linear(self.L,self.L),
            nn.LayerNorm(self.L),
        )

        self.crossattention =  CrossLayer(dim=self.L,d=dropout_node)
        self.classifier = nn.Sequential(
            nn.Linear(self.L,num_classes)
        )


    def forward(self, x, bag_mode='normal'):

        x = self.triple_optimizer(x,mode='instances')

        H = self.encoder_instances(x)

        lesion_enhacing = self.triple_optimizer(self.lesionRrepresentation,mode='global')

        #x = self.triple_optimizer(x)
        #H = self.encoder_instances(x)
        #lesion_enhacing = self.triple_optimizer(self.lesionRrepresentation)

        lesion_token =  torch.cat((self.token,lesion_enhacing), dim=1)

        lesion = self.encoder_globalLesion(lesion_token)



        out,A = self.crossattention(lesion,H,H) # 1 x n x L -> 1 x 5 x n
        #out = out.permute(1,0,2)
        out = self.crossffn(out)
        out = out[:,0,:]

        # print(cls.shape)
        if self.training:
            # cls = self.classifier(out)
            # cls = cls.squeeze(0)
            #cls = torch.sum(cls,dim=0,keepdim=True)
            cls = self.classifier(out)
            with torch.no_grad():
                if bag_mode == 'normal':
                    x = x.squeeze(0)
                    negative_instances = torch.mean(x,dim=0,keepdim=True)
                    self._momentum_update_nc(negative_instances)

                else:
                    x = x.squeeze(0)
                    postive_instances = torch.mean(x,dim=0,keepdim=True)
                    self._momentum_update_p(postive_instances)

            return cls, A ,H, self.postivecenter,self.normalcenter, lesion_enhacing
        else:
            # cls = self.classifier(out)
            # #cls = cls.squeeze(0)
            # cls = torch.mean(cls,dim=1,keepdim=True)[0].squeeze(1)
            cls = self.classifier(out)
            #cls = torch.sum(cls,dim=0,keepdim=True)

            return cls, A ,H


    @torch.no_grad()
    def _momentum_update_p(self,postive):
        self.postivecenter.data = self.postivecenter.data * self.m + postive.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_nc(self,negative):

        self.normalcenter.data = self.normalcenter.data * self.m + negative.data * (1. - self.m)



@torch.no_grad()
def concat_all_gather(tensor):

    tensor_gather = [torch.ones_like(tensor)
                     for _ in range(torch.distributions.get_world.size())]

    torch.distributions.all_gather(tensor_gather,tensor,async_op = False)

    output = torch.cat(tensor_gather,dim=0)

    return output

if __name__ == "__main__":
    random_seed = 1
    batch_size = 1
    num_classes = 2  # CAMELYON16: 2, CAMELYON16: 3, CAMELYON16: 3, CAMELYON16: 2
    epoch = 100
    gpu_device = 0
    mode_stats = 'train'  # train or test
    weight_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/DGRMIL_CAMELYON16.pth'
    testing_weights_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/DGRMIL_CAMELYON16.pth'
    data_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/CAMELYON16/CAMELYON_pretrained_without_PE_public/'
    roc_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Predictions/CAMELYON16/DGRMIL'
    resu_text_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/DGRMIL_train_log.txt'
    layers_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/CAMELYON16/DGR_MIL'
    feats_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/CAMELYON16/DGR_MIL'

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

    dgrmil_net = DGRMIL(768, num_classes=num_classes)
    #test_x = torch.randn((2, 85, 768))
    #pre_y, _, _, _, _, _ = dgrmil_net(test_x)


    #print(pre_y.shape)

    dgrmil_net = dgrmil_net.cuda(gpu_device)

    #### lr_fn: vit_amu or vit_amu
    if mode_stats == 'train':
        training_for_dgrmil(mil_net=dgrmil_net, train_loader=train_loader, val_loader=val_loader,
                              test_loader=test_loader,
                              proba_mode=False, lr_fn='vit_amu', epoch=epoch, gpu_device=gpu_device,
                              onecycle_mr=1e-2, current_lr=None,
                              data_parallel=False, weight_path=weight_path, proba_value=0.85, class_num=num_classes,
                              bags_stat=True, resu_text_path=resu_text_path)

        dgrmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        dgrmil_net.load_state_dict(dgrmil_weight, strict=True)
        testing_for_dgrmil(test_model=dgrmil_net, train_loader=train_loader, val_loader=val_loader,
                             proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                             out_mode=None, proba_mode=False, class_num=num_classes, roc_save_path=roc_save_path,
                             bags_stat=True, bag_relations_path=None, resu_text_path=resu_text_path)

    elif mode_stats == 'test':
        dgrmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        dgrmil_net.load_state_dict(dgrmil_weight, strict=True)

        ### get layers vlaues
        from Results_codes.get_layers import get_layers

        get_layers(layer=dgrmil_net.classifier[0], save_path=layers_save_path)

        ### get features
        from Results_codes.get_features import Get_Features

        get_feats = Get_Features(layer=dgrmil_net.encoder_instances, end_no=-5, save_path=feats_save_path,
                                 out_or_in='out', with_pe = False)
        get_feats.regis_layer()

        new_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        testing_for_dgrmil(test_model=dgrmil_net, train_loader=new_train_loader, val_loader=val_loader,
                             proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                             out_mode=None, proba_mode=False, class_num=num_classes, roc_save_path=roc_save_path,
                             bags_stat=True, bag_relations_path=None, resu_text_path=resu_text_path)
        ### get features
        get_feats.get_feats_grads()

    else:
        assert print('mode state error!!!')