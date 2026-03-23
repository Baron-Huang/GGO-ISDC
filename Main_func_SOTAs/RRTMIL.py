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



from torch import nn
from Main_func_SOTAs.APIs.emb_position import *
from Main_func_SOTAs.APIs.datten import *
from Main_func_SOTAs.APIs.rmsa import *
from Main_func_SOTAs.APIs.nystrom_attention import NystromAttention
from Main_func_SOTAs.APIs.datten import DAttention
from timm.models.layers import DropPath

from Utils.Setup_Seed import setup_seed
from torch.utils.data import DataLoader
from Read_Feats_Datasets import Read_Feats_Datasets
from Training_Testing_for_SOTA.training_testing_for_rrtmil import training_for_rrtmil, testing_for_rrtmil

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,attn='rmsa',n_region=8,epeg=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,crmsa_k=3,epeg_k=15,**kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

    def forward(self,x,need_attn=False):

        x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x,attn

class RRTEncoder(nn.Module):
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='rmsa',region_num=8,drop_out=0.1,n_layers=2,n_heads=8,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,epeg=True,epeg_k=15,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,peg_bias=True,peg_1d=False,cr_msa=True,crmsa_k=3,all_shortcut=False,crmsa_mlp=False,crmsa_heads=8,need_init=False,**kwargs):
        super(RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = []
        for i in range(n_layers-1):
            self.layers += [TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=region_num,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,epeg_k=epeg_k,**kwargs)]
        self.layers = nn.Sequential(*self.layers)
    
        # CR-MSA
        self.cr_msa = TransLayer(dim=mlp_dim,head=crmsa_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn='crmsa',qkv_bias=qkv_bias,crmsa_k=crmsa_k,crmsa_mlp=crmsa_mlp,**kwargs) if cr_msa else nn.Identity()

        # only for ablation
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4

        batch, num_patches, C = x.shape 
        x_shortcut = x

        # PEG/PPEG
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        
        # R-MSA within region
        for i,layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x+x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1,2)
            x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
        return x
    
class RRTMIL(nn.Module):
    def __init__(self, input_dim=1024,mlp_dim=512,act='relu',
                 n_classes=2,dropout=0.25,pos_pos=0,pos='none',peg_k=7,attn='rmsa',pool='attn',
                 region_num=8,n_layers=2,n_heads=8,drop_path=0.,da_act='relu',trans_dropout=0.1,
                 ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,
                 trans_dim=64,epeg=True,min_region_num=0,qkv_bias=True,**kwargs):
        super(RRTMIL, self).__init__()

        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.online_encoder = RRTEncoder(mlp_dim=mlp_dim,pos_pos=pos_pos,pos=pos,peg_k=peg_k,attn=attn,region_num=region_num,n_layers=n_layers,n_heads=n_heads,drop_path=drop_path,drop_out=trans_dropout,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,epeg=epeg,min_region_num=min_region_num,qkv_bias=qkv_bias,**kwargs)

        self.pool_fn = DAttention(self.online_encoder.final_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        
        self.predictor = nn.Linear(self.online_encoder.final_dim,n_classes)

        self.apply(initialize_weights)

    def forward(self, x, return_attn=False,no_norm=False):
        x = self.patch_to_emb(x) # n*512
        x = self.dp(x)
        
        # feature re-embedding
        x = self.online_encoder(x)
        
        # feature aggregation
        if return_attn:
            x,a = self.pool_fn(x,return_attn=True,no_norm=no_norm)
        else:
            x = self.pool_fn(x)

        # prediction
        logits = self.predictor(x)

        if return_attn:
            return logits,a
        else:
            return logits
        
if __name__ == "__main__":
    random_seed = 1
    batch_size = 1
    num_classes = 2  # CAMELYON16: 2, CAMELYON16: 3, CAMELYON16: 3, CAMELYON16: 3
    epoch = 100
    gpu_device = 1
    mode_stats = 'train'  # train or test
    weight_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/RRTMIL_CAMELYON16.pth'
    testing_weights_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/RRTMIL_CAMELYON16.pth'
    data_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/CAMELYON16/CAMELYON_pretrained_without_PE_public/'
    roc_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Predictions/CAMELYON16/RRTMIL'
    resu_text_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/RRTMIL_train_log.txt'

    layers_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/CAMELYON16/RRTMIL'
    feats_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/CAMELYON16/RRTMIL'

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

    rrtmil_net = rrt_mil = RRTMIL(n_classes=num_classes, input_dim=768)

    # x = torch.randn((85, 768))
    # _, y = frmil_net(x)
    # print(y.shape)
    rrtmil_net = rrtmil_net.cuda(gpu_device)

    if mode_stats == 'train':
        training_for_rrtmil(mil_net=rrtmil_net, train_loader=train_loader, val_loader=val_loader,
                           test_loader=test_loader, proba_mode=False, lr_fn='vit_amu', epoch=epoch,
                           gpu_device=gpu_device, onecycle_mr=1e-2, current_lr=None, data_parallel=False,
                           weight_path=weight_path, proba_value=0.85, class_num=num_classes, bags_stat=True,
                            resu_text_path=resu_text_path)

        rrtmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        rrtmil_net.load_state_dict(rrtmil_weight, strict=True)
        testing_for_rrtmil(test_model=rrtmil_net, train_loader=train_loader, val_loader=val_loader,
                           proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode=None, proba_mode=False, class_num=num_classes,
                           roc_save_path=roc_save_path, bags_stat=True, bag_relations_path=None,
                           resu_text_path=resu_text_path)

    elif mode_stats == 'test':
        rrtmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        rrtmil_net.load_state_dict(rrtmil_weight, strict=True)

        ### get layers vlaues
        from Results_codes.get_layers import get_layers

        get_layers(layer=rrtmil_net.predictor, save_path=layers_save_path)

        ### get features
        from Results_codes.get_features import Get_Features

        get_feats = Get_Features(layer=rrtmil_net.dp, end_no=-5, save_path=feats_save_path,
                                 out_or_in='in', with_pe = False)
        get_feats.regis_layer()

        new_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        testing_for_rrtmil(test_model=rrtmil_net, train_loader=new_train_loader, val_loader=val_loader,
                          proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                          out_mode=None, proba_mode=False, class_num=num_classes,
                          roc_save_path=roc_save_path, bags_stat=True, bag_relations_path=None,
                           resu_text_path=resu_text_path)

        ### get features
        get_feats.get_feats_grads()

    else:
        assert print('error!!!')