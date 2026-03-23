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

from Main_func_SOTAs.APIs.HAG_MIL_utils import Attn_Net, Attn_Net_Gated, TransLayer

from Utils.Setup_Seed import setup_seed
from torch.utils.data import DataLoader
from Read_Feats_Datasets import Read_Feats_Datasets
from Training_Testing_for_SOTA.training_testing_for_hagmil import training_for_hagmil, testing_for_hagmil

def create_attention_net(feat_in, sz, dropout, gate):
    fc = [nn.Linear(feat_in, sz[1]), nn.GELU()]
    if dropout:
        fc.append(nn.Dropout(dropout))
    if gate:
        attention_net = Attn_Net_Gated(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    else:
        attention_net = Attn_Net(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    fc.append(attention_net)
    return nn.Sequential(*fc)

class IAMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim, size, dropout, gate):
        super(IAMBlock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.attention_net = create_attention_net(out_dim, size, dropout, gate)
        self.layer = TransLayer(dim=out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.layer_map_to_final = nn.Sequential(nn.Linear(out_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())
        
    def aggregate(self, h, norm_func, attn_net):
        _h = norm_func(h).squeeze()
        A, _h = attn_net(_h)  # NxK    
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        result = (torch.mul(h.squeeze().T, A.squeeze())).sum(dim=1).unsqueeze(0)
        slide_level = norm_func(result)
        return A, A_raw, slide_level, _h
    
    def forward(self, h):
        h = self.fc(h)
        h = self.layer(h)
        A, A_raw, slide_level, _h = self.aggregate(h, self.norm, self.attention_net)
        slide_level = self.layer_map_to_final(slide_level)
        return h, A, A_raw, slide_level, _h


class IATModel(nn.Module):
    def __init__(self, gate=True, dropout=0.25, k_sample=8, n_classes=3,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feat_in=1024, hidden_dims=[1024,1536,512,1024]):
        super(IATModel, self).__init__()
        self.n_classes = n_classes
        size = [feat_in,512, 256]
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.weights = nn.Linear(len(hidden_dims), 1)

        self.layers = []
        self.layers.append(IAMBlock(feat_in, hidden_dims[0], hidden_dims[-1], size, dropout, gate))
        for i in range(len(hidden_dims)-1):
            self.layers.append(IAMBlock(hidden_dims[i], hidden_dims[i+1], hidden_dims[-1], size, dropout, gate))
        self.layers = nn.ModuleList(self.layers)

        self.fc5 = nn.Linear(hidden_dims[-1], self.n_classes)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device, dtype=torch.long)
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device, dtype=torch.long)
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits.cpu(), all_targets.cpu())
        return instance_loss.to(device), all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits.cpu(), p_targets.cpu())
        return instance_loss.to(device), p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False):

        h = h.unsqueeze(0)
        slide_reps = []
        for i in range(len(self.layers)):
            h, A, A_raw, slide_level, _h = self.layers[i](h)
            slide_reps.append(slide_level)

        h = torch.cat(slide_reps, dim=0)
        h = self.weights(h.T).T

        logits = self.fc5(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, _h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, _h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        return logits, Y_prob, Y_hat, A_raw, results_dict


if __name__ == "__main__":
    random_seed = 1
    batch_size = 1
    num_classes = 2  # CAMELYON16: 2, CAMELYON16: 5, CAMELYON16: 3
    epoch = 100
    gpu_device = 1
    mode_stats = 'train'  # train or test
    weight_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/HAGMIL_CAMELYON16.pth'
    testing_weights_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/HAGMIL_CAMELYON16.pth'
    data_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/CAMELYON16/CAMELYON_pretrained_without_PE_public/'
    roc_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Predictions/CAMELYON16/HAGMIL'
    resu_text_path = \
        r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/CAMELYON16/Other_models/HAGMIL_train_log.txt'

    layers_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/CAMELYON16/HAG_MIL'
    feats_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/CAMELYON16/HAG_MIL'

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

    hagmil_net = IATModel(feat_in=768, n_classes=num_classes)

    #x = torch.randn((85, 768))
    #_, y = frmil_net(x)
    #print(y.shape)
    hagmil_net = hagmil_net.cuda(gpu_device)

    if mode_stats == 'train':
        training_for_hagmil(mil_net=hagmil_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                           proba_mode=False, lr_fn='vit_amu', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2,
                           current_lr=None, data_parallel=False, weight_path=weight_path, proba_value=0.85,
                           class_num = num_classes, bags_stat = True, resu_text_path=resu_text_path)

        hagmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        hagmil_net.load_state_dict(hagmil_weight, strict=True)
        testing_for_hagmil(test_model=hagmil_net, train_loader=train_loader, val_loader=val_loader,
                           proba_value=None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode=None, proba_mode=False, class_num=num_classes,
                           roc_save_path=roc_save_path, bags_stat=True, bag_relations_path=None,
                           resu_text_path=resu_text_path)

    elif mode_stats == 'test':
        hagmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        hagmil_net.load_state_dict(hagmil_weight, strict=True)

        ### get layers vlaues
        from Results_codes.get_layers import get_layers

        get_layers(layer=hagmil_net.fc5, save_path=layers_save_path)

        ### get features
        from Results_codes.get_features import Get_Features

        get_feats = Get_Features(layer=hagmil_net.layers[3], end_no=-5, save_path=feats_save_path,
                                 out_or_in='in', with_pe = False)
        get_feats.regis_layer()

        new_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        testing_for_hagmil(test_model = hagmil_net, train_loader=new_train_loader, val_loader=val_loader,
                           proba_value = None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode = None, proba_mode=False, class_num=num_classes,
                           roc_save_path = roc_save_path, bags_stat=True, bag_relations_path = None,
                           resu_text_path=resu_text_path)

        ### get features
        get_feats.get_feats_grads()

    else:
        assert print('error mode state!!!')
















