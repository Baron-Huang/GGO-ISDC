############################# GGO-ISDC Demo main function ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: Testing and Traning GGO-ISDC main function on the DHMC-Lung dataset
'''
Paper Title:
Graph Game Optimization-driven Instance Subgraph-density Disentanglement
Clustering for Learning SCC Topology-invariant
'''
####

########################## API Section #########################
import skimage.color
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchsummary import summary
from tensorboardX import SummaryWriter
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from Models.GGO_ISDC_model_modules import GGO_ISDC_ACTA, GGO_ISDC_Feature_Parallel_Public
from Utils.training_testing_for_GGO_ISDC import training_GGO_ISDC_parallel, testing_GGO_ISDC_parallel
from Utils.Setup_Seed import setup_seed
from Utils.Read_MIL_Datasets import Read_MIL_Datasets
from Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import PIL
import seaborn as sns
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from torch.nn.parallel import DataParallel
import argparse
from Utils.Read_MILDats_HDF5 import Read_MILDats_HDF5


sns.set(font='Times New Roman', font_scale=0.6)

def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)

########################## main_function #########################
if __name__ == '__main__':
    ########################## Hyparameters #########################
    paras = argparse.ArgumentParser(description='GGO_ISDC Hyparameters')
    paras.add_argument('--random_seed', type=int, default=1)
    paras.add_argument('--gpu_device', type=int, default=0)
    paras.add_argument('--class_num', type=int, default=2)  #CAMELYON16: 2
    paras.add_argument('--batch_size', type=int, default=1)
    paras.add_argument('--epochs', type=int, default=100)
    paras.add_argument('--img_size', type=list, default=[224, 224])
    paras.add_argument('--bags_len', type=int, default=1200)
    paras.add_argument('--input_len', type=int, default=500)
    paras.add_argument('--num_workers', type=int, default=16)
    paras.add_argument('--worker_time_out', type=int, default=0)
    paras.add_argument('--data_parallel', type=bool, default=True)
    paras.add_argument('--run_mode', type=str, default='test')  # train or test or visual
    paras.add_argument('--parallel_gpu_ids', type=list, default=[0, 1])
    paras.add_argument('--roc_save_path', type=str,
                default=r'/root/autodl-tmp/GGO_ISDC/Results/ROC/DHMC_Lung/GGO_ISDC_Lung.xlsx')
    paras.add_argument('--train_read_path', type=str,
                default=r'/root/autodl-tmp/GGO_ISDC_public/Datasets/DHMC_Lung/Lung_WSI_without_PE/Train')
    paras.add_argument('--test_read_path', type=str,
                default=r'/root/autodl-tmp/GGO_ISDC_public/Datasets/DHMC_Lung/Lung_WSI_without_PE/Test')
    paras.add_argument('--val_read_path', type=str,
                default=r'/root/autodl-tmp/GGO_ISDC_public/Datasets/DHMC_Lung/Lung_WSI_without_PE/Test')

    paras.add_argument('--weights_save_path', type=str,
        default=r'/mnt/Weights_Result_Text/GGO_ISDC/DHMC_Lung/GGO_ISDC_DHMC_Lung.pth')
    paras.add_argument('--test_weights_path', type=str,
            default=r'/root/autodl-tmp/Weights_Result_Text/GGO_ISDC/CAMELYON16/GGO_ISDC_DHMC_Lung.pth')


    ### Parallel save
    paras.add_argument('--weights_save_feature', type=str,
                        default=r'/root/autodl-tmp/Weights_Result_Text/GGO_ISDC/DHMC_Lung/GGO_ISDC_Feature_DHMC_Lung.pth')
    paras.add_argument('--weights_save_head',  type=str,
                        default=r'/root/autodl-tmp/Weights_Result_Text/GGO_ISDC/DHMC_Lung/GGO_ISDC_Head_DHMC_Lung.pth')

    ### Parallel test
    paras.add_argument('--test_weights_feature', type=str,
                        default=r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/DHMC_Lung/GGO_ISDC_Feature_DHMC_Lung.pth')
    paras.add_argument('--test_weights_head', type=str,
                        default=r'/root/autodl-tmp/GGO_ISDC_public/Results/Our_wegts/DHMC_Lung/GGO_ISDC_Head_DHMC_Lung.pth')

    ### Pretrained
    paras.add_argument('--pretrained_weights_path', type=str,
                        default=r'/root/autodl-tmp/Weights/SwinT/swin_tiny_patch4_window7_224_22k.pth')

    args = paras.parse_args()
    setup_seed(args.random_seed)

    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    train_data = Read_MIL_Datasets(read_path=args.train_read_path ,img_size=args.img_size, bags_len=args.bags_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    test_data = Read_MIL_Datasets(read_path=args.test_read_path, img_size=args.img_size, bags_len=args.bags_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    val_data = Read_MIL_Datasets(read_path=args.val_read_path, img_size=args.img_size, bags_len=args.bags_len)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              timeout=args.worker_time_out)
    print('train_data:', '\n', train_data, '\n')

    #for i, j in train_loader:
    #    print(i.shape)
    #    print(j.shape)

    ########################## creating models and visuling models #########################
    print('########################## creating models and visuling models #########################')
    swinT_base = SwinTransformer(img_size=args.img_size[0], patch_size=4, in_chans=3, num_classes=args.class_num,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False)

    checkpoint = torch.load(args.pretrained_weights_path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = swinT_base.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            #logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = swinT_base.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            #logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = swinT_base.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            #logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(swinT_base.head.bias, 0.)
            torch.nn.init.constant_(swinT_base.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']

    swinT_base.load_state_dict(state_dict, strict=False)

    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)
    print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    ### creating a SPE-MIL model
    if args.data_parallel == False:
        GGO_ISDC_net = None
        with torch.no_grad():
            print('########################## SwinT_summary #########################')
            summary(GGO_ISDC_net, (3, args.img_size[0], args.img_size[1]), device='cpu')
            print('\n', '########################## SwinT_net #########################')
            print(GGO_ISDC_net, '\n')

            print(GGO_ISDC_net(torch.zeros(10, 3, args.img_size[0], args.img_size[1])).shape)
        GGO_ISDC_net = GGO_ISDC_net.cuda(args.gpu_device)
    else:
        GGO_ISDC_feature = GGO_ISDC_Feature_Parallel_Public(base_model=swinT_base, pre_hot = 5)
        GGO_ISDC_head = GGO_ISDC_ACTA(base_model=swinT_base, class_num=args.class_num, bags_len=args.bags_len,
                                    batch_size = args.batch_size, dis_rate_lamda = 0.1, graph_thre_value = 0.25,
                                    growth_vulue=1.1, close_rate=0.5)
        #GGO_ISDC_head = TransMIL(768, n_classes=args.class_num, mDim=484)
        with torch.no_grad():
            print('########################## SwinT_summary #########################')
            #summary(GGO_ISDC_feature, (3, args.img_size[0], args.img_size[1]), device='cpu')
            #summary(GGO_ISDC_head, (768,), device='cpu')
            print('\n', '########################## SwinT_net #########################')
            print(GGO_ISDC_feature, '\n')
            print(GGO_ISDC_head, '\n')
        GGO_ISDC_feature = GGO_ISDC_feature.cuda()
        GGO_ISDC_feature = DataParallel(GGO_ISDC_feature, device_ids=args.parallel_gpu_ids)
        GGO_ISDC_head = GGO_ISDC_head.cuda()
        #ticmil_head = DataParallel(ticmil_head, device_ids=[0, 1])


    ########################## fitting models and testing models #########################
    if args.run_mode == 'train':
        print('########################## fitting models #########################')
        if args.data_parallel == False:
            training_GGO_ISDC_parallel(mil_net=GGO_ISDC_net, train_loader=train_loader, val_loader=val_loader,
                           proba_mode=args.proba_mode, test_loader=test_loader, lr_fn='vit_public', epoch=args.epochs,
                           gpu_device=args.gpu_device, weight_path=args.weights_save_path,
                           data_parallel=args.data_parallel, proba_value=args.proba_value,
                           class_num=args.class_num, bags_stat=args.bags_stat)
        else:   ### lr_fn: vit_public or vit_amu
            training_GGO_ISDC_parallel(mil_feature=GGO_ISDC_feature, mil_head=GGO_ISDC_head, train_loader=train_loader,
                                    val_loader=val_loader, test_loader=test_loader,
                                    lr_fn='vit_public', epoch=args.epochs, gpu_device=args.gpu_device,
                                    weight_path=args.weights_save_feature, num_class=args.class_num,
                                    bags_len=args.bags_len, batch_size=args.batch_size,
                                    weight_head_path=args.weights_save_head, input_len=args.input_len)

    if args.run_mode == 'test':
        print('########################## testing function #########################')
        if args.data_parallel == False:
            larynx_weight = torch.load(args.test_weights_path, map_location='cuda:0')
            GGO_ISDC_net.load_state_dict(larynx_weight, strict=True)
            testing_GGO_ISDC_parallel(test_model=GGO_ISDC_net, train_loader=test_loader, val_loader=val_loader,
                              proba_value=args.proba_value, test_loader=test_loader, gpu_device=args.gpu_device,
                              out_mode=None, proba_mode=args.proba_mode, class_num=args.class_num,
                              roc_save_path=args.roc_save_path, bags_stat=args.bags_stat,
                              bag_relations_path=args.bag_relations_path)
        elif args.data_parallel == True:
            head_weight = torch.load(args.test_weights_head, map_location='cuda:0')
            feature_weight = torch.load(args.test_weights_feature, map_location='cuda:0')
            GGO_ISDC_feature.load_state_dict(feature_weight, strict=True)
            GGO_ISDC_head.load_state_dict(head_weight, strict=True)
            testing_GGO_ISDC_parallel(mil_feature=GGO_ISDC_feature, mil_head=GGO_ISDC_head, train_loader=train_loader,
                                   data_parallel=args.data_parallel, num_class=args.class_num,
                                    batch_size=args.batch_size,input_len = args.input_len, roc_save_path= args.roc_save_path,
                                   bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)





