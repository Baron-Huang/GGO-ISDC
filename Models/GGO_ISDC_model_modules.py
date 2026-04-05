############################# GGO_ISDC models function ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Helath, PolyU, Hong Kong
#### Attempt: creating GGO_ISDC model by loading pretrained weight for searching the best learning rate
import time
from numpy.random import geometric

########################## API Section #########################
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from torch import nn
import torch
from torchsummaryX import summary
import random
import torch.nn.functional as F
import numpy as np
import torch_geometric
from Main_func_SOTAs.APIs.nystrom_attention import NystromAttention
from torch_geometric.nn import GCNConv
from torch.nn import init


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.relu(self.conv1(x, edge_index))
        x_feats = self.conv2(x, edge_index)
        return x_feats


class Game_Graph_Tensor(nn.Module):
    def __init__(self, in_size = 768, hidden_size = 768 * 2, out_size = 768,
                 graph_thre_value = 0.35, growth_vulue = 1.1, close_rate = 0.75):
        super(Game_Graph_Tensor, self).__init__()
        self.graph_thre_w = nn.Parameter(torch.tensor(graph_thre_value))
        self.growth_w = nn.Parameter(torch.tensor(growth_vulue), requires_grad=True)
        self.gcn_blcok = GCN(in_channels=in_size, hidden_channels=hidden_size, out_channels=out_size)
        self.close_rate = close_rate


    def create_graph_tensor(self, x):
        graph_tensor = torch.stack([torch.sqrt(torch.sum((x[x_i] - x) ** 2, dim=1)) for x_i in range(x.shape[0])])
        org_value = torch.quantile(graph_tensor, self.close_rate)
        graph_tensor[graph_tensor > org_value] = torch.tensor(0)
        k = torch.count_nonzero(graph_tensor)
        graph_tensor = torch.pow(graph_tensor, self.growth_w)
        graph_tensor = graph_tensor / torch.max(graph_tensor)
        #print(graph_tensor.shape)
        # print(torch.mean(graph_tensor), graph_tensor.shape)
        graph_tensor[graph_tensor == torch.tensor(0)] = self.graph_thre_w.detach()
        graph_tensor[graph_tensor < self.graph_thre_w.detach()] = torch.tensor(1.0)
        graph_tensor[graph_tensor != torch.tensor(1.0)] = torch.tensor(0.0)
        return graph_tensor

    def forward(self, x):
        graph_tensor = self.create_graph_tensor(x)
        x_edges_list = torch.where(graph_tensor == torch.tensor(1.0))

        if x_edges_list != []:
            x_edges = torch.stack(x_edges_list)
            graph_feats = self.gcn_blcok(x, x_edges)
            return graph_tensor, graph_feats
        else:
            return torch.tensor(0), torch.tensor(0)


class Game_Graph_Tensor_Cluster(nn.Module):
    def __init__(self, k_nums = 3, sel_dis = 'l2', train_iters = 50, p = 1, feature_lens = 768, in_size = 768,
                 hidden_size = 768 * 2, out_size = 768, graph_thre_value = 0.35, growth_vulue = 1.1, close_rate = 0.75):
        super().__init__()
        self.k_nums = k_nums
        self.sel_dis = sel_dis
        self.train_iters = train_iters
        self.p = p
        self.feature_lens = feature_lens
        self.ggt_block = Game_Graph_Tensor(
            in_size = in_size, hidden_size = hidden_size, out_size = out_size,
            graph_thre_value = graph_thre_value, growth_vulue = growth_vulue,
            close_rate = close_rate
        )

    def l2_distance(self, x, y):
        return torch.squeeze(torch.sqrt((x - y).permute(1, 0) @ (x - y)))

    def l2_distance_mat_ver(self, x, y):
        x_y = x - y.expand(x.shape[0], -1)
        x_y_2 = x_y ** 2
        dis_sum = torch.sum(x_y_2, dim=1)
        dis_sqrt = torch.sqrt(dis_sum)
        dis_value = dis_sqrt.reshape(dis_sqrt.shape[0], 1)
        return dis_value

    def l1_distance(self, x, y):
        return torch.sum(torch.abs(x - y))

    def lmax_distance(self, x, y):
        return torch.max(torch.abs(x - y))

    def lp_distance(self, x, y, p):
        lp_sum = 0
        for i in range(int(x.shape[0])):
            lp_sum += (x[i] - y[i]) ** p
        lp_sum = torch.abs(lp_sum) ** (1 / p)
        return lp_sum

    def init_cluster_centre(self, x, k_num):
        y_shape = x.shape[1]
        clus_center = torch.zeros((1, y_shape)).cuda()
        for i in range(k_num):
            clus_center_k = torch.zeros((1, 1)).cuda()
            for j in range(y_shape):
                clus_center_inter = torch.tensor(random.uniform(torch.max(x[:, j]), torch.min(x[:, j])))
                clus_center_inter = torch.reshape(clus_center_inter, (1, 1)).cuda()
                clus_center_k = torch.cat((clus_center_k, clus_center_inter), dim=1)
            clus_center_k = clus_center_k[:, 1:]
            clus_center = torch.cat((clus_center, clus_center_k))
        clus_center = clus_center[1:, :]
        return clus_center

    def init_cluster_centre_simple(self, x, k_num):
        y_shape = x.shape[1]
        clus_center = torch.randn((k_num, y_shape)).cuda()
        return clus_center

    def assign_data_point(self, x, init_cluster_cen):
        assigned_set = {}
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []

        for i in range(x.shape[0]):
            cont_dis = torch.zeros((1, 1)).cuda()
            for j in range(init_cluster_cen.shape[0]):
                if self.sel_dis == 'l2':
                    dis_value = \
                    self.l2_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'l1':
                    dis_value = \
                    self.l1_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'lp':
                    dis_value = \
                self.lp_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1), p=self.p)
                elif self.sel_dis == 'lmax':
                    dis_value = \
                    self.lmax_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                else:
                    print('assign_data_point error!!!')
                cont_dis = torch.cat((cont_dis, dis_value.reshape(1, 1)))
            cont_dis = cont_dis[1:, :]
            max_id = torch.argmin(cont_dis).cpu().numpy()
            assigned_set[str(max_id)].append(i)
        return assigned_set

    def assign_data_point_mat_ver(self, x, init_cluster_cen):
        assigned_set = {}
        init_cluster_order_matrix = torch.zeros((x.shape[0], 1)).cuda()
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []
            if self.sel_dis == 'l2':
                dis_value = self.l2_distance_mat_ver(x, init_cluster_cen[i])
            else:
                print('assign_data_point_mat_ver error!!!')
            init_cluster_order_matrix = torch.cat((init_cluster_order_matrix, dis_value), dim=1)
        init_cluster_order_matrix = init_cluster_order_matrix[:, 1:]
        init_cluster_order = torch.argmin(init_cluster_order_matrix, dim=1)
        for i in range(init_cluster_cen.shape[0]):
            k = torch.nonzero(init_cluster_order == torch.tensor(i)).detach().cpu().numpy()
            k = list(k.reshape((k.shape[0])))
            assigned_set[str(i)] = k
        return assigned_set

    def upgrade_cluster_centre(self, x, assigned_set):
        new_centre = torch.zeros((1, x.shape[1])).cuda()
        for i in range(len(assigned_set)):
            new_inter = torch.mean(x[assigned_set[str(i)], :], dim=0)
            new_centre = torch.cat((new_centre, new_inter.reshape(1, x.shape[1])))
        new_centre = new_centre[1:, :]
        return new_centre

    def forward(self, x):
        k = self.k_nums
        clus_center = self.init_cluster_centre(x, self.k_nums)
        for train_i in range(self.train_iters):
            assigned_set = self.assign_data_point_mat_ver(x, clus_center)
            new_centre = self.upgrade_cluster_centre(x, assigned_set)
            if torch.mean(new_centre) == torch.mean(clus_center):
                break
            else:
                clus_center = new_centre

        x_graph_tensor, x_graph_feats = self.ggt_block(x)

        return assigned_set, x_graph_tensor, x_graph_feats


class Mean_Layer(nn.Module):
    def __init__(self, dim = 1, keepdim = True):
        super(Mean_Layer, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        y = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        return y


class GGO_ISDC_Head(nn.Module):
    def __init__(self, base_model = None, class_num = 3, bags_len = 1042, model_stats = 'train', dis_rate_lamda = 0.01,
                 batch_size = 2, k_nums = 2, graph_thre_value = 0.35, growth_vulue = 1.1, close_rate = 0.75,
                 test_seed = 1):
        super(GGO_ISDC_Head, self).__init__()
        self.head = nn.Linear(in_features=768, out_features=class_num)
        self.head_graph_feats = nn.Linear(in_features=768, out_features=class_num)
        self.head_sum_feats = nn.Linear(in_features=768 * 2, out_features=class_num)
        init.kaiming_normal(self.head.weight)
        init.kaiming_normal(self.head_graph_feats.weight)
        init.kaiming_normal(self.head_sum_feats.weight)
        self.bags_len = bags_len
        self.ggtc_block = Game_Graph_Tensor_Cluster(k_nums = k_nums, sel_dis='l2', graph_thre_value=graph_thre_value,
                                                    growth_vulue=growth_vulue, close_rate = close_rate)
        self.model_stats = model_stats
        self.batch_size = batch_size
        self.lamda_feat_tumor = 0.2
        self.feat_subspace_w = nn.Parameter(torch.randn(768, 768 // 4), requires_grad=True)
        self.samp_count = 0
        self.pooling = Mean_Layer(dim=0, keepdim=True)
        self.test_seed = test_seed


    def get_relat_clus_fail(self, input_y, assign_set_list):
        if self.model_stats == 'test':
            dis_all_list = []
            centre_inst = torch.mean(input_y, dim=0, keepdim=True).permute(1, 0)
            # print(prior_inst.shape)
            for tensor_i in range(input_y.shape[0]):
                dis_all_i = self.ggtc_block.l2_distance(input_y[tensor_i, :].reshape(input_y.shape[1], 1), centre_inst)
                dis_all_list.append(dis_all_i.detach().cpu().numpy())
            assign_set_list.append(dis_all_list)
        else:
            pass


    def get_relat_clus_succ(self, assign_y_0, assign_y_1, assign_sets, assign_set_list):
        if self.model_stats == 'test':
            dis_all_np = np.zeros((assign_y_0.shape[0] + assign_y_1.shape[0], 1))
            centre_0 = torch.mean(assign_y_0, dim=0, keepdim=True).permute(1, 0)
            centre_1 = torch.mean(assign_y_1, dim=0, keepdim=True).permute(1, 0)
            # print(prior_inst.shape)
            for tensor_i in range(assign_y_0.shape[0]):
                inter_tens = assign_y_0[tensor_i, :].reshape(assign_y_0.shape[1], 1)
                #print(inter_tens.shape, centre_0.shape)
                dis_all_i = self.ggtc_block.l2_distance(inter_tens, centre_0)
                dis_all_np[assign_sets['0'][tensor_i]] = dis_all_i.detach().cpu().numpy()

            for tensor_i in range(assign_y_1.shape[0]):
                dis_all_i = self.ggtc_block.l2_distance(assign_y_1[tensor_i, :].reshape(assign_y_1.shape[1], 1), centre_1)
                dis_all_np[assign_sets['1'][tensor_i]] = dis_all_i.detach().cpu().numpy()

            assign_set_list.append(dis_all_np.tolist())
        else:
            pass


    def forward(self, x):
        assign_set_list = []
        self.samp_count += 1
        if self.model_stats == 'train':
            pass
        elif self.model_stats == 'test':
            from Utils.Setup_Seed import setup_seed
            setup_seed(self.test_seed)
        else:
            print('error!!!!')

        if x.dim == 3:
            y = torch.reshape(x, (x.shape[1], x.shape[2]))
        else:
            y = x + 0

        assigned_sets, y_graph_tensor, y_graph_feats = self.ggtc_block.forward(y.clone())
        assign_set_list.append(assigned_sets)
        assign_y_0 = y[assigned_sets['0'], :]
        assign_y_1 = y[assigned_sets['1'], :]

        if assign_y_0.shape == (0, 768):
            tumor_graph_feats = torch.tensor(0).cuda()
            assign_set_list.append(1)
            self.get_relat_clus_fail(y, assign_set_list)
        elif assign_y_1.shape == (0, 768):
            tumor_graph_feats = torch.tensor(0).cuda()
            assign_set_list.append(0)
            self.get_relat_clus_fail(y, assign_set_list)
        else:
            assign_graph_tensor_0 = self.ggtc_block.ggt_block.create_graph_tensor(assign_y_0)
            assign_graph_tensor_1 = self.ggtc_block.ggt_block.create_graph_tensor(assign_y_1)
            k = torch.count_nonzero(assign_graph_tensor_0)
            p = torch.count_nonzero(assign_graph_tensor_1)
            print(k, p)
            cluster_list = [torch.mean(assign_graph_tensor_0), torch.mean(assign_graph_tensor_1)]
            cluster_dis = torch.stack(cluster_list)
            min_ord = int(torch.argmin(cluster_dis.clone()))
            max_ord = int(torch.argmax(cluster_dis.clone()))
            assign_set_list.append(min_ord)
            _, tumor_graph_feats = self.ggtc_block.ggt_block.forward(y[assigned_sets[str(min_ord)], :])
            self.get_relat_clus_succ(assign_y_0, assign_y_1, assigned_sets, assign_set_list)

        print(self.samp_count, len(assigned_sets['0']), len(assigned_sets['1']))


        if np.sum(tumor_graph_feats.detach().cpu().numpy()) == 0:
            y_tumor = tumor_graph_feats
        else:
            y_tumor = torch.mean(tumor_graph_feats, dim=0, keepdim=True)
        y = self.pooling(y)
        y = y + (self.lamda_feat_tumor * y_tumor)

        y_graph = torch.mean(y_graph_feats, dim=0, keepdim=True)
        y_sum = torch.concat((y, y_graph), dim=1)

        y_graph = self.head_graph_feats(y_graph)
        y_sum = self.head_sum_feats(y_sum)
        y_org = self.head(y)

        if self.model_stats == 'test':
            try:
                print(assign_set_list[2].shape)
            except:
                print('instance_important:', len(assign_set_list[2]))

        return y_org, y_graph, y_sum, assign_set_list


class GGO_ISDC_Feature_Parallel_swints(nn.Module):
    def __init__(self, base_model=None):
        super(GGO_ISDC_Feature_Parallel_swints, self).__init__()
        self.frozen_layers = torch.nn.Sequential(
            base_model.patch_embed, base_model.pos_drop, base_model.layers[0].blocks[0], base_model.layers[0].blocks[1],
            base_model.layers[0].downsample, base_model.layers[1].blocks[0], base_model.layers[1].blocks[1],
            base_model.layers[1].downsample, base_model.layers[2].blocks[0], base_model.layers[2].blocks[1],
            base_model.layers[2].blocks[2], base_model.layers[2].blocks[3], base_model.layers[2].blocks[4],
            base_model.layers[2].blocks[5], base_model.layers[2].blocks[6], base_model.layers[2].blocks[7],
            base_model.layers[2].blocks[8], base_model.layers[2].blocks[9], base_model.layers[2].blocks[10],
            base_model.layers[2].blocks[11], base_model.layers[2].blocks[12], base_model.layers[2].blocks[13],
            base_model.layers[2].blocks[14]
        )
        self.training_layers = torch.nn.Sequential(
            base_model.layers[2].blocks[15], base_model.layers[2].blocks[16], base_model.layers[2].blocks[17],
            base_model.layers[2].downsample, base_model.layers[3].blocks[0], base_model.layers[3].blocks[1],
            base_model.norm
        )
        self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)


    def training_func(self, x):
        with torch.no_grad():
            y = self.frozen_layers(x)

        y = self.training_layers(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y

    def forward(self, x, con_l):
        y = self.training_func(x)
        return y

class GGO_ISDC_Feature_Parallel_AMU(nn.Module):
    def __init__(self, base_model=None):
        super(GGO_ISDC_Feature_Parallel_AMU, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=9, stride=9)

    def forward(self, x, con_l):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y


class GGO_ISDC_Feature_Parallel_Public(nn.Module):
    def __init__(self, base_model=None, pre_hot = 1):
        super(GGO_ISDC_Feature_Parallel_Public, self).__init__()
        self.layers_1 = torch.nn.Sequential(
            base_model.patch_embed, base_model.pos_drop, base_model.layers[0].blocks[0],
            base_model.layers[0].blocks[1], base_model.layers[0].downsample, base_model.layers[1].blocks[0],
            base_model.layers[1].blocks[1], base_model.layers[1].downsample,  base_model.layers[2].blocks[0],
            base_model.layers[2].blocks[1], base_model.layers[2].blocks[2], base_model.layers[2].blocks[3],
        )
        self.layers_2 = torch.nn.Sequential(
            base_model.layers[2].blocks[4], base_model.layers[2].blocks[5], base_model.layers[2].downsample,
            base_model.layers[3].blocks[0], base_model.layers[3].blocks[1],
            base_model.norm
        )
        self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
        self.pre_hot = pre_hot


    def training_func_1(self, x):
        y = self.layers_1(x)

        with torch.no_grad():
            y = self.layers_2(y)

        #print(y.shape)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y

    def training_func_2(self, x):
        with torch.no_grad():
            y = self.layers_1(x)

        y = self.layers_2(y)

        #print(y.shape)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y

    def forward(self, x, con_l):
        if con_l <= self.pre_hot:
            y = self.training_func_1(x)
        else:
            y = self.training_func_2(x)
        return y


class GGO_ISDC_Head_Parallel_simple(nn.Module):
    def __init__(self, base_model = None, class_num = 3):
        super(GGO_ISDC_Head_Parallel_simple, self).__init__()
        self.head = base_model.head

    def forward(self, x):
        y = torch.mean(x, dim=0, keepdim=True)
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        y = self.head(y)
        return y


class GGO_ISDC_Head_Parallel_visual(nn.Module):
    def __init__(self, base_model = None, class_num = 3):
        super(GGO_ISDC_Head_Parallel_visual, self).__init__()
        self.head = base_model.head

    def forward(self, x):
        y = torch.mean(x, dim=1, keepdim=True)
        return y


class Baseline_Head_Parallel(nn.Module):
    def __init__(self, base_model = None, class_num = 3):
        super(Baseline_Head_Parallel, self).__init__()
        self.head = base_model.head

    def forward(self, x):
        y = torch.mean(x, dim=0, keepdim=True)
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        y = self.head(y)
        return y


if __name__ == '__main__':
    '''
    swinT_base = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=3,
                                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                 use_checkpoint=False, fused_window_process=False)
    GGO_ISDC_net = GGO_ISDC_Feature_Parallel_CAMELYON16(swinT_base)
    print(GGO_ISDC_net)

    x = torch.randn((50, 3, 224, 224))
    y = GGO_ISDC_net(x, con_l = 2)
    print(y.shape)
    '''
    torch.random.manual_seed(1)
    net = Game_Graph_Tensor_Cluster().cuda()
    x_test = torch.randn((251, 768)).cuda()
    y_graph, y_feats = net(x_test)
    print(y_graph.mean())
    print(y_feats.shape)

    x = torch.tensor(3)
    print(torch.pow(2, x))



