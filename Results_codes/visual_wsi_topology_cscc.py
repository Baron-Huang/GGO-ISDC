#############################  visiualizing whole-slide image via topology map ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: visiualizing whole-slide image via topology map

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import natsort
import os
import joblib
import matplotlib.font_manager as fm

font_path = "/home/root/.fonts/arial.ttf"
prop = fm.FontProperties(fname=font_path)


def visual_wsi_topology(inst_edges=None, group_arr=None, inst_name_list = None,
                        save_path = None, show_mode = True, offsets = None):
    '''
    :param inst_edges: tuple list, [(1, 3), (2, 5)]
    :param inst_name_list: list, ['Other', 'Tumor', 'Non-tumor']
    :param group_arr: list, [0, 0, 1, 1, 2, 2, 2]
    :return:
    '''
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    ### inst_nodes = [(1, {'category':Tumor})]
    inst_nodes = []
    cluster_assignments = {}
    for node_i, class_i in enumerate(group_arr):
        inst_nodes.append((node_i, {'category': inst_name_list[int(class_i)]}))
        cluster_assignments[node_i] = int(class_i)

    ##蓝色、绿色、黄色

    colors = ["#8b5cf6", "#f43f5e"]
    font_colors = ["red", "blue"]
    # ============ ★ 添加簇标签：在簇旁边，而不是中间 ============

    G = nx.Graph()
    G.add_nodes_from(inst_nodes)
    G.add_edges_from(inst_edges)

    # ---- spring layout ----
    pos = nx.spring_layout(G, seed=42, k=1.0, iterations=200)
    #pos = nx.kamada_kawai_layout(G)

    # ---- 节点颜色 ----
    node_color_list = [colors[cluster_assignments[n]] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    #fig.patch.set_facecolor('#f5f5f5')  # 整体背景
    #ax.set_facecolor('#f5f5f5')  # 绘图区背景

    # ---- 边 ----
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.5, width=2, edge_color="#000000"
    )

    # ---- 节点 ----
    nx.draw_networkx_nodes(
        G, pos,
        node_size=350,
        node_color=node_color_list,
        edgecolors="#333333",
        linewidths=0.6
    )

    for c in range(len(inst_name_list)):
        # 找出该簇的节点
        cluster_nodes = [n for n in G.nodes() if cluster_assignments[n] == c]

        xs = [pos[n][0] for n in cluster_nodes]
        ys = [pos[n][1] for n in cluster_nodes]
        center_x, center_y = np.mean(xs), np.mean(ys)

        off_x, off_y = offsets[c]

        # 标签放在簇中心的偏移位置（旁边）
        plt.text(
            center_x + off_x, center_y + off_y,
            inst_name_list[c],
            fontsize=70,
            fontstyle='normal',
            color=colors[c],
            ha="center", va="center",
            bbox=dict(
                facecolor="none",
                alpha=0.7,
                edgecolor="none",
                boxstyle="round,pad=0.4"
            )
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.02)
    if show_mode:
        plt.show()


def visu_wsi_topo_folder(folder_read_path = None, folder_save_path = None, inst_edges_mat=None,
                         group_mat=None, inst_name_mat = None, show_mode = True):
    images_name_list = natsort.natsorted(os.listdir(folder_read_path), alg=natsort.ns.PATH)
    print(images_name_list)

    for count_i, patch_name_i in enumerate(images_name_list):
        if not os.path.exists(folder_save_path):
            os.mkdir(folder_save_path)

        save_path = os.path.join(folder_save_path, patch_name_i + '.jpg')

        visual_wsi_topology(inst_edges=inst_edges_mat[count_i], group_arr=group_mat[count_i],
                            inst_name_list=inst_name_mat[count_i], save_path=save_path,
                            show_mode = show_mode)


def get_inputs(show_num=100):
    pass
    return inst_edges, group_arr, inst_name_list


if __name__ == '__main__':
    np.random.seed(0)

    ### single mode
    feats_full = joblib.load(r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/AMU_CSCC/GGO_ISDC/feats_full.joblib')
    relations = joblib.load(r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/GGO_ISDC/relations.joblib')
    labels = joblib.load(r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/GGO_ISDC/labels.joblib')

    inst_i = 0
    min_clus_show_num = 18
    max_clus_show_num = 18
    adje_value = 0.5
    offsets = [
        (0.59, 0.4),  # Cluster 0 标签放右上
        (-0.15, 0.0),  # Cluster 1 标签放左上
    ]

    feats_i = feats_full[inst_i]
    group_arr = relations[inst_i].tolist()[0]
    indices_0 = [i for i, v in enumerate(group_arr) if v == 0]
    indices_1 = [i for i, v in enumerate(group_arr) if v == 1]

    if (len(indices_0) > len(indices_1)):
        if len(indices_1) < min_clus_show_num:
            new_indices = indices_0[:max_clus_show_num] + indices_1
        else:
            new_indices = indices_0[:max_clus_show_num] + indices_1[:min_clus_show_num]
    else:
        if len(indices_0) < min_clus_show_num:
            new_indices = indices_0 + indices_1[:max_clus_show_num]
        else:
            new_indices = indices_0[:min_clus_show_num] + indices_1[:max_clus_show_num]

    feats_i = np.stack([feats_i[new_i] for new_i in new_indices])
    group_arr = [group_arr[new_i] for new_i in new_indices]

    label_i = labels[inst_i]

    #group_arr = [0 for i in range(100)] + [1 for i in range(100)]
    if label_i == 0:
        inst_name_list = ['Tumor', 'Non-tumor']
    else:
        inst_name_list = ['Non-tumor', 'Tumor']

    #edge_matx = np.random.rand(len(group_arr), len(group_arr))
    edge_matx = np.stack([np.sqrt(np.sum((feats_i[count_i] - feats_i) ** 2, axis=1)) for count_i in range(feats_i.shape[0])])
    edge_matx = 1 - edge_matx / np.max(edge_matx)

    cluster_assignments = {}
    for node_i, class_i in enumerate(group_arr):
        cluster_assignments[node_i] = class_i

    np.fill_diagonal(edge_matx, 0)
    inst_edges = []
    for i in range(edge_matx.shape[0]):
        for j in range(edge_matx.shape[1]):
            if edge_matx[i, j] > adje_value and cluster_assignments[i] == cluster_assignments[j]:
                inst_edges.append((i, j))

    save_root_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Visual/AMU_CSCC/GGO_ISDC/Topology_map'
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    save_path = os.path.join(save_root_path, 'wsi_topology_cscc_' + str(inst_i) + '.jpg')

    topo_inva_dict = {'Tumor':0, 'Non-tumor':0}
    inst_num_count_dict = {'Tumor': 0, 'Non-tumor': 0}

    for count_i, inst_edge_i in enumerate(inst_edges):
        print(count_i)
        topo_inva_dict[inst_name_list[int(group_arr[inst_edge_i[0]])]] += 1

    for count_i, inst_i in enumerate(group_arr):
        inst_num_count_dict[inst_name_list[int(inst_i)]] += 1

    topo_inva_dict['Tumor'] = topo_inva_dict['Tumor'] / inst_num_count_dict['Tumor']
    topo_inva_dict['Non-tumor'] = topo_inva_dict['Non-tumor'] / inst_num_count_dict['Non-tumor']


    visual_wsi_topology(inst_edges=inst_edges, group_arr=group_arr, inst_name_list = inst_name_list,
                        save_path = save_path, show_mode = True, offsets = offsets)

    print(topo_inva_dict)
    print(inst_name_list)


    ### folder mode
    '''
    group_arr = [0 for i in range(100)] + [1 for i in range(100)] + [2 for i in range(100)]
    inst_name_list = ['Other', 'Tumor', 'Non-tumor']
    edge_matx = np.random.rand(len(group_arr), len(group_arr))

    cluster_assignments = {}
    for node_i, class_i in enumerate(group_arr):
        cluster_assignments[node_i] = class_i

    np.fill_diagonal(edge_matx, 0)
    inst_edges = []
    for i in range(edge_matx.shape[0]):
        for j in range(edge_matx.shape[1]):
            if edge_matx[i, j] > 0.99 and cluster_assignments[i] == cluster_assignments[j]:
                inst_edges.append((i, j))

    files_read_path = r'F:\Pytorch_Projects\Visual_WSI\Results\Segs_delte'
    files_save_path = r'F:\Pytorch_Projects\Visual_WSI\Results\Visual\Topology_map'
    inst_edges_mat = [inst_edges, inst_edges]
    group_mat = [group_arr, group_arr]
    inst_name_mat = [inst_name_list, inst_name_list]


    visu_wsi_topo_files(files_read_path = files_read_path, files_save_path = files_save_path, inst_edges_mat=inst_edges_mat,
                        group_mat=group_mat, inst_name_mat = inst_name_mat, show_mode = False)
    '''

