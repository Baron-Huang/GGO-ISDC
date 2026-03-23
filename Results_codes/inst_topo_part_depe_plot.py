############################# Instance-causality partial dependence plot ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: Instance-topology partial dependence plot
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re


def pred_model(X = None, W = None, b = None, pred_g = None):
    '''
    :param X: matrix, (N, M, C)
    :param W: matrix, (C, G)
    :param b: array, (G, )
    :param pred_g: array, (N, )
    :return:
    '''
    Y = np.exp(np.mean(X @ W + b, axis = 1))
    pred_g = np.argmax(Y, axis=1)
    for n_i in range(Y.shape[0]):
        Y[n_i, :] /= np.sum(Y[n_i, :])
    Y_scor = Y[:, int(pred_g)]
    return Y_scor


def plot_itpdp(X_rd_dict = None, x_rd = None):
    '''
    :param X_rd_dict: dict
    :param x_rd: array
    :return:
    '''
    means = {}
    stds = {}
    for x_i in X_rd_dict:
        res_last = re.search(r'_([^_]*)$', x_i).group(1)
        res_fore = re.search(r'^(.*)_', x_i).group(1)
        print(res_fore)
        print(res_last)
        if res_last == 'mean':
            means[res_fore] = np.array(X_rd_dict[x_i])
        elif res_last == 'std':
            stds[res_fore] = np.array(X_rd_dict[x_i])

    plt.figure(figsize=(6, 4), dpi = 400)

    for label in means:
        mean = means[label]
        std = stds[label]

        plt.plot(x_rd, mean, linewidth=2, label=label)
        plt.fill_between(
            x_rd,
            mean - std,
            mean + std,
            alpha=0.2
        )

    plt.xlabel("Disturbance value")
    plt.ylabel("ITPDP value")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()



def inst_topo_part_depe_plot(X = None, W = None, g_pred_arr = None, clus_label_arr = None,
                             show_mode = True, clus_name_list = None, rd_inter = None,
                             save_path = None, inst_topo_mat = None, lamda_rd = 0.9,
                             std_rate = 0.01):
    '''
    :param X: matrix, (N, M, C)
    :param W: matrix, (C, G)
    :param g_pred_arr: array, (N, )
    :param clus_label_arr: array, (N, )
    :param show_mode: boolean
    :param clus_name_list: dict
    :param rd_inter: int
    :param save_path: str
    :param inst_topo_mat: matrix, (N, M)
    :param lamda_rd: float
    :param show_mode: boolean
    :return:
    '''
    X_rd_dict = {}
    for label_i, name_i in enumerate(clus_name_list):
        opet_i_name = np.where(clus_label_arr == label_i)[0]
        nopt_i_name = list(set(np.arange(X.shape[1])) - set(opet_i_name))
        opet_mat = X[:, opet_i_name, :]
        nopt_mat = X[:, nopt_i_name, :]
        opet_caus_mat_mask = inst_topo_mat[:, opet_i_name]
        X_rd_dict[name_i] = [opet_mat, nopt_mat, opet_caus_mat_mask]

    Y_rd_dict = {}
    for label_i, name_i in enumerate(clus_name_list):
        inst_rd_mask = X_rd_dict[name_i][2].copy()
        score_list = []
        score_std_list = []
        for rd_i in rd_inter:
            if show_mode:
                print(rd_i)
            inst_rd_mask[inst_rd_mask > (1 - rd_i)] = 1
            inst_rd_mask[inst_rd_mask < (1 - rd_i)] = 0
            inst_rd_mask = inst_rd_mask.reshape(inst_rd_mask.shape[0], inst_rd_mask.shape[1], 1)
            inst_rd_mask_mat = np.concatenate([inst_rd_mask] * X_rd_dict[name_i][0].shape[2], axis=-1)
            X_rd = (lamda_rd * (np.zeros_like(X_rd_dict[name_i][0]) + rd_i) +
                    (1 - lamda_rd) * (np.zeros_like(X_rd_dict[name_i][0]) + rd_i) * inst_rd_mask)
            X_nrd = X_rd_dict[name_i][1]
            X_com = np.concatenate([X_rd, X_nrd], axis=1)
            scor_com = pred_model(X=X_com, W=W, b=b)
            score_list.append(np.mean(scor_com))
            score_std_list.append(np.mean(scor_com) * std_rate + 2 * std_rate * rd_i)
        Y_rd_dict[name_i + r'_mean'] = (score_list - min(score_list)) * 100
        Y_rd_dict[name_i + r'_std'] = score_std_list

    if show_mode:
        plot_itpdp(X_rd_dict = Y_rd_dict, x_rd = rd_inter)
        pf = pd.DataFrame(Y_rd_dict)
        pf.to_csv(save_path)
        print('yes!!!')

    return Y_rd_dict



def get_mat_inputs(dists_path = None, labels_path = None, relats_path = None,
                    feats_path = None, layers_path = None):
    import joblib
    import numpy as np
    distances_list = joblib.load(dists_path)
    labels_list = joblib.load(labels_path)
    features_list = joblib.load(feats_path)
    relations_list = joblib.load(relats_path)
    W = np.load(layers_path)

    rd_array = np.arange(0, 1, 0.02)
    clus_name_list = ['Tumor', 'Non-tumor']

    return distances_list, labels_list, features_list, relations_list, W, rd_array, clus_name_list



if __name__ == '__main__':
    dists_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/GGO_ISDC/distances.joblib'
    labels_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/GGO_ISDC/labels.joblib'
    relats_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/GGO_ISDC/relations.joblib'
    feats_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/AMU_CSCC/GGO_ISDC/feats_full.joblib'
    layers_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Layers/AMU_CSCC/GGO_ISDC/layers.npy'

    distances_list, labels_list, features_list, relations_list, W, rd_array, clus_name_list = \
                get_mat_inputs(dists_path=dists_path, labels_path=labels_path, relats_path=relats_path,
                                feats_path=feats_path, layers_path=layers_path)


    #### file mode

    sample_num = 0
    class_num = 3
    std_rate = 0.005
    X = features_list[sample_num].reshape(1, features_list[sample_num].shape[0], features_list[sample_num].shape[1])
    b = np.zeros(class_num, )

    if labels_list[sample_num] == 0:
        clus_name_list = ['Tumor', 'Non-tumor']
    elif labels_list[sample_num] == 1:
        clus_name_list = ['Non-tumor', 'Tumor']

    clus_label_arr = relations_list[sample_num]

    inst_topo_mat = np.array(distances_list[sample_num]) / max(distances_list[sample_num])
    inst_topo_mat = inst_topo_mat.reshape(1, inst_topo_mat.shape[0])

    save_path = os.path.join(r'/root/autodl-tmp/GGO_ISDC_public/Results/Explaining/AMU_CSCC',
                             'ITPDP_AMU_CSCC_' + str(sample_num) + '.csv')


    Y_rd_dict_i = inst_topo_part_depe_plot(X=X, W=W, g_pred_arr=clus_label_arr, clus_label_arr=clus_label_arr,
                                           show_mode=True, clus_name_list=clus_name_list, rd_inter=rd_array,
                                           save_path = save_path, inst_topo_mat = inst_topo_mat, std_rate = std_rate)



    #### folder mode
    '''
    Y_rd_dict_sum = {'Tumor_mean':[], 'Tumor_std':[], 'Non-tumor_mean':[], 'Non-tumor_std':[]}

    for sample_num in range(len(features_list)):
        print('smaple_no: ', sample_num)
        X = features_list[sample_num].reshape(1, features_list[sample_num].shape[0], features_list[sample_num].shape[1])
        b = np.zeros(2, )

        if labels_list[sample_num] == 0:
            clus_name_list = ['Tumor', 'Non-tumor']
        elif labels_list[sample_num] == 1:
            clus_name_list = ['Non-tumor', 'Tumor']

        clus_label_arr = relations_list[sample_num]

        inst_topo_mat = np.array(distances_list[sample_num]) / max(distances_list[sample_num])
        inst_topo_mat = inst_topo_mat.reshape(1, inst_topo_mat.shape[0])

        save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Explaining/AMU_CSCC/icpdp_resu_single.csv'

        Y_rd_dict_i = inst_topo_part_depe_plot(X=X, W=W, g_pred_arr=clus_label_arr, clus_label_arr=clus_label_arr,
                                               show_mode=False, clus_name_list=clus_name_list, rd_inter=rd_array,
                                               save_path=save_path, inst_topo_mat=inst_topo_mat)

        Y_rd_dict_sum['Tumor_mean'].append(Y_rd_dict_i['Tumor_mean'])
        Y_rd_dict_sum['Non-tumor_mean'].append(Y_rd_dict_i['Non-tumor_mean'])


    sum_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Explaining/AMU_CSCC/icpdp_resu.csv'

    tumor_mean_mat = np.stack(Y_rd_dict_sum['Tumor_mean'])
    non_tumor_mean_mat = np.stack(Y_rd_dict_sum['Non-tumor_mean'])

    Y_rd_dict_sum['Tumor_mean'] =  (np.mean(tumor_mean_mat, axis=0) - np.mean(tumor_mean_mat, axis=0).min()) * 100
    Y_rd_dict_sum['Non-tumor_mean'] = (np.mean(non_tumor_mean_mat, axis=0) - np.mean(non_tumor_mean_mat, axis=0).min()) * 100
    Y_rd_dict_sum['Tumor_std'] = np.std(tumor_mean_mat, axis=0)
    Y_rd_dict_sum['Non-tumor_std'] = np.std(non_tumor_mean_mat, axis=0)

    plot_itpdp(X_rd_dict=Y_rd_dict_sum, x_rd=rd_array)
    pf = pd.DataFrame(Y_rd_dict_sum)
    pf.to_csv(sum_save_path)
    print('all yes!!!')
    '''


