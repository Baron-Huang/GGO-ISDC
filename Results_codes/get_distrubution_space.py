############################# get features distribution and space function ##############################
#### Author: Dr. Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: get the outputs of feature space and distribution
import os

import joblib
import numpy as np
import pandas as pd
import natsort

from sklearn.decomposition import PCA


def get_feat_dist(rela_path = None, label_path = None, feats_full_path = None,
                  feat_dist_save_path = None, sample_label_path = None,
                  sele_num = None, clus_num = None, clus_name_list = None):
    '''
    :param rela_path:
    :param label_path:
    :param feats_full_path:
    :param feat_dist_save_path:
    :param sample_label_path:
    :param sele_num:
    :param clus_num:
    :param clus_name_list:
    :param class_num:
    :return:
    '''

    samp_label_name_list = natsort.natsorted(os.listdir(sample_label_path), alg=natsort.ns.PATH)

    samp_label_sing = []
    for clas_i in range(len(samp_label_name_list)):
        clas_list_i = natsort.natsorted(os.listdir(os.path.join(sample_label_path, samp_label_name_list[clas_i])),
                                        alg=natsort.ns.PATH)
        inter_list = [clas_i for i in range(len(clas_list_i))]
        samp_label_sing += inter_list

    samp_label = []  ### 获取的第一个数据
    for clus_i in range(clus_num):
        samp_label += samp_label_sing

    feats_full = joblib.load(feats_full_path)[sele_num:]
    realtions = joblib.load(rela_path)[sele_num:]
    labels = joblib.load(label_path)[sele_num:]

    get_value_dict = {}
    get_name_dict = {}

    for name_i in clus_name_list:
        get_value_dict[name_i] = []
        get_name_dict[name_i] = []

    for samp_i in range(len(feats_full)):
        feat_i = feats_full[samp_i]
        rela_i = realtions[samp_i]
        sele_num_0 = [i for i in range(rela_i.shape[1]) if rela_i[:, i] == 0]
        sele_num_1 = [i for i in range(rela_i.shape[1]) if rela_i[:, i] == 1]
        label_i = labels[samp_i]

        if label_i == 0:
            get_value_dict['Tumor'].append(np.mean(feat_i[sele_num_0], axis=0))
            get_name_dict['Tumor'].append('Tumor')
            get_value_dict['Non-tumor'].append(np.mean(feat_i[sele_num_1], axis=0))
            get_name_dict['Non-tumor'].append('Non-tumor')
        elif label_i == 1:
            get_value_dict['Tumor'].append(np.mean(feat_i[sele_num_1], axis=0))
            get_name_dict['Tumor'].append('Tumor')
            get_value_dict['Non-tumor'].append(np.mean(feat_i[sele_num_0], axis=0))
            get_name_dict['Non-tumor'].append('Non-tumor')
        else:
            assert print('label_i is error!!!')

        print(feats_full[samp_i].shape)

    tumor_np = np.stack(get_value_dict['Tumor'])
    non_tumor_np = np.stack(get_value_dict['Non-tumor'])

    non_tumor_np = np.nan_to_num(non_tumor_np, nan=0)
    tumor_np = np.nan_to_num(tumor_np, nan=0)

    tumor_list = list(PCA(n_components=1).fit_transform(tumor_np).squeeze())
    non_tumor_list = list(PCA(n_components=1).fit_transform(non_tumor_np).squeeze())

    get_value_dict['Tumor'] = tumor_list
    get_value_dict['Non-tumor'] = non_tumor_list

    feat_value = []  ### 获取的第二个数据
    feat_name = []  ### 获取的第三个数据

    for name_i in clus_name_list:
        feat_value += get_value_dict[name_i]
        feat_name += get_name_dict[name_i]

    feat_dist_dict = {'feat_name': feat_name, 'samp_label': samp_label, 'feat_value': feat_value}
    feat_dist_df = pd.DataFrame(feat_dist_dict)

    feat_dist_df.to_csv(feat_dist_save_path)

    print('yes!!!')



def get_feat_space(feats_full_path = None, feat_space_root_path = None,
                   sample_label_path = None, sele_num = None, clus_num = None,
                   class_num = None, model_name = None):
    '''
    :param feats_full_path:
    :param feat_space_root_path:
    :param sample_label_path:
    :param sele_num:
    :param clus_num:
    :param class_num:
    :return:
    '''

    samp_label_name_list = natsort.natsorted(os.listdir(sample_label_path), alg=natsort.ns.PATH)

    samp_label_sing = []
    for clas_i in range(len(samp_label_name_list)):
        clas_list_i = natsort.natsorted(os.listdir(os.path.join(sample_label_path, samp_label_name_list[clas_i])),
                                        alg=natsort.ns.PATH)
        inter_list = [clas_i for i in range(len(clas_list_i))]
        samp_label_sing += inter_list

    samp_label = []  ### 获取的第一个数据
    for clus_i in range(clus_num):
        samp_label += samp_label_sing

    feats_full = joblib.load(feats_full_path)[sele_num:]

    feat_space_list = [np.mean(i, axis=0) for i in feats_full]

    feat_space_npy = np.stack(feat_space_list)

    feat_space_npy = PCA(n_components=2).fit_transform(feat_space_npy).squeeze()

    feat_space_dict = {}  ### 获取feature space数据
    for new_i in range(class_num):
        sele_order_i = [i for i in range(len(samp_label_sing)) if samp_label_sing[i] == new_i]
        feat_space_dict['0'] = feat_space_npy[sele_order_i][:, 0]
        feat_space_dict['1'] = feat_space_npy[sele_order_i][:, 1]
        feat_space_df = pd.DataFrame(feat_space_dict)
        feat_space_df.to_csv(os.path.join(feat_space_root_path, model_name + '_AMU_CSCC_' + str(new_i) + '_feat_space.csv'))
        feat_space_dict = {}

    print('yes!!!')


if __name__ == '__main__':

    ### AMU-CSCC: 156, AMU_CSCC: 204 , CAMLE: 240 , Prosta: 96;
    sele_num = 156
    clus_num = 2
    clus_name_list = ['Tumor', 'Non-tumor']
    class_num = 3

    rela_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/RRTMIL/relations.joblib'
    label_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/AMU_CSCC/RRTMIL/labels.joblib'
    feats_full_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Features/AMU_CSCC/RRTMIL/feats_full.joblib'

    feat_dist_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Distribution_Space/Dist_AMU_CSCC_RRTMIL.csv'
    feat_space_root_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Distribution_Space'

    sample_label_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/AMU_CSCC/CSCC_WSI_without_PE_public/Test'

    #get_feat_dist(rela_path = rela_path, label_path = label_path, feats_full_path = feats_full_path,
    #              feat_dist_save_path = feat_dist_save_path, sample_label_path = sample_label_path,
    #              sele_num = sele_num, clus_num = clus_num, clus_name_list = clus_name_list,
    #              class_num = class_num)


    get_feat_space(feats_full_path = feats_full_path, feat_space_root_path = feat_space_root_path,
                   sample_label_path = sample_label_path, sele_num = sele_num, clus_num = clus_num,
                   class_num = class_num, model_name = 'RRTMIL')


