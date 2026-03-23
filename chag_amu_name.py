import torch
import numpy as np
from sympy.physics.units import joule


def rename_dele_amu(files_read_path=None, father_name_order=[1, 0], targ_color = 0, pe_num = 0):
    import os
    import natsort
    father_path_name = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    father_name_order = [1, 0]
    father_path_new_name = [father_path_name[father_name_order[i]] for i in range(len(father_path_name))]

    son_path_name_list = []
    for father_i in father_path_new_name:
        new_path = os.path.join(files_read_path, father_i)
        new_path_list = natsort.natsorted(os.listdir(new_path), alg=natsort.ns.PATH)
        for new_j in new_path_list:
            for all_i in natsort.natsorted(os.listdir((os.path.join(new_path, new_j))), alg=natsort.ns.PATH):
                son_path_name_list.append(os.path.join(new_path, new_j, all_i))

    for folder_i in son_path_name_list:
        file_name_list = natsort.natsorted(os.listdir(folder_i), alg=natsort.ns.PATH)
        for count_i in range(len(file_name_list) - pe_num):
            h = count_i // 31
            w = count_i % 31
            new_name = '0_' + str(h) + '_' + str(w) + '_' + '31x31.jpg'
            print(new_name)
            os.rename(os.path.join(folder_i, file_name_list[count_i]), os.path.join(folder_i, new_name))
            from skimage import io
            img = io.imread(os.path.join(folder_i, new_name))
            if np.mean(img) == targ_color:
                os.remove(os.path.join(folder_i, new_name))

    print(father_path_new_name)
    print(son_path_name_list)

    return son_path_name_list


###找到所有黑色或白色图像，进行删除
def find_all_folder(files_read_path = None):
    import os
    import natsort
    father_path_name = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    father_name_order = [1, 0]
    father_path_new_name = [father_path_name[father_name_order[i]] for i in range(len(father_path_name))]

    son_path_name_list = []
    for father_i in father_path_new_name:
        new_path = os.path.join(files_read_path, father_i)
        new_path_list = natsort.natsorted(os.listdir(new_path), alg=natsort.ns.PATH)
        for new_j in new_path_list:
            for all_i in natsort.natsorted(os.listdir((os.path.join(new_path, new_j))), alg=natsort.ns.PATH):
                #print(os.path.join(new_path[-4:], new_j, all_i))
                son_path_name_list.append(os.path.join(new_path[-4:], new_j, all_i))
    return son_path_name_list


def check_dataset_complete(path_1 = None, path_2 = None):
    path_1_list = find_all_folder(path_1)
    path_2_list = find_all_folder(path_2)
    print(len(path_1_list), len(path_2_list))

    if len(path_1_list) > len(path_2_list):
        for check_i in path_1_list:
            if check_i not in path_2_list:
                print(check_i, 'path_1 is big')
    else:
        for check_i in path_2_list:
            if check_i not in path_1_list:
                print(check_i, 'path_2 is big')



if __name__ == '__main__':

    '''
    import joblib
    files_read_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_private/Datasets/AMU_CSCC/CSCC_WSI_without_PE'
    son_path_name_list = rename_amu(files_read_path=files_read_path, father_name_order=[1, 0])
    print(len(son_path_name_list))
    '''

    check_dataset_complete(path_1='/root/autodl-tmp/GGO_ISDC_public/Datasets/AMU_LSCC/LSCC_WSI_without_PE_public',
                           path_2='/root/autodl-tmp/GGO_ISDC_public/Datasets/AMU_LSCC/LSCC_WSI_without_PE_private')

    files_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/AMU_LSCC/LSCC_WSI_without_PE_public'
    rename_dele_amu(files_read_path=files_read_path, targ_color=0, pe_num = 81)

    ######### reading npz files
    '''
    file_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_private/Results/Features/AMU_LSCC/IGI_PAEC/feats.npz'
    with np.load(file_path) as f:
        for key in f.keys():
            print(key)

    feats_np = np.load(file_path)
    feat_1d = feats_np['feat_1d']
    print(feat_1d.shape)
    '''

    #import joblib
    #xxx = joblib.load(r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Results/Relations/AMU_CSCC/IGI_PAEC/relations.joblib')
    #print(len(xxx))
    #print(xxx[0])
