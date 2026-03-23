############################# get relation function ##############################
#### Author: Dr. Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: get the relations of model

import numpy as np
import os

def get_relations(relation_list = None, save_path = None):
    new_relat_list = []
    new_label_list = []
    new_dis_list = []
    for relat_list_i in relation_list:
        relat_npy = np.zeros((1, len(relat_list_i[2])))
        for key_i in relat_list_i[0].keys():
            relat_npy[:, relat_list_i[0][key_i]] = int(key_i)
        new_relat_list.append(relat_npy)
        new_label_list.append(relat_list_i[1])
        new_dis_list.append(relat_list_i[2])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import joblib

    joblib.dump(new_relat_list, os.path.join(save_path, 'relations.joblib'))
    joblib.dump(new_label_list, os.path.join(save_path, 'labels.joblib'))
    joblib.dump(new_dis_list, os.path.join(save_path, 'distances.joblib'))


if __name__ == '__main__':
    import joblib
    relations_list = joblib.load(r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Results/Relations/relations.joblib')
    get_relations(relation_list = relations_list, save_path = '/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Results/Relations')

