############################# get preditions function ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: get predition results

import numpy as np
import pandas as pd
import os

def get_predictions(read_path = None, save_path = None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    name_list = os.listdir(read_path)

    for name_i in name_list:
        file_i = pd.read_excel(os.path.join(read_path, name_i))
        file_i = file_i.T
        file_i.to_excel(os.path.join(save_path, name_i), index=False)
        print(os.path.join(save_path, name_i))


if __name__ == '__main__':
    read_path = r'F:\Pytorch_Projects\IGI_PAEC_private\Results_codes\Multi_centre'
    save_path = r'F:\Pytorch_Projects\IGI_PAEC_private\Results_codes\Multi_centre_new'

    get_predictions(read_path = read_path, save_path = save_path)
