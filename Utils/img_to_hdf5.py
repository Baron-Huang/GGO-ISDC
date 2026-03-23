import shutil
import h5py as h5
from PIL import Image
import os
import natsort
import matplotlib.pyplot as plt
import torch
import numpy as np

def imgs_to_hdf5(root_path = r'/home/dataset-hpfs-0/Datasets/CAMELYON16_SefMIC/Test/Tumor', img_size = [224, 224, 3],
                 save_path = r'/home/dataset-hpfs-0/Datasets/CAMEL YON16_HDF5/Test/Tumor'):

    dir_name_list = natsort.natsorted(os.listdir(root_path), alg=natsort.ns.PATH)

    for img_path_i in dir_name_list:
        img_name_list = natsort.natsorted(os.listdir(root_path + r'/' + img_path_i), alg=natsort.ns.PATH)
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        with h5.File(save_path + r'/' + img_path_i + '.hdf5', 'w') as f:
            dataset = f.create_dataset(img_path_i, (len(img_name_list), img_size[0], img_size[1], img_size[2]),
                                       dtype='i')
            for samp_i in range(len(img_name_list)):
                dataset[samp_i] = Image.open(root_path + r'/' + img_path_i + r'/' + img_name_list[samp_i])
                print(img_name_list[samp_i])

def hdf5_to_imgs(read_path = r'/home/dataset-hpfs-0/Datasets/CAMELYON16_HDF5/Train/Normal/normal_003_1.hdf5'):
    with h5.File(read_path, 'r+') as f:
        read_dataset_name = [key for key in f.keys()]
        print(read_dataset_name[0])
        dataset = f[read_dataset_name[0]];
        data_list = dataset[:]

    data_tensor = torch.tensor(data_list).float()
    print(data_tensor.shape)

    x_i = data_tensor[2].numpy()
    plt.imshow(x_i.astype(np.uint8))
    plt.show()


if __name__ == '__main__':
    imgs_to_hdf5(root_path=r'/root/autodl-tmp/Datasets/DHMC_Lung/Test/acinar', img_size=[224, 224, 3],
                 save_path=r'/root/autodl-tmp/Datasets/DHMC_Lung_HDF5/Test/acinar')
    #hdf5_to_imgs(read_path=r'/root/autodl-tmp/Datasets/CAMELYON16_HDF5/Train/Normal/normal_003_1.hdf5')