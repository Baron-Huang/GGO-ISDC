from torch.utils.data import Dataset, DataLoader
import natsort
import torch
import os
import h5py as h5
import time
from Utils.Read_MIL_Datasets import Read_MIL_Datasets

########################## Read_MILDats_HDF5 #########################
class Read_MILDats_HDF5(Dataset):
    def __init__(self, root_path = None):
        super(Read_MILDats_HDF5, self).__init__()
        self.root_path = root_path

    def __create_label_hdf5__(self):
        img_hdf5_name_list = []
        label_count = 0; img_label_list = torch.zeros(1, dtype=torch.int64)
        for name_i in natsort.natsorted(os.listdir(self.root_path), alg=natsort.ns.PATH):
            img_list_i = natsort.natsorted(os.listdir(self.root_path + r'/' + name_i), alg=natsort.ns.PATH)
            img_list_i = [self.root_path + '/' + name_i + '/' + new_i for new_i in img_list_i]
            img_label_i = torch.zeros(len(img_list_i), dtype=torch.int64) + label_count
            img_label_list = torch.cat((img_label_list, img_label_i))
            img_hdf5_name_list += img_list_i
            label_count += 1

        return img_hdf5_name_list, img_label_list

    def __hdf5_to_imgs__(self, read_path = None):
        with h5.File(read_path, 'r+') as f:
            read_dataset_name = [key for key in f.keys()]; #print(read_dataset_name[0])
            dataset = f[read_dataset_name[0]]; data_list = dataset[:]

        data_tensor = torch.tensor(data_list).float()
        return data_tensor
        #print(data_tensor.shape)
        #x_i = data_tensor[2].numpy(); plt.imshow(x_i.astype(np.uint8)); plt.show()

    def __getitem__(self, item):
        img_hdf5_name_list, img_label_list = self.__create_label_hdf5__()
        img_hdf5 = self.__hdf5_to_imgs__(read_path=img_hdf5_name_list[item]).permute(0, 3, 1, 2)
        img_label = img_label_list[item]
        return img_hdf5, img_label

    def __len__(self):
        item_list, _ = self.__create_label_hdf5__()
        return len(item_list)


if __name__ == '__main__':
    hd_dataset = Read_MILDats_HDF5(root_path='/root/autodl-tmp/Datasets/DHMC_Lung_HDF5/Train')
    img_hdf5_name_list, img_label = hd_dataset.__create_label_hdf5__()
    print(img_hdf5_name_list); print(img_label)

    train_dataset = Read_MIL_Datasets(read_path='/root/autodl-tmp/Datasets/DHMC_Lung/Train',
                                      img_size=[224, 224], bags_len=5000)

    train_loader = DataLoader(hd_dataset, batch_size=1, shuffle=True, num_workers=16,
               timeout=0)

    t_1 = time.time()
    for img_i, label_i in train_dataset:
        print(img_i.shape, label_i)

    t_2 = time.time()
    print(t_2 - t_1)