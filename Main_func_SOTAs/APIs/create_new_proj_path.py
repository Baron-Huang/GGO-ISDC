import os
import natsort


if __name__ == '__main__':
    path_name = r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/SefMIC/Datasets/CAMELYON16_400_224'
    target_name = r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/SefMIC/Datasets/CAMELYON16_Guding'
    path_name_list = natsort.natsorted(os.listdir(path_name), alg=natsort.ns.PATH)
    print(path_name_list)
    for i in path_name_list:
        for j in os.listdir(path_name + r'/' + i):
            for k in os.listdir(path_name + r'/' + i + '/' + j):
                print(k)
                os.makedirs(target_name + r'/' + i + r'/'+ j + '/' + k, exist_ok=True)

