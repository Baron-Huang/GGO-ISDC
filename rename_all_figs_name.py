import os
import natsort


if __name__ == '__main__':
    target_name = r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/SefMIC/Datasets/DHMC_Kidney_new'
    target_name_list = natsort.natsorted(os.listdir(target_name), alg=natsort.ns.PATH)
    print(target_name_list)
    new_name = ['8_1.jpg', '8_2.jpg', '8_3.jpg', '8_4.jpg','8_5.jpg']
    for i in target_name_list:
        for j in natsort.natsorted(os.listdir(target_name + r'/' + i), alg=natsort.ns.PATH):
            for k in natsort.natsorted(os.listdir(target_name + r'/' + i + '/' + j), alg=natsort.ns.PATH):
                count = 0
                for img_name in natsort.natsorted(os.listdir(target_name + r'/' + i + '/' + j  + r'/' + k), alg=natsort.ns.PATH):
                    print(img_name)
                    os.rename(target_name + r'/' + i + '/' + j  + r'/' + k + r'/' + img_name,
                                         target_name + r'/' + i + '/' + j  + r'/' + k + r'/' + new_name[count])
                    print(target_name + r'/' + i + '/' + j  + r'/' + k + r'/' + new_name[count])
                    count = count + 1
                    print(count)
