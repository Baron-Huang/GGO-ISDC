import os
import numpy as np
from skimage import io
from cv2 import imwrite

if __name__ == '__main__':
    neg_dir = os.listdir(r'/data/HP_Projects/DmcMIL/Datasets/BCNB_for_HER2/Test/Negative')
    neg_dir_len = []
    for i in neg_dir:
        neg_dir_len.append(len(os.listdir(r'/data/HP_Projects/DmcMIL/Datasets/BCNB_for_HER2/Test/Negative' + r'//' + i)))
    print(max(neg_dir_len))

    pos_dir = os.listdir(r'/data/HP_Projects/DmcMIL/Datasets/BCNB_for_HER2/Test/Positive')
    pos_dir_len = []
    for i in pos_dir:
        pos_dir_len.append(
            len(os.listdir(r'/data/HP_Projects/DmcMIL/Datasets/BCNB_for_HER2/Test/Positive' + r'//' + i)))
    print(max(pos_dir_len))

    dark_img = np.zeros((96, 96, 3), dtype=float)


    for i in neg_dir:
        add_len = (5928 -
                   len(os.listdir(r'/data/HP_Projects/DmcMIL/Datasets/BCNB_for_HER2/Test/Negative' + r'//' + i)))
        for j in range(add_len):
            kkk_path = r'/data/HP_Projects/DmcMIL/Datasets/BCNB_for_HER2/Test/Negative' + r'//' + i + r'//' + '1_' + str(j) + '.jpg'
            print(kkk_path)
            imwrite(kkk_path, dark_img)



